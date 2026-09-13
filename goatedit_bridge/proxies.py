"""Background Hardware Proxy Generation Engine for GoatEdit.

Why we need this:
High-resolution 4K/8K camera footage causes browser video decoders to lag, drop frames,
and stutter during timeline scrubbing. In professional NLEs (Premiere / DaVinci Resolve),
the editor generates lightweight low-resolution "editing proxies".

This module uses native GPU hardware encoders (Apple VideoToolbox, NVIDIA NVENC, Intel QSV)
to transcode source media into lightweight 540p proxies. The browser
scrubs the silky-smooth proxy while editing, and the Native Timeline Compiler conforms
back to the full-resolution master files when exporting!
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .render import probe_hardware


@dataclass
class ProxyJob:
    """A background proxy generation task."""
    job_id: str
    source_path: str
    target_path: str
    height: int = 540
    codec_id: str = "h264_hw"
    state: str = "queued"  # queued | transcoding | ready | error
    progress: float = 0.0
    error: str | None = None
    file_size_bytes: int = 0
    created_at: float = field(default_factory=time.time)

    _process: subprocess.Popen | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def run(self) -> None:
        with self._lock:
            if self.state != "queued":
                return
            self.state = "transcoding"

        try:
            if not os.path.exists(self.source_path):
                raise FileNotFoundError(f"Source file not found: {self.source_path}")

            os.makedirs(os.path.dirname(os.path.abspath(self.target_path)), exist_ok=True)

            hw_info = probe_hardware()
            codec_map = {c["id"]: c for c in hw_info.get("codecs", [])}
            preset = codec_map.get(self.codec_id)
            encoder = preset["encoder"] if preset else "libx264"

            cmd = ["ffmpeg", "-y"]

            # Hardware decoding
            if hw_info.get("hasHardwareAcceleration"):
                if "videotoolbox" in str(hw_info.get("codecs", [])):
                    cmd.extend(["-hwaccel", "videotoolbox"])
                elif "nvenc" in str(hw_info.get("codecs", [])):
                    cmd.extend(["-hwaccel", "cuda"])

            cmd.extend([
                "-i", self.source_path,
                "-vf", f"scale=-2:{self.height}",
                "-c:v", encoder,
                "-pix_fmt", "yuv420p",
                "-b:v", "2500k",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                self.target_path,
            ])

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = self._process.communicate()

            with self._lock:
                if self._process.returncode == 0 and os.path.exists(self.target_path):
                    self.state = "ready"
                    self.progress = 100.0
                    self.file_size_bytes = os.path.getsize(self.target_path)
                else:
                    self.state = "error"
                    self.error = f"FFmpeg proxy transcode failed: {stderr.decode('utf-8', errors='replace')[-500:]}"
        except Exception as exc:
            with self._lock:
                self.state = "error"
                self.error = str(exc)

    def public(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "sourcePath": self.source_path,
            "targetPath": self.target_path,
            "height": self.height,
            "state": self.state,
            "progress": self.progress,
            "fileSizeBytes": self.file_size_bytes,
            "error": self.error,
        }


class ProxyManager:
    """Coordinates proxy creation jobs and maintains disk cache."""

    def __init__(self, cache_dir: str | None = None) -> None:
        if not cache_dir:
            home = os.path.expanduser("~")
            cache_dir = os.path.join(home, ".cache", "goatedit-bridge", "proxies")
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self._jobs: dict[str, ProxyJob] = {}
        self._lock = threading.Lock()
        # Job ids outlive this process. A proxy URL handed to the editor is
        # stored in the project, so it has to survive a bridge restart — without
        # this index every saved proxyUrl 404s the next time the bridge starts,
        # with the proxy sitting on disk the whole time.
        self._index_path = os.path.join(self.cache_dir, "proxy-index.json")
        self._load_index()

    def _load_index(self) -> None:
        """Re-registers proxies produced by earlier runs that are still on disk."""
        try:
            with open(self._index_path, encoding="utf-8") as fh:
                entries = json.load(fh)
        except (OSError, ValueError):
            return
        if not isinstance(entries, dict):
            return
        for job_id, rec in entries.items():
            target = rec.get("targetPath", "") if isinstance(rec, dict) else ""
            if not target or not os.path.isfile(target):
                continue  # deleted since; a fresh request will rebuild it
            self._jobs[job_id] = ProxyJob(
                job_id=job_id,
                source_path=rec.get("sourcePath", ""),
                target_path=target,
                height=int(rec.get("height") or 540),
                state="ready",
                progress=100.0,
                file_size_bytes=os.path.getsize(target),
            )

    def _save_index(self) -> None:
        """Writes the id-to-path map. Best effort: losing it costs a re-transcode,
        never correctness."""
        with self._lock:
            entries = {
                job_id: {
                    "targetPath": job.target_path,
                    "sourcePath": job.source_path,
                    "height": job.height,
                }
                for job_id, job in self._jobs.items()
                if job.state == "ready" and job.target_path
            }
        try:
            tmp = f"{self._index_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(entries, fh, indent=1)
            os.replace(tmp, self._index_path)
        except OSError:
            pass

    def get_proxy_path(self, source_path: str, height: int = 540) -> str:
        """Where a file's proxy lives: a `Proxies` folder beside the footage.

        Next to the source rather than in ~/.cache, so the proxies are findable
        and relinkable by hand — the same place Premiere and Resolve put them.
        They keep the source's own name (`B0035_540p.mp4`), which is what makes
        a manual relink possible at all; a content hash in the filename would
        make every one of them unidentifiable.

        Freshness is therefore checked against mtime at use, not baked into the
        name — see `_is_stale`.

        Falls back to the cache directory when the footage sits somewhere we
        cannot write: a read-only volume, a mounted card, someone else's share.
        """
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        clean_name = "".join(c for c in base_name if c.isalnum() or c in ("-", "_", "."))[:60]
        file_name = f"{clean_name}_{height}p.mp4"

        source_dir = os.path.dirname(os.path.abspath(source_path))
        if source_dir and os.path.isdir(source_dir) and os.access(source_dir, os.W_OK):
            return os.path.join(source_dir, PROXY_DIR_NAME, file_name)

        # Unwritable source location — keep the deterministic cache name there,
        # including the path hash, because two cards can both hold a B0035.MP4.
        h = hashlib.sha256(f"{source_path}:{height}".encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{clean_name}_{h}_{height}p.mp4")

    @staticmethod
    def _is_stale(source_path: str, target_path: str) -> bool:
        """Whether an existing proxy predates its source and must be rebuilt.

        The old naming scheme hashed the source mtime into the filename, so a
        re-recorded or re-exported source simply produced a different name. A
        stable, human-readable name gives that up, so the check moves here.
        """
        try:
            return os.path.getmtime(target_path) < os.path.getmtime(source_path)
        except OSError:
            return False

    def create_proxy(self, source_path: str, height: int = 540, codec_id: str = "h264_hw") -> ProxyJob:
        target_path = self.get_proxy_path(source_path, height)

        # Reuse what is already on disk, unless the source has changed since.
        if (
            os.path.exists(target_path)
            and os.path.getsize(target_path) > 0
            and not self._is_stale(source_path, target_path)
        ):
            job = ProxyJob(
                job_id=hashlib.md5(target_path.encode()).hexdigest(),
                source_path=source_path,
                target_path=target_path,
                height=height,
                codec_id=codec_id,
                state="ready",
                progress=100.0,
                file_size_bytes=os.path.getsize(target_path),
            )
            with self._lock:
                self._jobs[job.job_id] = job
            self._save_index()
            return job

        job_id = hashlib.md5(target_path.encode()).hexdigest()
        job = ProxyJob(
            job_id=job_id,
            source_path=source_path,
            target_path=target_path,
            height=height,
            codec_id=codec_id,
        )
        with self._lock:
            self._jobs[job_id] = job

        # Run the transcode, then record it so its id survives a restart.
        def _run_and_index() -> None:
            job.run()
            if job.state == "ready":
                self._save_index()

        threading.Thread(target=_run_and_index, daemon=True).start()
        return job

    def get(self, job_id: str) -> ProxyJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def is_known_target(self, path: str) -> bool:
        """Whether this exact file is one we produced.

        Proxies now live beside the footage, so "is it under the cache dir?" no
        longer answers "are we allowed to serve it?". Matching against the
        targets we actually created keeps that answer narrow: this authorises
        one file per proxy job, never a directory, so pointing the bridge at a
        media folder does not turn it into a file server for that folder.
        """
        try:
            wanted = os.path.realpath(path)
        except OSError:
            return False
        with self._lock:
            targets = [j.target_path for j in self._jobs.values() if j.target_path]
        for target in targets:
            try:
                if os.path.realpath(target) == wanted:
                    return True
            except OSError:
                continue
        return False


# Global singleton
proxy_manager = ProxyManager()


# --- deciding whether a source needs a proxy at all -------------------------
#
# Browsers decode video through the platform's hardware decoder or not at all —
# there is no fast software fallback in a tab. On macOS, VideoToolbox decodes
# H.264 in hardware ONLY for 8-bit 4:2:0. Camera formats like Sony XAVC-I
# (H.264 High 4:2:2, 10-bit, ~200 Mbps) have no hardware path on ANY Mac, so
# Chrome decodes them entirely in software: measured at ~150-180ms per 2160p
# frame against single-digit ms for a format it can offload. That is not a
# playback that degrades, it is a playback that never worked.
#
# So proxying is not about resolution. A 4K 8-bit 4:2:0 clip plays fine; a
# 1080p 4:2:2 10-bit one does not. Deciding on width alone proxies the wrong
# files and misses the ones that matter.

# Pixel formats the hardware decoders actually accept.
HARDWARE_DECODABLE_PIX_FMTS = frozenset({
    "yuv420p", "yuvj420p", "nv12", "yuv420p10le",  # 10-bit 4:2:0 is fine for HEVC/AV1
})

# Codecs a current browser can offload at all.
HARDWARE_DECODABLE_CODECS = frozenset({"h264", "hevc", "vp9", "av1"})

# Above this, decode plus the texture upload costs more than the proxy saves,
# even when the format itself is offloadable.
PROXY_ABOVE_WIDTH = 2560

# Folder created beside the footage to hold its proxies. Named, not hidden, so
# it is obvious what it is and safe to delete.
PROXY_DIR_NAME = "Proxies"


def probe_decode_profile(source_path: str) -> dict[str, Any]:
    """Video codec, pixel format, profile and size of a file — the fields that
    decide whether a browser can hardware-decode it."""
    info: dict[str, Any] = {
        "codec": "", "pix_fmt": "", "profile": "",
        "width": 0, "height": 0, "fps": 0.0, "bit_rate": 0,
    }
    if not source_path or not os.path.exists(source_path):
        return info
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt,profile,width,height,r_frame_rate,bit_rate",
             "-of", "json", source_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
        )
        if res.returncode != 0:
            return info
        streams = json.loads(res.stdout).get("streams", [])
        if not streams:
            return info
        s = streams[0]
        info["codec"] = s.get("codec_name", "") or ""
        info["pix_fmt"] = s.get("pix_fmt", "") or ""
        info["profile"] = s.get("profile", "") or ""
        info["width"] = int(s.get("width") or 0)
        info["height"] = int(s.get("height") or 0)
        info["bit_rate"] = int(s.get("bit_rate") or 0)
        # r_frame_rate is a rational string like "50/1".
        rate = s.get("r_frame_rate") or ""
        if "/" in rate:
            num, _, den = rate.partition("/")
            try:
                info["fps"] = round(float(num) / float(den), 3) if float(den) else 0.0
            except (ValueError, ZeroDivisionError):
                info["fps"] = 0.0
    except Exception:
        pass
    return info


def needs_proxy(source_path: str) -> dict[str, Any]:
    """Whether this file should be proxied before a browser tries to play it.

    Returns the decision, a human-readable reason, and the probe it rested on,
    so the caller can show the user WHY a transcode is about to happen.
    """
    probe = probe_decode_profile(source_path)
    codec = probe["codec"]
    pix_fmt = probe["pix_fmt"]
    profile = (probe["profile"] or "").lower()

    if not codec:
        return {"needed": False, "reason": "could not probe the file", "probe": probe}

    if codec not in HARDWARE_DECODABLE_CODECS:
        return {"needed": True,
                "reason": f"codec {codec} has no browser hardware decoder",
                "probe": probe}

    # 4:2:2 and 4:4:4 sampling is the common case that looks fine in a probe and
    # then decodes on the CPU. Checked by profile as well as pix_fmt because
    # some builds report one and not the other.
    if "4:2:2" in profile or "4:4:4" in profile:
        return {"needed": True,
                "reason": f"{codec} {probe['profile']} is chroma-subsampled beyond 4:2:0 — no hardware decoder exists",
                "probe": probe}

    if pix_fmt and pix_fmt not in HARDWARE_DECODABLE_PIX_FMTS:
        return {"needed": True,
                "reason": f"pixel format {pix_fmt} has no hardware decoder",
                "probe": probe}

    # 10-bit H.264 has no hardware path even at 4:2:0.
    if codec == "h264" and "10" in pix_fmt:
        return {"needed": True,
                "reason": "10-bit H.264 has no hardware decoder",
                "probe": probe}

    if probe["width"] > PROXY_ABOVE_WIDTH:
        return {"needed": True,
                "reason": f"{probe['width']}px wide — decode and upload cost more than the proxy",
                "probe": probe}

    return {"needed": False, "reason": "the browser can hardware-decode this as-is", "probe": probe}
