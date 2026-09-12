"""Background Hardware Proxy Generation Engine for GoatEdit.

Why we need this:
High-resolution 4K/8K camera footage causes browser video decoders to lag, drop frames,
and stutter during timeline scrubbing. In professional NLEs (Premiere / DaVinci Resolve),
the editor generates lightweight 720p "editing proxies".

This module uses native GPU hardware encoders (Apple VideoToolbox, NVIDIA NVENC, Intel QSV)
to transcode source media into lightweight 720p proxies in 1-2 seconds. The browser
scrubs the silky-smooth proxy while editing, and the Native Timeline Compiler conforms
back to the full-resolution master files when exporting!
"""

from __future__ import annotations

import hashlib
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
    height: int = 720
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

    def get_proxy_path(self, source_path: str, height: int = 720) -> str:
        """Returns deterministic path for a file's proxy based on content path and mtime."""
        mtime = os.path.getmtime(source_path) if os.path.exists(source_path) else 0
        key = f"{source_path}:{mtime}:{height}"
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        # Clean filename
        clean_name = "".join(c for c in base_name if c.isalnum() or c in ("-", "_"))[:30]
        return os.path.join(self.cache_dir, f"{clean_name}_{h}_{height}p.mp4")

    def create_proxy(self, source_path: str, height: int = 720, codec_id: str = "h264_hw") -> ProxyJob:
        target_path = self.get_proxy_path(source_path, height)

        # If proxy already exists on disk, return instant ready job
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
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

        # Run transcode in background thread
        threading.Thread(target=job.run, daemon=True).start()
        return job

    def get(self, job_id: str) -> ProxyJob | None:
        with self._lock:
            return self._jobs.get(job_id)


# Global singleton
proxy_manager = ProxyManager()
