"""Download jobs: yt-dlp runs here, on a worker thread, one job per request."""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import yt_dlp

# The plugin polls /job/<id>, so a job id ends up in a URL path. Only hex.
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Finished files are kept so the editor can pull them, then reaped. Two hours
# is long enough for a user who wandered off mid-download.
JOB_TTL_SECONDS = 2 * 60 * 60

_EXT_TO_TYPE = {
    "mp4": "video", "mkv": "video", "webm": "video", "mov": "video",
    "m4a": "audio", "mp3": "audio", "opus": "audio", "wav": "audio", "flac": "audio",
}


@dataclass
class Job:
    id: str
    url: str
    state: str = "queued"          # queued | downloading | processing | ready | error
    progress: float = 0.0          # 0..100
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None
    filepath: str | None = None
    media: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def public(self) -> dict[str, Any]:
        return {
            "jobId": self.id,
            "state": self.state,
            "progress": round(self.progress, 1),
            "downloadedBytes": self.downloaded_bytes,
            "totalBytes": self.total_bytes,
            "error": self.error,
            "media": self.media or None,
        }


def ffmpeg_available() -> bool:
    """yt-dlp needs ffmpeg to mux the separate 1080p+ video and audio streams."""
    return shutil.which("ffmpeg") is not None


def _base_opts() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Matches api/resolve-video.py: these clients serve the full format
        # ladder without the "sign in to confirm you're not a bot" gate.
        "extractor_args": {"youtube": {"player_client": ["android_vr", "android", "tv", "ios"]}},
    }


def probe(url: str) -> dict[str, Any]:
    """Metadata + selectable formats for one URL. No bytes are fetched."""
    opts = _base_opts() | {"skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    heights: dict[int, dict[str, Any]] = {}
    for f in info.get("formats") or []:
        height = f.get("height")
        if not height or f.get("vcodec") in (None, "none"):
            continue
        best = heights.get(height)
        if not best or (f.get("tbr") or 0) > (best.get("tbr") or 0):
            heights[height] = f

    formats = [
        {
            "formatId": f["format_id"],
            "label": f"{h}p60" if (f.get("fps") or 0) > 45 else f"{h}p",
            "height": h,
            "fps": int(f.get("fps") or 0),
            "ext": f.get("ext") or "mp4",
            "hasAudio": f.get("acodec") not in (None, "none"),
            "sizeBytes": int(f.get("filesize") or f.get("filesize_approx") or 0),
        }
        for h, f in sorted(heights.items(), key=lambda kv: kv[0], reverse=True)
    ]

    thumbnails = info.get("thumbnails") or []
    return {
        "title": info.get("title") or "Video",
        "duration": int(info.get("duration") or 0),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "thumbnail": thumbnails[-1]["url"] if thumbnails else (info.get("thumbnail") or ""),
        "formats": formats,
        "audioOnly": True,
        "ffmpeg": ffmpeg_available(),
    }


def _format_selector(format_id: str | None, audio_only: bool) -> str:
    if audio_only:
        return "ba[ext=m4a]/ba/b"
    if format_id:
        # Pair the chosen video-only rendition with the best m4a; yt-dlp falls
        # back to the bare format when it already carries audio.
        return f"{format_id}+ba[ext=m4a]/{format_id}"
    if ffmpeg_available():
        return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
    # No ffmpeg means no muxing, so only already-combined streams will play.
    return "b[ext=mp4]/b"


class JobStore:
    """Every job this bridge has run, plus the directory their files live in.

    `keep_files` is on when the user pointed `--dir` at a folder of their own:
    the files are theirs now, named after the video, and neither the reaper nor
    shutdown may touch them. Without it the dir is a temp one we made and wipe.
    """

    def __init__(self, work_dir: str, keep_files: bool = False) -> None:
        self.work_dir = work_dir
        self.keep_files = keep_files
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, url: str, format_id: str | None, audio_only: bool) -> Job:
        self._reap()
        job = Job(id=uuid.uuid4().hex, url=url)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run, args=(job, format_id, audio_only), daemon=True,
        ).start()
        return job

    def _reap(self) -> None:
        """Drops jobs past their TTL and deletes the files they were holding."""
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            stale = [j for j in self._jobs.values() if j.created_at < cutoff]
            for job in stale:
                self._jobs.pop(job.id, None)
        if self.keep_files:
            # The job record expires so /file/<id> stops answering, but the file
            # itself is in a folder the user chose. Deleting it would be theft.
            return
        for job in stale:
            if job.filepath and os.path.exists(job.filepath):
                try:
                    os.remove(job.filepath)
                except OSError:
                    pass

    def _run(self, job: Job, format_id: str | None, audio_only: bool) -> None:
        def hook(d: dict[str, Any]) -> None:
            if d.get("status") == "downloading":
                job.state = "downloading"
                job.downloaded_bytes = int(d.get("downloaded_bytes") or 0)
                job.total_bytes = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
                if job.total_bytes:
                    job.progress = min(99.0, job.downloaded_bytes / job.total_bytes * 100)
            elif d.get("status") == "finished":
                # Merging/remuxing happens after the last stream lands.
                job.state = "processing"
                job.progress = 99.0

        # In a temp dir the name is throwaway, so the job id keeps it collision-free.
        # In the user's own folder they want to recognise what they downloaded;
        # the video id keeps two different videos of the same name apart, and
        # yt-dlp sanitises both fields, so the result is always a single
        # filename directly inside work_dir.
        name_tmpl = "%(title)s [%(id)s].%(ext)s" if self.keep_files else f"{job.id}.%(ext)s"
        opts = _base_opts() | {
            "format": _format_selector(format_id, audio_only),
            "outtmpl": os.path.join(self.work_dir, name_tmpl),
            "trim_file_name": 120,
            "progress_hooks": [hook],
            # A job is one file; a URL that expands to a playlist is user error.
            "playlist_items": "1",
        }
        if not audio_only and ffmpeg_available():
            opts["merge_output_format"] = "mp4"

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(job.url, download=True)
            requested = (info.get("requested_downloads") or [{}])[0]
            path = requested.get("filepath") or ydl.prepare_filename(info)
            if not path or not os.path.exists(path):
                raise RuntimeError("yt-dlp finished but produced no file")

            ext = os.path.splitext(path)[1].lstrip(".").lower()
            job.filepath = path
            job.media = {
                "name": info.get("title") or os.path.basename(path),
                "type": "audio" if audio_only else _EXT_TO_TYPE.get(ext, "video"),
                "ext": ext,
                "sizeBytes": os.path.getsize(path),
                "duration": float(info.get("duration") or 0),
                "width": int(info.get("width") or 0),
                "height": int(info.get("height") or 0),
                "fps": float(info.get("fps") or 0),
            }
            job.progress = 100.0
            job.state = "ready"
        except Exception as exc:  # noqa: BLE001 — the panel shows whatever went wrong
            job.error = str(exc)[:500]
            job.state = "error"
