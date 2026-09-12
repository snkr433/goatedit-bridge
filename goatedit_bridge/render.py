"""Hardware-accelerated video rendering engine for GoatEdit.

This module acts as the "Media Encoder" for GoatEdit:
1. Detects local GPU encoders (Apple VideoToolbox, NVIDIA NVENC, Intel QSV).
2. Spawns and manages native FFmpeg encoding pipelines.
3. Consumes frame streams and audio from the GoatEdit web application.
4. Directly writes broadcast-quality master files (ProRes, HEVC, H.264) to local disk.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


BANNER_PREFIXES: tuple[str, ...] = (
    "ffmpeg version", "built with", "configuration:", "lib", "Input #",
    "Output #", "Duration:", "Stream #", "Metadata:", "encoder", "Press [q]",
    "Stream mapping:", "  ",
)

_ERROR_KEYWORDS = (
    "Error", "error", "Invalid", "Unable", "No such", "Conversion failed",
    "failed", "not supported", "Permission denied", "Is a directory",
)


def summarize_ffmpeg_stderr(lines: "Iterable[str]") -> str:
    """
    The lines from FFmpeg's output that explain a failure, banner dropped.

    FFmpeg says a great deal before it says anything useful, and what matters is
    almost always in the last few lines. Shared with the timeline compiler:
    both watch an FFmpeg process and both have to explain an exit code to
    someone looking at a toast in a browser.
    """
    kept = [ln for ln in lines if ln]
    errors = [ln for ln in kept if any(k in ln for k in _ERROR_KEYWORDS)]
    chosen = errors[-4:] if errors else [
        ln for ln in kept if not ln.startswith(BANNER_PREFIXES)
    ][-4:]
    return " | ".join(chosen)


# Rough encoded size, in megabits per second, at 1920x1080 and 30 fps. Scaled by
# pixel count and frame rate below. These are order-of-magnitude figures whose
# only job is to catch an export that cannot possibly fit.
_CODEC_MBPS_1080P30 = {
    "prores_422_hq": 220.0,
    "prores_422": 147.0,
    "prores_proxy": 45.0,
    "hevc_hw": 15.0,
    "h264_hw": 20.0,
    "h264_cpu": 20.0,
}

_REFERENCE_PIXEL_RATE = 1920 * 1080 * 30


def estimate_output_bytes(
    codec_id: str, width: int, height: int, fps: float, duration_seconds: float
) -> int:
    """Roughly how large this export will be, for a headroom check."""
    mbps = _CODEC_MBPS_1080P30.get(codec_id, 25.0)
    pixel_rate = max(1, width * height) * max(1.0, fps)
    scaled = mbps * (pixel_rate / _REFERENCE_PIXEL_RATE)
    return int((scaled * 1_000_000 / 8) * max(0.0, duration_seconds))


def check_disk_headroom(
    output_path: str, codec_id: str, width: int, height: int, fps: float, duration_seconds: float
) -> None:
    """
    Refuses an export that will not fit, before a frame is encoded.

    Running out of disk halfway leaves a file with no trailer — unplayable, and
    still occupying every byte it managed to write. ProRes is the one that bites:
    422 is about 1.1 GB per minute at 1080p30, so a five minute master needs more
    room than a nearly-full disk has. Raising here costs nothing and the message
    can say what to do about it; the alternative is FFmpeg's ENOSPC after several
    minutes of encoding.
    """
    if duration_seconds <= 0:
        return
    needed = estimate_output_bytes(codec_id, width, height, fps, duration_seconds)
    if needed <= 0:
        return

    target_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    while target_dir and not os.path.isdir(target_dir):
        parent = os.path.dirname(target_dir)
        if parent == target_dir:
            return
        target_dir = parent

    try:
        free = shutil.disk_usage(target_dir).free
    except OSError:
        return

    # A tenth on top, for the container overhead and whatever else the machine
    # is writing while this runs.
    if free >= needed * 1.1:
        return

    gb = 1024 ** 3
    raise RuntimeError(
        f"Not enough disk space for this export. "
        f"{codec_id} at {width}x{height} for {duration_seconds / 60:.1f} min needs about "
        f"{needed / gb:.1f} GB, and {target_dir} has {free / gb:.1f} GB free. "
        f"Free some space, choose a folder on another drive, or export h264_hw / hevc_hw "
        f"instead of ProRes."
    )


def _get_ffmpeg_encoders() -> set[str]:
    """Queries ffmpeg for the list of supported video encoders."""
    if not shutil.which("ffmpeg"):
        return set()
    try:
        # Run ffmpeg -encoders and collect all available encoder names
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        encoders = set()
        for line in res.stdout.splitlines():
            # Format: ' V..... encoder_name description'
            line = line.strip()
            if line.startswith("V") or " V" in line[:6]:
                parts = line.split()
                if len(parts) >= 2:
                    encoders.add(parts[1])
        return encoders
    except Exception:
        return set()


def probe_hardware() -> dict[str, Any]:
    """Inspects the machine's GPU and FFmpeg capabilities to return available hardware profiles."""
    encoders = _get_ffmpeg_encoders()
    has_ffmpeg = bool(encoders)

    sys_platform = platform.system()
    machine = platform.machine()

    gpu_name = "CPU Software Encoding"
    has_hw = False

    # 1. Detect Apple Silicon / macOS VideoToolbox
    if "h264_videotoolbox" in encoders:
        has_hw = True
        if machine in ("arm64", "aarch64"):
            gpu_name = "Apple Silicon GPU (VideoToolbox Media Engine)"
        else:
            gpu_name = "Apple VideoToolbox (Hardware Accelerated)"

    # 2. Detect NVIDIA NVENC
    elif "h264_nvenc" in encoders:
        has_hw = True
        gpu_name = "NVIDIA GPU (NVENC Hardware Acceleration)"

    # 3. Detect Intel QuickSync (QSV)
    elif "h264_qsv" in encoders:
        has_hw = True
        gpu_name = "Intel QuickSync (QSV Hardware Acceleration)"

    # 4. Detect AMD AMF
    elif "h264_amf" in encoders:
        has_hw = True
        gpu_name = "AMD GPU (AMF Hardware Acceleration)"

    # Build available codec presets based on detected encoders
    codecs = []

    # ProRes Master Presets (Apple VideoToolbox or prores_ks)
    if "prores_videotoolbox" in encoders:
        codecs.append({
            "id": "prores_422_hq",
            "name": "Apple ProRes 422 HQ",
            "description": "Industry-standard broadcast master (10-bit 4:2:2, Hardware Accelerated)",
            "container": "mov",
            "encoder": "prores_videotoolbox",
            "isHardware": True,
            "profile": "3",
        })
        codecs.append({
            "id": "prores_422",
            "name": "Apple ProRes 422",
            "description": "Standard editing master (10-bit 4:2:2, Hardware Accelerated)",
            "container": "mov",
            "encoder": "prores_videotoolbox",
            "isHardware": True,
            "profile": "2",
        })
        codecs.append({
            "id": "prores_proxy",
            "name": "Apple ProRes Proxy",
            "description": "High-speed lightweight proxy master",
            "container": "mov",
            "encoder": "prores_videotoolbox",
            "isHardware": True,
            "profile": "0",
        })
    elif "prores_ks" in encoders:
        codecs.append({
            "id": "prores_422_hq",
            "name": "Apple ProRes 422 HQ (CPU)",
            "description": "Industry-standard broadcast master (10-bit 4:2:2)",
            "container": "mov",
            "encoder": "prores_ks",
            "isHardware": False,
            "profile": "3",
        })

    # HEVC / H.265 (High efficiency 4K/8K 10-bit)
    if "hevc_videotoolbox" in encoders:
        codecs.append({
            "id": "hevc_hw",
            "name": "H.265 / HEVC (Apple VideoToolbox)",
            "description": "Ultra HD 10-bit hardware accelerated video",
            "container": "mp4",
            "encoder": "hevc_videotoolbox",
            "isHardware": True,
        })
    elif "hevc_nvenc" in encoders:
        codecs.append({
            "id": "hevc_hw",
            "name": "H.265 / HEVC (NVIDIA NVENC)",
            "description": "Ultra HD 10-bit hardware accelerated video",
            "container": "mp4",
            "encoder": "hevc_nvenc",
            "isHardware": True,
        })
    elif "hevc_qsv" in encoders:
        codecs.append({
            "id": "hevc_hw",
            "name": "H.265 / HEVC (Intel QuickSync)",
            "description": "Ultra HD hardware accelerated video",
            "container": "mp4",
            "encoder": "hevc_qsv",
            "isHardware": True,
        })

    # H.264 (Maximum compatibility)
    if "h264_videotoolbox" in encoders:
        codecs.append({
            "id": "h264_hw",
            "name": "H.264 (Apple VideoToolbox)",
            "description": "High-speed GPU hardware encoding",
            "container": "mp4",
            "encoder": "h264_videotoolbox",
            "isHardware": True,
        })
    elif "h264_nvenc" in encoders:
        codecs.append({
            "id": "h264_hw",
            "name": "H.264 (NVIDIA NVENC)",
            "description": "High-speed GPU hardware encoding",
            "container": "mp4",
            "encoder": "h264_nvenc",
            "isHardware": True,
        })
    elif "h264_qsv" in encoders:
        codecs.append({
            "id": "h264_hw",
            "name": "H.264 (Intel QuickSync)",
            "description": "High-speed GPU hardware encoding",
            "container": "mp4",
            "encoder": "h264_qsv",
            "isHardware": True,
        })

    # Software Fallback H.264
    if "libx264" in encoders:
        codecs.append({
            "id": "h264_cpu",
            "name": "H.264 Master (libx264)",
            "description": "High-quality multi-threaded CPU encoder",
            "container": "mp4",
            "encoder": "libx264",
            "isHardware": False,
        })

    # Default export directory on local disk
    home = os.path.expanduser("~")
    movies_dir = os.path.join(home, "Movies", "GoatEdit")
    if not os.path.exists(os.path.join(home, "Movies")):
        movies_dir = os.path.join(home, "Downloads", "GoatEdit")

    return {
        "available": has_ffmpeg,
        "platform": sys_platform,
        "machine": machine,
        "gpuName": gpu_name,
        "hasHardwareAcceleration": has_hw,
        "codecs": codecs,
        "defaultExportDir": movies_dir,
    }


@dataclass
class RenderSession:
    """Represents an active or completed hardware render pipeline."""
    session_id: str
    width: int
    height: int
    fps: float
    total_frames: int
    codec_id: str
    output_path: str
    audio_path: str | None = None
    bitrate: str | None = None
    # What arrives on stdin. "rawvideo" is uncompressed RGBA, one frame per push,
    # and FFmpeg encodes it here. "h264"/"hevc" is an Annex-B elementary stream
    # the browser already encoded with WebCodecs — which on every platform we
    # support is the same hardware encoder FFmpeg would have reached for — so
    # FFmpeg only muxes. That path moves roughly a hundredth of the bytes and
    # skips a full GPU-to-CPU readback per frame in the tab.
    stream_format: str = "rawvideo"

    state: str = "initialized"  # initialized | rendering | finalizing | ready | error | cancelled
    frames_written: int = 0
    encoded_fps: float = 0.0
    progress: float = 0.0
    error: str | None = None
    encoder_used: str | None = None
    fallback_reason: str | None = None
    created_at: float = field(default_factory=time.time)

    # Internal process handles
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _stderr_thread: threading.Thread | None = field(default=None, repr=False)
    # FFmpeg explains itself on stderr and then dies; without the last lines all
    # the caller ever sees is the EPIPE from the next frame it tried to push.
    _stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=40), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _build_command(self, preset: dict[str, Any], encoder: str) -> list[str]:
        """Assembles the FFmpeg invocation for one encoder choice."""
        if self.stream_format != "rawvideo":
            # An elementary stream carries no timestamps of its own, so the input
            # rate is what gives the muxer a timebase. Constant, which is what an
            # export is.
            cmd = [
                "ffmpeg", "-y",
                "-f", self.stream_format,
                "-r", str(self.fps),
                "-i", "-",
            ]
            if self.audio_path and os.path.exists(self.audio_path):
                cmd.extend(["-i", self.audio_path])
            cmd.extend(["-c:v", "copy"])
            if self.stream_format == "hevc":
                # Without hvc1 the track is tagged hev1, which QuickTime and
                # Finder preview refuse to open even though the bytes are fine.
                cmd.extend(["-tag:v", "hvc1"])
            if self.audio_path:
                cmd.extend(["-c:a", "aac", "-b:a", "256k"])
            cmd.extend(["-movflags", "+faststart", self.output_path])
            return cmd

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",  # Video from stdin pipe
        ]

        # If audio file is provided, add as second input
        if self.audio_path and os.path.exists(self.audio_path):
            cmd.extend(["-i", self.audio_path])

        # Codec-specific arguments
        if "prores" in encoder:
            cmd.extend([
                "-c:v", encoder,
                "-profile:v", preset.get("profile", "3"),
                "-pix_fmt", "yuv422p10le",
            ])
            if self.audio_path:
                cmd.extend(["-c:a", "pcm_s24le"])
        else:
            # yuv420p cannot represent an odd width or height, and every H.264 /
            # HEVC encoder refuses the stream outright rather than rounding — the
            # process dies before the first frame lands and the browser only sees
            # a broken pipe. Round down to even here instead.
            if self.width % 2 or self.height % 2:
                cmd.extend(["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
            cmd.extend([
                "-c:v", encoder,
                "-pix_fmt", "yuv420p",
            ])
            if self.bitrate:
                cmd.extend(["-b:v", self.bitrate])
            if self.audio_path:
                cmd.extend(["-c:a", "aac", "-b:a", "256k"])

        cmd.extend([
            "-movflags", "+faststart",
            self.output_path,
        ])
        return cmd

    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        """Launches FFmpeg and gives it a moment to reject the arguments."""
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        # An encoder that refuses its configuration exits within milliseconds.
        # Catching that here turns "Broken pipe" into FFmpeg's own explanation.
        try:
            proc.wait(timeout=0.35)
        except subprocess.TimeoutExpired:
            return proc
        return proc

    def _drain_stderr(self, proc: subprocess.Popen) -> str:
        try:
            if proc.stderr:
                return proc.stderr.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        return ""

    def start(self) -> None:
        """Constructs FFmpeg pipeline and launches subprocess with stdin pipe for raw frames."""
        with self._lock:
            if self.state != "initialized":
                raise RuntimeError(f"Session already in state: {self.state}")

            if self.stream_format == "rawvideo":
                hw_info = probe_hardware()
                codec_map = {c["id"]: c for c in hw_info["codecs"]}
                preset = codec_map.get(self.codec_id)
                if not preset:
                    raise ValueError(f"Unsupported codec id: {self.codec_id}")
            else:
                # There is no encoder to pick. Probing for VideoToolbox here would
                # decide nothing and could refuse a mux that was going to work.
                preset = {}

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            check_disk_headroom(
                self.output_path, self.codec_id, self.width, self.height, self.fps,
                self.total_frames / self.fps if self.fps > 0 else 0.0,
            )

            encoder = preset.get("encoder", "copy")
            proc = self._spawn(self._build_command(preset, encoder))

            if proc.poll() is not None:
                first_error = self._drain_stderr(proc)
                # A GPU encoder can be present in `-encoders` and still refuse the
                # job (driver asleep, session limit, unsupported dimensions). The
                # CPU encoder is slower but always there, so a hardware failure
                # should cost speed, not the export.
                fallback = "libx264" if preset.get("isHardware") and "prores" not in encoder else None
                if fallback and fallback in _get_ffmpeg_encoders():
                    self.fallback_reason = (
                        f"{encoder} failed to start; re-encoding on CPU with {fallback}."
                    )
                    proc = self._spawn(self._build_command(preset, fallback))
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"FFmpeg failed to start ({fallback}): "
                            f"{self._drain_stderr(proc)[-600:] or first_error[-600:]}"
                        )
                    self.encoder_used = fallback
                else:
                    raise RuntimeError(
                        f"FFmpeg failed to start ({encoder}): {first_error[-600:] or 'no output'}"
                    )
            else:
                self.encoder_used = encoder

            self._process = proc
            self.state = "rendering"

            # Monitor stderr in a background thread for live progress parsing
            self._stderr_thread = threading.Thread(target=self._monitor_progress, daemon=True)
            self._stderr_thread.start()

    def _monitor_progress(self) -> None:
        """Parses FFmpeg stderr to track encoding speed and progress."""
        if not self._process or not self._process.stderr:
            return

        frame_pattern = re.compile(r"frame=\s*(\d+)")
        fps_pattern = re.compile(r"fps=\s*([\d\.]+)")

        try:
            # Stderr from FFmpeg is line-buffered or carriage-return delimited (\r)
            buffer = ""
            while self._process.poll() is None:
                char = self._process.stderr.read(1).decode("utf-8", errors="replace")
                if not char:
                    break
                if char in ("\r", "\n"):
                    line = buffer.strip()
                    buffer = ""
                    if line and "frame=" not in line:
                        self._stderr_tail.append(line)
                    if "frame=" in line:
                        f_match = frame_pattern.search(line)
                        fps_match = fps_pattern.search(line)
                        if f_match:
                            enc_frame = int(f_match.group(1))
                            if self.total_frames > 0:
                                self.progress = min(100.0, round((enc_frame / self.total_frames) * 100, 1))
                        if fps_match:
                            self.encoded_fps = float(fps_match.group(1))
                else:
                    buffer += char

            # poll() went non-None: the diagnosis is in whatever is still buffered.
            if self._process.stderr:
                rest = self._process.stderr.read().decode("utf-8", errors="replace")
                for line in (buffer + rest).replace("\r", "\n").split("\n"):
                    line = line.strip()
                    if line and "frame=" not in line:
                        self._stderr_tail.append(line)
        except Exception:
            pass

    def push_frame(self, frame_data: bytes, frames: int = 1) -> int:
        """Pipes one uncompressed RGBA frame, or one encoded chunk, into FFmpeg.

        `frames` is how much of the timeline the payload accounts for. A raw
        frame is always one. An encoded chunk is usually one too, but the caller
        knows for certain and progress should follow what it says rather than
        assuming.
        """
        with self._lock:
            if self.state != "rendering":
                raise RuntimeError(f"Cannot push frame: session is {self.state}")
            if not self._process or not self._process.stdin:
                raise RuntimeError("FFmpeg process is not running")

            if self.stream_format == "rawvideo":
                expected_size = self.width * self.height * 4
                if len(frame_data) != expected_size:
                    raise ValueError(
                        f"Invalid frame buffer size: got {len(frame_data)}, expected {expected_size} bytes."
                    )
            elif not frame_data:
                # An empty write would be indistinguishable from end-of-stream.
                raise ValueError("Empty encoded chunk")

            if self._process.poll() is not None:
                self.state = "error"
                self.error = self._encoder_failure()
                raise RuntimeError(self.error)

            try:
                self._process.stdin.write(frame_data)
                self.frames_written += frames
                if self.total_frames > 0:
                    # Update progress based on frames received if encoder is fast
                    self.progress = max(self.progress, round((self.frames_written / self.total_frames) * 98.0, 1))
                return self.frames_written
            except (BrokenPipeError, OSError) as exc:
                self.state = "error"
                # The pipe broke because FFmpeg exited. Its own last words are the
                # useful part; "Broken pipe" on its own tells the user nothing.
                self.error = self._encoder_failure(fallback=f"FFmpeg pipe broken: {exc}")
                raise RuntimeError(self.error) from exc

    _BANNER_PREFIXES = BANNER_PREFIXES

    def _stderr_summary(self) -> str:
        """The lines from FFmpeg's output that explain a failure, banner dropped."""
        return summarize_ffmpeg_stderr(self._stderr_tail)

    def _encoder_failure(self, fallback: str = "") -> str:
        """Builds an error string out of FFmpeg's exit code and its last output."""
        proc = self._process
        if proc is None:
            return fallback or "FFmpeg process is not running"
        try:
            code = proc.wait(timeout=1.0)
        except Exception:
            code = proc.poll()
        # The monitor thread owns the stderr fd, so read the tail it collected.
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=0.5)
        detail = self._stderr_summary() or self._drain_stderr(proc)
        encoder = self.encoder_used or self.codec_id
        if detail:
            return f"FFmpeg ({encoder}) exited with code {code}: {detail[-600:]}"
        return fallback or f"FFmpeg ({encoder}) exited with code {code}"

    def finish(self) -> dict[str, Any]:
        """Closes stdin and waits for FFmpeg to finalize the container file."""
        with self._lock:
            if self.state not in ("rendering", "finalizing"):
                return self.public()

            self.state = "finalizing"

            if self._process and self._process.stdin:
                try:
                    self._process.stdin.close()
                    self._process.stdin = None
                except Exception:
                    pass

        # Wait outside lock so stderr reader can finish
        if self._process:
            stderr = b""
            try:
                # The progress monitor is reading the same stderr fd, so
                # communicate() can come back empty or raise outright. Either way
                # the encoder's exit code is what decides the outcome.
                _stdout, stderr = self._process.communicate()
            except Exception:
                self._process.wait()
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1.0)
            if self._process.returncode != 0:
                self.state = "error"
                # stderr is usually empty here: the progress monitor has been
                # draining the same pipe, so its retained tail is the real record.
                err_msg = self._stderr_summary() or (
                    stderr.decode("utf-8", errors="replace") if stderr else ""
                )
                self.error = f"FFmpeg failed with code {self._process.returncode}: {err_msg[-600:] or 'Unknown error'}"
            else:
                self.state = "ready"
                self.progress = 100.0

        # Clean up temporary audio file if used
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
            except Exception:
                pass

        return self.public()

    def cancel(self) -> None:
        """Aborts encoding and cleans up temporary resources."""
        with self._lock:
            self.state = "cancelled"
            if self._process:
                try:
                    self._process.kill()
                except Exception:
                    pass

        # Clean up partial output file on cancel
        if os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
            except Exception:
                pass

        if self.audio_path and os.path.exists(self.audio_path):
            try:
                os.remove(self.audio_path)
            except Exception:
                pass

    def public(self) -> dict[str, Any]:
        """Public session payload returned to the GoatEdit web client."""
        file_size = 0
        if os.path.exists(self.output_path):
            try:
                file_size = os.path.getsize(self.output_path)
            except Exception:
                pass

        return {
            "sessionId": self.session_id,
            "state": self.state,
            "progress": self.progress,
            "framesWritten": self.frames_written,
            "totalFrames": self.total_frames,
            "encodedFps": self.encoded_fps,
            "outputPath": self.output_path,
            "fileSizeBytes": file_size,
            "error": self.error,
            "encoder": self.encoder_used,
            "fallbackReason": self.fallback_reason,
            "streamFormat": self.stream_format,
        }


class RenderManager:
    """Thread-safe store of active and completed render sessions."""
    def __init__(self) -> None:
        self._sessions: dict[str, RenderSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        width: int,
        height: int,
        fps: float,
        total_frames: int,
        codec_id: str,
        output_filename: str,
        output_dir: str | None = None,
        audio_path: str | None = None,
        bitrate: str | None = None,
        stream_format: str = "rawvideo",
    ) -> RenderSession:
        if stream_format not in ("rawvideo", "h264", "hevc"):
            raise ValueError(
                f"Unknown stream format: {stream_format}. "
                "Expected rawvideo, h264 or hevc."
            )
        hw_info = probe_hardware()
        target_dir = output_dir or hw_info["defaultExportDir"]
        target_path = os.path.join(target_dir, output_filename)

        session_id = uuid.uuid4().hex
        session = RenderSession(
            session_id=session_id,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            codec_id=codec_id,
            output_path=target_path,
            audio_path=audio_path,
            bitrate=bitrate,
            stream_format=stream_format,
        )

        with self._lock:
            self._sessions[session_id] = session
            # Prune sessions older than 2 hours
            now = time.time()
            for sid, s in list(self._sessions.items()):
                if now - s.created_at > 7200:
                    self._sessions.pop(sid, None)

        session.start()
        return session

    def get(self, session_id: str) -> RenderSession | None:
        with self._lock:
            return self._sessions.get(session_id)


# Global singleton manager
render_manager = RenderManager()
