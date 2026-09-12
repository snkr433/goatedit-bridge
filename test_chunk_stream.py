"""
The browser encodes; the bridge only muxes.

WebCodecs on every platform this bridge runs on reaches the same hardware
encoder FFmpeg would have (VideoToolbox, NVENC, QSV), so having the tab read
back 8 MB of RGBA per frame and post it here to be encoded again was strictly
worse than letting it encode and sending the result. A 1080p frame is about
8.3 MB raw and about 80 KB as H.264 — two orders of magnitude, per frame, plus
a GPU-to-CPU readback the browser no longer has to do at all.

This checks the receiving half: an Annex-B elementary stream arriving on stdin
is copied into the container without re-encoding, keeps its length, and takes
the audio alongside it.
"""

import os
import subprocess
import tempfile

from goatedit_bridge.render import render_manager

tmp = tempfile.mkdtemp(prefix="chunk_stream_")
W, H, FPS, SECONDS = 320, 180, 30, 2


def ffprobe(path: str, *entries: str) -> list[str]:
    """Values for the asked-for entries, in order. Stream and format entries live
    in different sections, so each needs its own -show_entries."""
    cmd = ["ffprobe", "-v", "error"]
    for entry in entries:
        cmd.extend(["-show_entries", entry])
    cmd.extend(["-of", "default=nw=1:nk=1", path])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def make_annexb(name: str, codec: str, fmt: str) -> bytes:
    """What WebCodecs hands back with `format: 'annexb'`, near enough."""
    path = os.path.join(tmp, name)
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-f", "lavfi",
         "-i", f"color=c=green:s={W}x{H}:d={SECONDS}:r={FPS}",
         "-c:v", codec, "-pix_fmt", "yuv420p", "-f", fmt, path],
        check=True, capture_output=True,
    )
    return open(path, "rb").read()


def make_wav() -> str:
    path = os.path.join(tmp, "mix.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-f", "lavfi",
         "-i", f"sine=f=440:d={SECONDS}", "-c:a", "pcm_s16le", path],
        check=True, capture_output=True,
    )
    return path


def check_h264_chunks_are_muxed_not_reencoded():
    stream = make_annexb("green.h264", "libx264", "h264")
    session = render_manager.create_session(
        width=W, height=H, fps=FPS, total_frames=SECONDS * FPS,
        codec_id="h264_hw", output_filename="chunked.mp4", output_dir=tmp,
        stream_format="h264",
    )
    assert session.encoder_used == "copy", f"expected a stream copy, got {session.encoder_used}"

    # Sent in pieces, the way an encoder hands them over.
    step = max(1, len(stream) // 8)
    for offset in range(0, len(stream), step):
        session.push_frame(stream[offset:offset + step], frames=0)
    result = session.finish()

    out = result["outputPath"]
    assert os.path.exists(out), "no output file"
    codec, duration = ffprobe(out, "stream=codec_name", "format=duration")
    assert codec == "h264", f"expected h264 in the container, got {codec}"
    assert abs(float(duration) - SECONDS) < 0.3, f"duration drifted: {duration}"
    print(f"✓ h264 copied through, {os.path.getsize(out)} bytes, {duration}s")


def check_audio_rides_along():
    stream = make_annexb("green2.h264", "libx264", "h264")
    session = render_manager.create_session(
        width=W, height=H, fps=FPS, total_frames=SECONDS * FPS,
        codec_id="h264_hw", output_filename="chunked_audio.mp4", output_dir=tmp,
        audio_path=make_wav(), stream_format="h264",
    )
    session.push_frame(stream, frames=SECONDS * FPS)
    result = session.finish()

    streams = ffprobe(result["outputPath"], "stream=codec_type")
    assert "video" in streams and "audio" in streams, f"expected both tracks, got {streams}"
    print("✓ the uploaded mix is muxed alongside the copied video")


def check_frame_count_drives_progress():
    stream = make_annexb("green3.h264", "libx264", "h264")
    session = render_manager.create_session(
        width=W, height=H, fps=FPS, total_frames=SECONDS * FPS,
        codec_id="h264_hw", output_filename="progress.mp4", output_dir=tmp,
        stream_format="h264",
    )
    session.push_frame(stream, frames=30)
    assert session.frames_written == 30, session.frames_written
    assert 0 < session.progress < 98, f"progress should have moved, got {session.progress}"
    session.finish()
    print("✓ progress follows the frame count the caller reports, not the chunk count")


def check_raw_sessions_still_police_frame_size():
    session = render_manager.create_session(
        width=W, height=H, fps=FPS, total_frames=1,
        codec_id="h264_hw", output_filename="raw.mp4", output_dir=tmp,
    )
    assert session.stream_format == "rawvideo"
    try:
        session.push_frame(b"too short")
    except ValueError as exc:
        assert "Invalid frame buffer size" in str(exc), exc
        print("✓ a raw session still rejects a wrong-sized frame")
    else:
        raise AssertionError("a 9-byte RGBA frame should not have been accepted")
    finally:
        session.cancel()


def check_unknown_format_is_refused():
    try:
        render_manager.create_session(
            width=W, height=H, fps=FPS, total_frames=1, codec_id="h264_hw",
            output_filename="nope.mp4", output_dir=tmp, stream_format="theora",
        )
    except ValueError as exc:
        assert "Unknown stream format" in str(exc), exc
        print("✓ an unrecognised stream format is refused up front")
    else:
        raise AssertionError("theora should not have been accepted")


check_h264_chunks_are_muxed_not_reencoded()
check_audio_rides_along()
check_frame_count_drives_progress()
check_raw_sessions_still_police_frame_size()
check_unknown_format_is_refused()
print("\nALL CHUNK STREAM CHECKS PASSED")
