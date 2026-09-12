"""Tests for Smart Rendering (Lossless Stream Copy) and Fast-Seek GPU Compilation."""

import os
import shutil
import subprocess
import tempfile
import time

from goatedit_bridge.timeline_compiler import (
    ClipSpec,
    TimelineCompiler,
    TimelineSpec,
    TrackSpec,
    probe_hardware,
    timeline_manager,
)


def test_smart_render_and_fast_seek():
    temp_dir = tempfile.mkdtemp(prefix="goatedit_bench_")
    print(f"Creating test video in {temp_dir}...")

    src = os.path.join(temp_dir, "test_source_60s.mp4")
    # Generate 60-second 1080p source video
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=60:size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=f=440:duration=60",
        "-c:v", "h264_videotoolbox", "-b:v", "3000k",
        "-c:a", "aac", "-b:a", "128k",
        src,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Test source generated successfully.")

    # Define 3 cuts from across the 60-second video:
    # Cut 1: 5s to 12s (dur 7s)
    # Cut 2: 25s to 35s (dur 10s)
    # Cut 3: 45s to 55s (dur 10s)
    # Total sequence duration = 27s
    clips = [
        ClipSpec(clip_id="c1", media_id="m1", file_path=src, start_time=0.0, duration=7.0, source_in=5.0, source_out=12.0),
        ClipSpec(clip_id="c2", media_id="m1", file_path=src, start_time=7.0, duration=10.0, source_in=25.0, source_out=35.0),
        ClipSpec(clip_id="c3", media_id="m1", file_path=src, start_time=17.0, duration=10.0, source_in=45.0, source_out=55.0),
    ]

    tracks = [TrackSpec(track_id="t1", track_type="video", clips=clips)]

    # --- Test 1: Smart Render (Lossless Stream Copy) ---
    out_smart = os.path.join(temp_dir, "out_smart.mp4")
    spec_smart = TimelineSpec(
        width=1920,
        height=1080,
        fps=30.0,
        duration=27.0,
        codec_id="h264_hw",
        output_path=out_smart,
        tracks=tracks,
    )

    compiler = TimelineCompiler(spec_smart)
    assert compiler.can_smart_render() is True, "Expected spec to be eligible for Smart Render!"
    print("✓ can_smart_render() correctly identified eligible cuts.")

    t0 = time.time()
    job_smart = timeline_manager.create_job(spec_smart)
    for _ in range(50):
        if job_smart.state in ("ready", "error"):
            break
        time.sleep(0.05)

    dur_smart = time.time() - t0
    assert job_smart.state == "ready", f"Smart render failed: {job_smart.error}"
    assert job_smart.render_mode == "smart_copy", f"Unexpected mode: {job_smart.render_mode}"
    assert os.path.exists(out_smart) and os.path.getsize(out_smart) > 0
    print(f"✓ Smart Render completed in {dur_smart:.2f}s! Size: {os.path.getsize(out_smart) / (1024*1024):.2f} MB (Mode: {job_smart.render_mode})")

    # --- Test 2: Fast-Seek GPU Compilation (when effect/speed change is present) ---
    clips_effect = [
        ClipSpec(clip_id="c1", media_id="m1", file_path=src, start_time=0.0, duration=7.0, source_in=5.0, source_out=12.0, speed=1.5),
        ClipSpec(clip_id="c2", media_id="m1", file_path=src, start_time=7.0, duration=10.0, source_in=25.0, source_out=35.0),
    ]
    tracks_effect = [TrackSpec(track_id="t1", track_type="video", clips=clips_effect)]

    out_gpu = os.path.join(temp_dir, "out_gpu.mp4")
    spec_gpu = TimelineSpec(
        width=1920,
        height=1080,
        fps=30.0,
        duration=17.0,
        codec_id="h264_hw",
        output_path=out_gpu,
        tracks=tracks_effect,
    )

    compiler_gpu = TimelineCompiler(spec_gpu)
    assert compiler_gpu.can_smart_render() is False, "Expected speed change to disable Smart Render"
    print("✓ can_smart_render() correctly detected speed effect requiring GPU transcode.")

    t0 = time.time()
    job_gpu = timeline_manager.create_job(spec_gpu)
    for _ in range(100):
        if job_gpu.state in ("ready", "error"):
            break
        time.sleep(0.05)

    dur_gpu = time.time() - t0
    assert job_gpu.state == "ready", f"GPU render failed: {job_gpu.error}"
    assert job_gpu.render_mode == "gpu_accelerated"
    assert os.path.exists(out_gpu) and os.path.getsize(out_gpu) > 0
    print(f"✓ Fast-Seek GPU render completed in {dur_gpu:.2f}s! Size: {os.path.getsize(out_gpu) / (1024*1024):.2f} MB (Mode: {job_gpu.render_mode})")

    # --- Test 3: Fast-Seek GPU Compilation with Silent Video Clip (No Audio Stream) ---
    silent_src = os.path.join(temp_dir, "silent_source.mp4")
    # Generate a pure video clip without any audio stream
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=5:size=1280x720:rate=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", silent_src
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    clips_silent = [
        ClipSpec(clip_id="c_aud", media_id="m1", file_path=src, start_time=0.0, duration=3.0, source_in=0.0, source_out=3.0),
        # 1-second gap on timeline, followed by silent clip
        ClipSpec(clip_id="c_sil", media_id="m2", file_path=silent_src, start_time=4.0, duration=3.0, source_in=0.0, source_out=3.0),
    ]
    tracks_silent = [TrackSpec(track_id="t1", track_type="video", clips=clips_silent)]

    out_silent = os.path.join(temp_dir, "out_silent.mp4")
    spec_silent = TimelineSpec(
        width=1280,
        height=720,
        fps=30.0,
        duration=7.0,
        codec_id="h264_hw",
        output_path=out_silent,
        tracks=tracks_silent,
    )

    t0 = time.time()
    job_silent = timeline_manager.create_job(spec_silent)
    for _ in range(100):
        if job_silent.state in ("ready", "error"):
            break
        time.sleep(0.05)

    dur_silent = time.time() - t0
    assert job_silent.state == "ready", f"Silent clip compilation failed: {job_silent.error}"
    assert os.path.exists(out_silent) and os.path.getsize(out_silent) > 0
    print(f"✓ Silent video clip + timeline gap compilation passed in {dur_silent:.2f}s! Size: {os.path.getsize(out_silent) / (1024*1024):.2f} MB")

    # --- Test 4: Multi-Track Hardware Overlay Compilation (Base Track + B-Roll Overlay) ---
    clips_base = [ClipSpec(clip_id="c_base", media_id="m1", file_path=src, start_time=0.0, duration=5.0, source_in=0.0, source_out=5.0)]
    clips_overlay = [ClipSpec(clip_id="c_ov", media_id="m2", file_path=silent_src, start_time=1.0, duration=2.5, source_in=0.0, source_out=2.5)]
    tracks_multi = [
        TrackSpec(track_id="t_base", track_type="video", clips=clips_base),
        TrackSpec(track_id="t_ov", track_type="video", clips=clips_overlay),
    ]

    out_multi = os.path.join(temp_dir, "out_multi.mp4")
    spec_multi = TimelineSpec(
        width=1280,
        height=720,
        fps=30.0,
        duration=5.0,
        codec_id="h264_hw",
        output_path=out_multi,
        tracks=tracks_multi,
    )

    t0 = time.time()
    job_multi = timeline_manager.create_job(spec_multi)
    for _ in range(100):
        if job_multi.state in ("ready", "error"):
            break
        time.sleep(0.05)

    dur_multi = time.time() - t0
    assert job_multi.state == "ready", f"Multi-track overlay compilation failed: {job_multi.error}"
    assert os.path.exists(out_multi) and os.path.getsize(out_multi) > 0
    print(f"✓ Multi-Track GPU Overlay compilation passed in {dur_multi:.2f}s! Size: {os.path.getsize(out_multi) / (1024*1024):.2f} MB")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nALL SMART RENDER AND FAST-SEEK TESTS PASSED!")


if __name__ == "__main__":
    test_smart_render_and_fast_seek()
