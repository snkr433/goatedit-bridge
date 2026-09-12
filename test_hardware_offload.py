"""Automated test suite for Native Timeline Compilation, Proxies, and Audio AI."""

import os
import subprocess
import tempfile
import time
from goatedit_bridge.timeline_compiler import TimelineCompiler, TimelineSpec, TrackSpec, ClipSpec, timeline_manager
from goatedit_bridge.proxies import proxy_manager
from goatedit_bridge.ai_engine import detect_silences

def main():
    temp_dir = tempfile.mkdtemp(prefix="goatedit_test_")
    print(f"Creating test assets in {temp_dir}...")

    # 1. Generate 2 synthetic test video files with audio
    # clip1.mp4: 3 seconds of blue video with 440Hz tone
    clip1_path = os.path.join(temp_dir, "clip1.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=3:r=30",
        "-f", "lavfi", "-i", "sine=f=440:d=3",
        "-c:v", "libx264", "-c:a", "aac",
        clip1_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # clip2.mp4: 3 seconds of red video with 880Hz tone
    clip2_path = os.path.join(temp_dir, "clip2.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=3:r=30",
        "-f", "lavfi", "-i", "sine=f=880:d=3",
        "-c:v", "libx264", "-c:a", "aac",
        clip2_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # clip_speech_silence.mp4: 1s tone followed by 2s silence
    silence_test_path = os.path.join(temp_dir, "silence_test.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=640x360:d=3:r=30",
        "-f", "lavfi", "-i", "sine=f=440:d=1,apad=pad_dur=2",
        "-c:v", "libx264", "-c:a", "aac",
        silence_test_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("Test assets generated.")

    # --- Test 1: Native Timeline Compiler ---
    print("\n--- Testing Pillar 1: Native Timeline Compilation ---")
    out_mov = os.path.join(temp_dir, "compiled_master.mov")
    spec = TimelineSpec(
        width=1280,
        height=720,
        fps=30,
        duration=4.0,
        codec_id="prores_422",
        output_path=out_mov,
        tracks=[
            TrackSpec(
                track_id="t1",
                track_type="video",
                clips=[
                    # Cut 1: clip1 from 0.5s to 2.5s (2.0s duration), placed at 0.0s
                    ClipSpec(
                        clip_id="c1",
                        media_id="m1",
                        file_path=clip1_path,
                        start_time=0.0,
                        duration=2.0,
                        source_in=0.5,
                        source_out=2.5,
                        speed=1.0,
                        volume=0.8,
                    ),
                    # Cut 2: clip2 from 0.0s to 2.0s (2.0s duration), placed at 2.0s
                    ClipSpec(
                        clip_id="c2",
                        media_id="m2",
                        file_path=clip2_path,
                        start_time=2.0,
                        duration=2.0,
                        source_in=0.0,
                        source_out=2.0,
                        speed=1.0,
                        volume=1.0,
                    ),
                ]
            )
        ]
    )

    t0 = time.time()
    job = timeline_manager.create_job(spec)
    while job.state in ("initialized", "compiling"):
        time.sleep(0.05)
    elapsed = time.time() - t0

    assert job.state == "ready", f"Timeline compile failed: {job.error}"
    assert os.path.exists(out_mov), f"Output file does not exist: {out_mov}"
    size_mb = os.path.getsize(out_mov) / (1024 * 1024)
    print(f"✓ Native Timeline Compilation SUCCESS in {elapsed:.2f}s! Size: {size_mb:.2f} MB, Output: {out_mov}")

    # --- Test 2: Background Hardware Proxy Engine ---
    print("\n--- Testing Pillar 2: Background Hardware Proxy Engine ---")
    t0 = time.time()
    pjob = proxy_manager.create_proxy(clip1_path, height=720, codec_id="h264_hw")
    while pjob.state in ("queued", "transcoding"):
        time.sleep(0.05)
    proxy_elapsed = time.time() - t0

    assert pjob.state == "ready", f"Proxy generation failed: {pjob.error}"
    assert os.path.exists(pjob.target_path), f"Proxy file does not exist: {pjob.target_path}"
    print(f"✓ Proxy Generation SUCCESS in {proxy_elapsed:.2f}s! Target: {pjob.target_path}")

    # --- Test 3: Local Audio AI Silence Detection ---
    print("\n--- Testing Pillar 3: Local Audio AI Silence Detection ---")
    t0 = time.time()
    silence_result = detect_silences(silence_test_path, noise_threshold_db=-30.0, min_duration_sec=0.5)
    ai_elapsed = time.time() - t0

    print(f"✓ Silence Detection SUCCESS in {ai_elapsed:.3f}s! Detected: {len(silence_result['silences'])} silent intervals, {len(silence_result['speechSegments'])} speech segments.")
    assert len(silence_result["silences"]) >= 1, "Expected at least 1 silence detected"

    print("\n=========================================")
    print("ALL 3 HARDWARE OFFLOAD PILLARS TESTED AND PASSING!")
    print("=========================================")

if __name__ == "__main__":
    main()
