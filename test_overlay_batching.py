"""
Many image overlays have to compile, and land where they belong.

FFmpeg binds every input to the filtergraph before decoding anything, and it
runs out of room somewhere between 200 and 390 inputs: `Error binding filtergraph
inputs/outputs: Resource temporarily unavailable`, exit code 221. One input per
caption reaches that on a normal subtitled video, so image overlays are batched
into concat sequences — one input per lane of non-overlapping overlays.
"""

import os
import subprocess
import tempfile

from PIL import Image

from goatedit_bridge.timeline_compiler import (
    ClipSpec,
    TimelineCompiler,
    TimelineSpec,
    TrackSpec,
    _lay_out_overlay_lanes,
)

W, H, FPS = 320, 180, 30
tmp = tempfile.mkdtemp(prefix="overlay_batch_")


def solid(name: str, rgba: tuple[int, int, int, int]) -> str:
    path = os.path.join(tmp, name)
    Image.new("RGBA", (W, H), rgba).save(path)
    return path


def colour_at(video: str, t: float) -> tuple[int, int, int]:
    frame = os.path.join(tmp, f"probe_{t}.png")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-ss", str(t), "-i", video,
                    "-frames:v", "1", frame], check=True, capture_output=True)
    return Image.open(frame).convert("RGB").getpixel((W // 2, H // 2))


def base_clip() -> TrackSpec:
    src = os.path.join(tmp, "base.mp4")
    if not os.path.exists(src):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        f"color=c=blue:s={W}x{H}:d=12:r={FPS}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", src],
                       check=True, capture_output=True)
    return TrackSpec(track_id="v0", track_type="video", clips=[
        ClipSpec(clip_id="b1", media_id="m0", file_path=src,
                 start_time=0.0, duration=12.0, source_in=0.0, source_out=12.0),
    ])


def check_lane_layout():
    clips = [
        ClipSpec(clip_id="a", media_id="m", file_path="a.png", start_time=0.0, duration=1.0, source_in=0.0, source_out=1.0),
        ClipSpec(clip_id="b", media_id="m", file_path="b.png", start_time=1.0, duration=1.0, source_in=0.0, source_out=1.0),
        ClipSpec(clip_id="c", media_id="m", file_path="c.png", start_time=0.5, duration=1.0, source_in=0.0, source_out=1.0),
    ]
    lanes = _lay_out_overlay_lanes(clips)
    assert len(lanes) == 2, f"expected two lanes for one overlap, got {len(lanes)}"
    assert [c.clip_id for c in lanes[0]] == ["a", "b"], [c.clip_id for c in lanes[0]]
    assert [c.clip_id for c in lanes[1]] == ["c"], [c.clip_id for c in lanes[1]]
    print("✓ overlapping overlays go to separate lanes, the rest share one")


def check_many_overlays_compile():
    """250 overlays: past what FFmpeg will bind one at a time."""
    red = solid("red.png", (255, 0, 0, 255))
    green = solid("green.png", (0, 255, 0, 255))

    overlays = []
    for i in range(250):
        # Alternating colours, back to back, 40ms each, starting at 1s.
        overlays.append(ClipSpec(
            clip_id=f"o{i}", media_id="mo",
            file_path=red if i % 2 == 0 else green,
            start_time=1.0 + i * 0.04, duration=0.04, source_in=0.0, source_out=0.04,
        ))

    spec = TimelineSpec(
        width=W, height=H, fps=FPS, duration=12.0,
        codec_id="h264_hw", output_path=os.path.join(tmp, "many.mp4"),
        tracks=[base_clip(), TrackSpec(track_id="v1", track_type="video", clips=overlays)],
    )
    compiler = TimelineCompiler(spec)
    cmd = compiler.build_fast_seek_command()

    inputs = cmd.count("-i")
    assert inputs <= 4, f"250 overlays should not become {inputs} inputs"
    print(f"✓ 250 overlays compiled down to {inputs} FFmpeg inputs")

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"compile failed ({res.returncode}): {res.stderr[-800:]}"
    assert os.path.exists(spec.output_path), "no output file"
    print("✓ FFmpeg accepted the batched graph")

    # Baseline stays visible where no overlay is scheduled, and the overlays
    # land inside their window rather than smeared across the whole clip.
    r, g, b = colour_at(spec.output_path, 0.5)
    assert b > 200 and r < 60, f"expected the base at 0.5s, got {(r, g, b)}"
    r, g, b = colour_at(spec.output_path, 11.5)
    assert b > 200 and r < 60, f"expected the base after the overlays, got {(r, g, b)}"
    covered = colour_at(spec.output_path, 5.0)
    assert covered[2] < 120, f"expected an overlay at 5.0s, got {covered}"
    print("✓ overlays land in their window and nowhere else:", covered)

    for scratch in compiler.scratch_dirs:
        assert os.path.isdir(scratch), "scratch should still exist until the job clears it"
    print(f"✓ scratch dirs registered for cleanup: {len(compiler.scratch_dirs)}")


def check_few_overlays_stay_unbatched():
    """Below the threshold nothing changes — the proven path stays in use."""
    red = solid("red2.png", (255, 0, 0, 255))
    overlays = [
        ClipSpec(clip_id=f"s{i}", media_id="mo", file_path=red,
                 start_time=1.0 + i, duration=0.5, source_in=0.0, source_out=0.5)
        for i in range(3)
    ]
    spec = TimelineSpec(
        width=W, height=H, fps=FPS, duration=12.0,
        codec_id="h264_hw", output_path=os.path.join(tmp, "few.mp4"),
        tracks=[base_clip(), TrackSpec(track_id="v1", track_type="video", clips=overlays)],
    )
    compiler = TimelineCompiler(spec)
    cmd = compiler.build_fast_seek_command()
    assert "concat" not in cmd, "three overlays should not be batched"
    assert cmd.count("-i") == 4, f"expected base + 3 overlay inputs, got {cmd.count('-i')}"
    print("✓ a handful of overlays still get one input each")


check_lane_layout()
check_few_overlays_stay_unbatched()
check_many_overlays_compile()
print("\nALL OVERLAY BATCHING CHECKS PASSED")
