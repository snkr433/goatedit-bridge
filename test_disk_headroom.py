"""
An export that cannot fit is refused before it starts.

Running out of disk part-way through leaves a file with no trailer — unplayable,
and still holding every byte it managed to write:

  [out#0/mov] Error writing trailer: No space left on device
  Conversion failed!

ProRes is the one that reaches it. 422 is roughly 1.1 GB per minute at 1080p30.
"""

import os
import shutil
import tempfile

from goatedit_bridge.render import check_disk_headroom, estimate_output_bytes

GB = 1024 ** 3
tmp = tempfile.mkdtemp(prefix="headroom_")


def check_estimates_are_in_the_right_range():
    # ProRes 422, one minute at 1080p30: about a gigabyte.
    one_min = estimate_output_bytes("prores_422", 1920, 1080, 30, 60)
    assert 0.8 * GB < one_min < 1.5 * GB, f"{one_min / GB:.2f} GB/min is not ProRes 422"

    # H.264 of the same thing is an order of magnitude smaller.
    h264 = estimate_output_bytes("h264_hw", 1920, 1080, 30, 60)
    assert h264 * 5 < one_min, f"h264 {h264 / GB:.2f} GB vs prores {one_min / GB:.2f} GB"

    # Half the pixels, half the size.
    half = estimate_output_bytes("prores_422", 1280, 720, 30, 60)
    ratio = half / one_min
    assert 0.4 < ratio < 0.5, f"720p should be ~0.44 of 1080p, got {ratio:.2f}"

    # An unknown codec still gets an estimate rather than zero.
    assert estimate_output_bytes("something_new", 1920, 1080, 30, 60) > 0
    print("✓ size estimates track codec, resolution and duration")


def check_impossible_export_is_refused():
    out = os.path.join(tmp, "master.mov")
    # A hundred hours of ProRes 422 HQ is tens of terabytes. No machine running
    # this test has room, so the refusal is not a guess about the environment.
    try:
        check_disk_headroom(out, "prores_422_hq", 1920, 1080, 30, 100 * 3600)
    except RuntimeError as exc:
        message = str(exc)
        assert "Not enough disk space" in message, message
        assert "GB free" in message, message
        assert "h264_hw" in message, "the message should say what to do instead"
        print("✓ refused with:", message[:120])
    else:
        raise AssertionError("a 100-hour ProRes master should not have been accepted")


def check_a_normal_export_passes():
    out = os.path.join(tmp, "short.mp4")
    free = shutil.disk_usage(tmp).free
    # Ten seconds of h264 is a few megabytes; only fails if the disk is already
    # full, which is worth knowing about anyway.
    assert free > 200 * 1024 * 1024, "this machine has no disk space at all"
    check_disk_headroom(out, "h264_hw", 1920, 1080, 30, 10)
    print("✓ a short H.264 export is allowed through")


def check_unknown_duration_is_not_blocked():
    """A frame-stream session with no frame count yet cannot be estimated."""
    check_disk_headroom(os.path.join(tmp, "x.mov"), "prores_422_hq", 1920, 1080, 30, 0)
    print("✓ an export of unknown length is left alone")


check_estimates_are_in_the_right_range()
check_impossible_export_is_refused()
check_a_normal_export_passes()
check_unknown_duration_is_not_blocked()
print("\nALL DISK HEADROOM CHECKS PASSED")
