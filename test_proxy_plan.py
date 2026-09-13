"""Tests for the proxy decision: which footage a browser genuinely cannot play.

The point of these is that the decision is about DECODE CAPABILITY, not size.
A 4K 8-bit 4:2:0 clip plays fine in a tab; a 1080p 4:2:2 10-bit one does not,
because no Mac has a hardware decoder for 4:2:2 H.264 and a browser has no fast
software fallback. Getting this backwards proxies the wrong files.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

from goatedit_bridge.proxies import needs_proxy, probe_decode_profile


def _make(path: str, pix_fmt: str, size: str, extra: list[str] | None = None) -> None:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration=1",
        "-c:v", "libx264", "-pix_fmt", pix_fmt,
    ]
    if extra:
        cmd.extend(extra)
    cmd.append(path)
    subprocess.run(cmd, check=True, timeout=120)


def test_422_10bit_is_proxied_even_at_small_sizes():
    """The case that started all this: Sony XAVC-I. Small, and still unplayable."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "xavc.mp4")
        _make(p, "yuv422p10le", "640x360", ["-profile:v", "high422"])
        verdict = needs_proxy(p)
        assert verdict["needed"] is True, verdict
        assert "4:2:2" in verdict["reason"], verdict["reason"]
        assert verdict["probe"]["pix_fmt"] == "yuv422p10le", verdict["probe"]


def test_8bit_420_hd_is_left_alone():
    """Proxying what already hardware-decodes is pure waste."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plain.mp4")
        _make(p, "yuv420p", "1280x720")
        verdict = needs_proxy(p)
        assert verdict["needed"] is False, verdict


def test_large_frame_is_proxied_even_when_decodable():
    """Past a point the texture upload costs more than the transcode saves."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "big.mp4")
        _make(p, "yuv420p", "3840x2160")
        verdict = needs_proxy(p)
        assert verdict["needed"] is True, verdict
        assert "wide" in verdict["reason"], verdict["reason"]


def test_probe_reports_frame_rate_as_a_number():
    """r_frame_rate arrives as the rational "25/1"; downstream wants 25.0.

    The editor currently mis-reads 50fps sources as 30, so the bridge giving a
    real number is the thing that can correct it.
    """
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rate.mp4")
        _make(p, "yuv420p", "320x240")
        probe = probe_decode_profile(p)
        assert probe["fps"] == 25.0, probe


def test_missing_file_is_not_a_crash_and_not_a_proxy():
    verdict = needs_proxy("/nope/does/not/exist.mp4")
    assert verdict["needed"] is False, verdict
    assert "probe" in verdict


def main():
    test_422_10bit_is_proxied_even_at_small_sizes()
    test_8bit_420_hd_is_left_alone()
    test_large_frame_is_proxied_even_when_decodable()
    test_probe_reports_frame_rate_as_a_number()
    test_missing_file_is_not_a_crash_and_not_a_proxy()
    print("\nALL PROXY PLAN TESTS PASSED!")


if __name__ == "__main__":
    main()
