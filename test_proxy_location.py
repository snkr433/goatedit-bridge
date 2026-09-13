"""Where proxies get written, and what that means for serving them.

Proxies live beside the footage so they can be found and relinked by hand.
Two consequences are pinned here: the name has to stay readable (a content hash
would make a manual relink impossible), and because the name no longer carries
the source's mtime, staleness has to be detected some other way.
"""

from __future__ import annotations

import os
import tempfile
import time

from goatedit_bridge.proxies import PROXY_DIR_NAME, ProxyManager


def test_proxy_lands_in_a_proxies_folder_beside_the_source():
    with tempfile.TemporaryDirectory() as d:
        footage = os.path.join(d, "Cam 01")
        os.makedirs(footage)
        src = os.path.join(footage, "B0035.MP4")
        open(src, "wb").write(b"x")

        mgr = ProxyManager(cache_dir=os.path.join(d, "cache"))
        target = mgr.get_proxy_path(src, height=540)

        assert os.path.dirname(target) == os.path.join(footage, PROXY_DIR_NAME), target
        # Readable, and it says what it is. No hash: a hashed name cannot be relinked by hand.
        assert os.path.basename(target) == "B0035_540p.mp4", target


def test_unwritable_source_location_falls_back_to_the_cache():
    with tempfile.TemporaryDirectory() as d:
        footage = os.path.join(d, "card")
        os.makedirs(footage)
        src = os.path.join(footage, "B0035.MP4")
        open(src, "wb").write(b"x")
        cache = os.path.join(d, "cache")
        mgr = ProxyManager(cache_dir=cache)

        os.chmod(footage, 0o555)  # read-only, like a mounted card
        try:
            target = mgr.get_proxy_path(src, height=540)
            assert os.path.dirname(target) == os.path.realpath(cache) or target.startswith(cache), target
            # Two cards can both hold a B0035.MP4, so the fallback name stays unique.
            assert "B0035_" in os.path.basename(target)
        finally:
            os.chmod(footage, 0o755)


def test_a_proxy_older_than_its_source_is_rebuilt():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "B0035.MP4")
        open(src, "wb").write(b"new")
        proxy = os.path.join(d, "B0035_540p.mp4")
        open(proxy, "wb").write(b"old")

        old = time.time() - 600
        os.utime(proxy, (old, old))

        assert ProxyManager._is_stale(src, proxy) is True
        os.utime(proxy, None)  # touch it: now newer than the source
        assert ProxyManager._is_stale(src, proxy) is False


def test_only_files_we_produced_may_be_served():
    """The serving guard authorises exact files, never the folder they sit in —
    otherwise putting proxies next to footage would expose the footage."""
    with tempfile.TemporaryDirectory() as d:
        mgr = ProxyManager(cache_dir=os.path.join(d, "cache"))
        neighbour = os.path.join(d, "PRIVATE.MP4")
        open(neighbour, "wb").write(b"x")

        assert mgr.is_known_target(neighbour) is False

        produced = os.path.join(d, "B0035_540p.mp4")
        open(produced, "wb").write(b"x")
        from goatedit_bridge.proxies import ProxyJob
        mgr._jobs["j1"] = ProxyJob(job_id="j1", source_path="/src.mp4", target_path=produced)

        assert mgr.is_known_target(produced) is True
        # A sibling in the same directory is still refused.
        assert mgr.is_known_target(neighbour) is False


def main():
    test_proxy_lands_in_a_proxies_folder_beside_the_source()
    test_unwritable_source_location_falls_back_to_the_cache()
    test_a_proxy_older_than_its_source_is_rebuilt()
    test_only_files_we_produced_may_be_served()
    test_job_ids_survive_a_bridge_restart()
    test_index_forgets_a_proxy_that_was_deleted()
    print("\nALL PROXY LOCATION TESTS PASSED!")




def test_job_ids_survive_a_bridge_restart():
    """A proxyUrl is saved into the project, so its id must outlive the process.

    Without a persisted index every stored proxy URL 404s the next time the
    bridge starts, while the proxy file sits on disk untouched — which is
    exactly what happened the first time this was run end to end.
    """
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        footage = os.path.join(d, "Cam 01")
        os.makedirs(footage)
        src = os.path.join(footage, "B0035.MP4")
        open(src, "wb").write(b"source")

        first = ProxyManager(cache_dir=cache)
        target = first.get_proxy_path(src, height=540)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "wb").write(b"proxy-bytes")

        job = first.create_proxy(src, height=540)   # finds it on disk, marks ready
        assert job.state == "ready", job.state

        # A brand new manager, as if the bridge had been restarted.
        second = ProxyManager(cache_dir=cache)
        revived = second.get(job.job_id)
        assert revived is not None, "job id did not survive the restart"
        assert revived.state == "ready", revived.state
        assert revived.target_path == target
        assert second.is_known_target(target) is True


def test_index_forgets_a_proxy_that_was_deleted():
    with tempfile.TemporaryDirectory() as d:
        cache = os.path.join(d, "cache")
        footage = os.path.join(d, "Cam 01")
        os.makedirs(footage)
        src = os.path.join(footage, "B0035.MP4")
        open(src, "wb").write(b"source")

        first = ProxyManager(cache_dir=cache)
        target = first.get_proxy_path(src, height=540)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        open(target, "wb").write(b"proxy-bytes")
        job = first.create_proxy(src, height=540)

        os.remove(target)  # user cleaned out the Proxies folder

        second = ProxyManager(cache_dir=cache)
        assert second.get(job.job_id) is None, "a deleted proxy must not be re-registered"


if __name__ == "__main__":
    main()
