"""A proxy URL has to work for a <video> element, not just for fetch().

A media element cannot send an Authorization header. Handing the editor a URL
that only authenticates by header means every proxy 401s for the one consumer
that matters — while passing every test written with fetch(). That is exactly
how it got past a unit test here and was caught only end to end, so the
header-less GET is pinned.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import urllib.error
import urllib.request

from goatedit_bridge.jobs import JobStore
from goatedit_bridge.proxies import ProxyJob, proxy_manager
from goatedit_bridge.server import BridgeConfig, make_server

TOKEN = "testtoken123"
ORIGIN = "http://localhost:4173"


def _serve(work_dir: str):
    config = BridgeConfig(port=0, token=TOKEN, origins=[ORIGIN])
    server = make_server(config, JobStore(work_dir=work_dir))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main():
    with tempfile.TemporaryDirectory() as d:
        work = os.path.join(d, "work")
        os.makedirs(work)

        # A finished proxy, registered the way a real job would be.
        target = os.path.join(d, "B0035_540p.mp4")
        with open(target, "wb") as fh:
            fh.write(b"0123456789" * 20)
        job_id = "jobtest"
        proxy_manager._jobs[job_id] = ProxyJob(
            job_id=job_id, source_path="/src.MP4", target_path=target,
            state="ready", progress=100.0, file_size_bytes=os.path.getsize(target),
        )

        server, port = _serve(work)
        try:
            base = f"http://127.0.0.1:{port}/file/{job_id}"
            hdrs = {"Origin": ORIGIN}

            status, _ = _get(base, hdrs)
            assert status == 401, f"no credentials should be refused, got {status}"

            status, body = _get(f"{base}?token={TOKEN}", hdrs)
            assert status == 200, f"a <video>-style GET with ?token must be served, got {status}"
            assert len(body) == 200, len(body)

            status, _ = _get(f"{base}?token=wrong", hdrs)
            assert status == 401, f"a wrong query token must still be refused, got {status}"

            status, _ = _get(base, {**hdrs, "Authorization": f"Bearer {TOKEN}"})
            assert status == 200, f"the header path must keep working, got {status}"

            # Range, because seeking is most of the point of a proxy.
            req = urllib.request.Request(f"{base}?token={TOKEN}",
                                         headers={**hdrs, "Range": "bytes=10-19"})
            with urllib.request.urlopen(req, timeout=5) as res:
                assert res.status == 206, res.status
                assert res.headers.get("Content-Range") == "bytes 10-19/200", res.headers.get("Content-Range")
                assert res.read() == b"0123456789", "wrong bytes for the requested range"

            print("\nALL MEDIA TOKEN TESTS PASSED!")
        finally:
            server.shutdown()
            proxy_manager._jobs.pop(job_id, None)


if __name__ == "__main__":
    main()
