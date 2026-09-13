"""Stopping the bridge from the page it serves.

The process outlives the tab that needed it, and the only way to stop it used
to be Ctrl-C in whichever terminal started it. That is fine when you remember
where that terminal is.

Being able to stop a local process from a web page is worth being careful
about, so what is pinned here is the guard: the same loopback, paired-origin,
constant-time-token check every other route uses, POST only so that no URL on
its own can trigger it, and a refusal while work is still in flight.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import urllib.error
import urllib.request

from goatedit_bridge.jobs import Job, JobStore
from goatedit_bridge.server import BridgeConfig, make_server

TOKEN = "testtoken123"
ORIGIN = "http://localhost:4173"


def _serve(jobs: JobStore):
    config = BridgeConfig(port=0, token=TOKEN, origins=[ORIGIN])
    server = make_server(config, jobs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _post(url: str, headers: dict[str, str] | None = None, body: dict | None = None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read() or b"{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return exc.code, {}


def _get(url: str, headers: dict[str, str] | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _auth():
    return {"Origin": ORIGIN, "Authorization": f"Bearer {TOKEN}"}


def test_requires_the_token():
    with tempfile.TemporaryDirectory() as work_dir:
        server, port = _serve(JobStore(work_dir=work_dir))
        try:
            status, _ = _post(f"http://127.0.0.1:{port}/shutdown", {"Origin": ORIGIN})
            assert status == 401, f"no token should be refused, got {status}"

            status, _ = _post(f"http://127.0.0.1:{port}/shutdown",
                              {"Origin": ORIGIN, "Authorization": "Bearer wrong"})
            assert status == 401, f"wrong token should be refused, got {status}"

            # Still alive.
            status, _ = _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            assert status == 200, "a refused shutdown must not have stopped the bridge"
        finally:
            server.shutdown()
        print("  ✓ shutdown needs the bridge token")


def test_requires_a_paired_origin():
    with tempfile.TemporaryDirectory() as work_dir:
        server, port = _serve(JobStore(work_dir=work_dir))
        try:
            status, _ = _post(f"http://127.0.0.1:{port}/shutdown",
                              {"Origin": "https://evil.example", "Authorization": f"Bearer {TOKEN}"})
            assert status == 403, f"unpaired origin should be refused, got {status}"

            status, _ = _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            assert status == 200, "a refused shutdown must not have stopped the bridge"
        finally:
            server.shutdown()
        print("  ✓ shutdown needs a paired origin")


def test_get_cannot_shut_it_down():
    """The query-token path is GET-only, so it must not reach this route."""
    with tempfile.TemporaryDirectory() as work_dir:
        server, port = _serve(JobStore(work_dir=work_dir))
        try:
            status, _ = _get(f"http://127.0.0.1:{port}/shutdown?token={TOKEN}", {"Origin": ORIGIN})
            assert status == 404, f"GET /shutdown should not be a route, got {status}"

            status, _ = _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            assert status == 200, "a URL alone must not stop the bridge"
        finally:
            server.shutdown()
        print("  ✓ a URL alone cannot stop the bridge")


def test_refuses_while_work_is_running():
    with tempfile.TemporaryDirectory() as work_dir:
        jobs = JobStore(work_dir=work_dir)
        running = Job(id="abc123", url="https://example.com/v")
        running.state = "downloading"
        jobs._jobs[running.id] = running

        server, port = _serve(jobs)
        try:
            status, body = _post(f"http://127.0.0.1:{port}/shutdown", _auth())
            assert status == 409, f"running work should block a plain stop, got {status}"
            assert body.get("busy") == ["abc123"], body

            status, _ = _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            assert status == 200, "a refused shutdown must not have stopped the bridge"
        finally:
            server.shutdown()
        print("  ✓ refuses to stop mid-job unless forced")


def test_force_stops_even_with_work_running():
    with tempfile.TemporaryDirectory() as work_dir:
        jobs = JobStore(work_dir=work_dir)
        running = Job(id="abc123", url="https://example.com/v")
        running.state = "processing"
        jobs._jobs[running.id] = running

        server, port = _serve(jobs)
        status, body = _post(f"http://127.0.0.1:{port}/shutdown", _auth(), {"force": True})
        assert status == 200, f"forced stop should be accepted, got {status}"
        assert body.get("stopping") is True, body
        assert body.get("interrupted") == ["abc123"], body

        # The reply lands before the server goes down; the loop stops right after.
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            except Exception:
                break
            time.sleep(0.1)
        else:
            server.shutdown()
            raise AssertionError("the bridge kept serving after a forced stop")
        print("  ✓ force stops it, and reports what it interrupted")


def test_stops_when_idle():
    with tempfile.TemporaryDirectory() as work_dir:
        server, port = _serve(JobStore(work_dir=work_dir))
        status, body = _post(f"http://127.0.0.1:{port}/shutdown", _auth())
        assert status == 200, f"idle stop should be accepted, got {status}"
        assert body.get("interrupted") == [], body

        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                _get(f"http://127.0.0.1:{port}/health", {"Origin": ORIGIN})
            except Exception:
                break
            time.sleep(0.1)
        else:
            server.shutdown()
            raise AssertionError("the bridge kept serving after a stop")
        print("  ✓ stops when idle")


def main() -> None:
    print("\nSTOPPING THE BRIDGE FROM THE EDITOR")
    print("=" * 41)
    test_requires_the_token()
    test_requires_a_paired_origin()
    test_get_cannot_shut_it_down()
    test_refuses_while_work_is_running()
    test_force_stops_even_with_work_running()
    test_stops_when_idle()
    print("=" * 41)
    print("ALL SHUTDOWN TESTS PASSED!\n")


if __name__ == "__main__":
    main()
