"""The local-file routes as the browser meets them.

test_localfs.py proves the path logic. This proves the guards in front of it
are actually wired to these routes — a containment check that a route forgets
to call is worth nothing, and that is the failure this catches.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request

import pytest

from goatedit_bridge.jobs import JobStore
from goatedit_bridge.localfs import LocalFs
from goatedit_bridge.server import BridgeConfig, make_server

ORIGIN = "http://localhost:5173"
TOKEN = "test-token"


@pytest.fixture()
def server(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "clip.mp4").write_bytes(b"video bytes")

    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "id_rsa").write_text("PRIVATE KEY")

    work = tempfile.mkdtemp()
    localfs = LocalFs([str(shared)], os.path.join(work, "frames"))
    config = BridgeConfig(port=0, token=TOKEN, origins=[ORIGIN], localfs=localfs)
    httpd = make_server(config, JobStore(work, keep_files=False))
    config.port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", secret
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None, token=TOKEN, origin=ORIGIN, host=None):
    url = f"{base}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if origin:
        req.add_header("Origin", origin)
    if host:
        req.add_header("Host", host)
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# --- the guards are actually in front of these routes ---------

def test_wrong_token_is_refused(server):
    base, _ = server
    status, _ = call(base, "/local/open", {"filename": "clip.mp4"}, token="wrong")
    assert status == 401


def test_unpaired_origin_is_refused(server):
    base, _ = server
    status, _ = call(base, "/local/open", {"filename": "clip.mp4"}, origin="https://evil.example")
    assert status == 403


def test_non_loopback_host_header_is_refused(server):
    # DNS rebinding: a name that resolves to 127.0.0.1 still arrives with its
    # own Host header, and that is what gets turned away.
    base, _ = server
    status, _ = call(base, "/local/open", {"filename": "clip.mp4"}, host="evil.example")
    assert status == 403


# --- the routes themselves ------------------------------------

def test_open_returns_a_handle_url_and_not_a_path(server):
    base, _ = server
    status, raw = call(base, "/local/open", {"filename": "clip.mp4"})
    assert status == 200
    body = json.loads(raw)
    assert body["name"] == "clip.mp4"
    assert body["size"] == len(b"video bytes")
    assert "/local/file/" in body["url"]
    # The browser is never handed a filesystem path for a read.
    assert "path" not in body


def test_the_handle_serves_the_bytes(server):
    base, _ = server
    body = json.loads(call(base, "/local/open", {"filename": "clip.mp4"})[1])
    status, raw = call(base, f"/local/file/{body['handle']}")
    assert status == 200
    assert raw == b"video bytes"


def test_a_forged_handle_serves_nothing(server):
    base, _ = server
    status, _ = call(base, "/local/file/made-up-handle")
    assert status == 404


def test_a_secret_outside_the_shared_root_is_refused_over_the_wire(server):
    base, secret = server
    for attempt in ("id_rsa", str(secret / "id_rsa"), "../secret/id_rsa"):
        status, raw = call(base, "/local/open", {"filename": attempt})
        assert status == 400, attempt
        assert b"PRIVATE KEY" not in raw


def test_frame_write_returns_a_path_under_the_frames_dir(server):
    base, _ = server
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 16).decode()
    status, raw = call(base, "/local/frame", {"data": png, "contentType": "image/png", "label": "t3"})
    assert status == 200
    body = json.loads(raw)
    # Here the path IS returned: it is what the agent reads with its own tools
    # instead of carrying the image through the conversation.
    assert body["path"].endswith(".png")
    assert os.path.isfile(body["path"])
    assert "frames" in body["path"]


def test_roots_are_discoverable_so_the_panel_can_say_what_is_shared(server):
    base, _ = server
    status, raw = call(base, "/local/roots")
    assert status == 200
    body = json.loads(raw)
    assert body["enabled"] is True
    assert len(body["roots"]) == 1


def test_routes_are_off_entirely_when_no_folder_was_shared(tmp_path):
    work = tempfile.mkdtemp()
    localfs = LocalFs([], os.path.join(work, "frames"))
    config = BridgeConfig(port=0, token=TOKEN, origins=[ORIGIN], localfs=localfs)
    httpd = make_server(config, JobStore(work, keep_files=False))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, raw = call(base, "/local/open", {"filename": "clip.mp4"})
        assert status == 403
        assert b"--allow-dir" in raw
    finally:
        httpd.shutdown()
        httpd.server_close()
