"""Test loopback HTTP API endpoints for the full hardware offload suite."""

import json
import os
import subprocess
import tempfile
import time
import urllib.request
from goatedit_bridge.jobs import JobStore
from goatedit_bridge.server import BridgeConfig, make_server

def main():
    temp_dir = tempfile.mkdtemp(prefix="goatedit_api_test_")
    token = "test_token_123456"
    port = 8769
    origin = "https://ai.goatedit.com"

    # Generate synthetic video asset
    src_video = os.path.join(temp_dir, "test_clip.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=green:s=640x360:d=2:r=30",
        "-f", "lavfi", "-i", "sine=f=440:d=2",
        "-c:v", "libx264", "-c:a", "aac",
        src_video
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    config = BridgeConfig(port=port, token=token, origins=[origin])
    jobs = JobStore(work_dir=temp_dir)
    server = make_server(config, jobs)

    import threading
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Test server running on port {port}...")

    def req(path, method="GET", body=None, content_type="application/json", extra_headers=None):
        url = f"http://127.0.0.1:{port}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Origin": origin,
            "Host": f"127.0.0.1:{port}",
        }
        if extra_headers:
            headers.update(extra_headers)
        data = None
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
            else:
                data = body
                if content_type:
                    headers["Content-Type"] = content_type

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(request) as resp:
            return resp.status, json.loads(resp.read().decode())

    # 0. Preflight. Every custom header the editor sends has to be named in
    # Access-Control-Allow-Headers or the browser refuses the request before it
    # is made, and the page sees a bare "Failed to fetch" with no status behind
    # it. /media/upload sends X-File-Name; the frame stream sends X-Session-Id.
    preflight = urllib.request.Request(
        f"http://127.0.0.1:{port}/media/upload",
        method="OPTIONS",
        headers={
            "Origin": origin,
            "Host": f"127.0.0.1:{port}",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-file-name",
            "Access-Control-Request-Private-Network": "true",
        },
    )
    with urllib.request.urlopen(preflight) as resp:
        allowed = {h.strip().lower() for h in (resp.headers.get("Access-Control-Allow-Headers") or "").split(",")}
        assert resp.status == 204, f"preflight returned {resp.status}"
        for header in ("authorization", "content-type", "x-session-id", "x-file-name"):
            assert header in allowed, f"preflight does not allow {header}: {sorted(allowed)}"
        assert resp.headers.get("Access-Control-Allow-Private-Network") == "true", \
            "Chrome preflights a public page reaching 127.0.0.1 and needs this opt-in"
    print("✓ OPTIONS preflight allows every header the editor sends")

    # 1. Test /hardware/info
    status, data = req("/hardware/info")
    assert status == 200, f"/hardware/info failed: {data}"
    assert data["hasHardwareAcceleration"] is True
    print("✓ GET /hardware/info passed:", data["gpuName"])

    # 2. Test /timeline/render
    timeline_payload = {
        "width": 640,
        "height": 360,
        "fps": 30,
        "duration": 2.0,
        "codec": "prores_422",
        "filename": "api_test_export.mov",
        "outputDir": temp_dir,
        "tracks": [
            {
                "id": "v1",
                "type": "video",
                "clips": [
                    {
                        "id": "c1",
                        "mediaId": "m1",
                        "filePath": src_video,
                        "startTime": 0.0,
                        "duration": 2.0,
                        "sourceIn": 0.0,
                        "sourceOut": 2.0,
                    }
                ]
            }
        ]
    }
    status, job = req("/timeline/render", method="POST", body=timeline_payload)
    assert status == 201, f"/timeline/render failed: {job}"
    job_id = job["jobId"]
    print(f"✓ POST /timeline/render created job {job_id}")

    # Poll /timeline/status/<id>
    for _ in range(50):
        time.sleep(0.05)
        _, st = req(f"/timeline/status/{job_id}")
        if st["state"] in ("ready", "error"):
            break

    assert st["state"] == "ready", f"Timeline render failed: {st.get('error')}"
    print("✓ GET /timeline/status/<id> completed ready:", st["outputPath"])

    # 3. Test /proxy/generate
    status, pjob = req("/proxy/generate", method="POST", body={"sourcePath": src_video, "height": 360})
    assert status == 200, f"/proxy/generate failed: {pjob}"
    proxy_id = pjob["jobId"]
    print(f"✓ POST /proxy/generate created proxy job {proxy_id}")

    for _ in range(50):
        time.sleep(0.05)
        _, pst = req(f"/proxy/status/{proxy_id}")
        if pst["state"] in ("ready", "error"):
            break

    assert pst["state"] == "ready", f"Proxy failed: {pst.get('error')}"
    print("✓ GET /proxy/status/<id> ready:", pst["targetPath"])

    # 4. Test /ai/silence-detect
    status, s_res = req("/ai/silence-detect", method="POST", body={"filePath": src_video})
    assert status == 200, f"/ai/silence-detect failed: {s_res}"
    print(f"✓ POST /ai/silence-detect passed: detected {len(s_res['speechSegments'])} speech segments")

    # 5. Test /media/upload
    test_bytes = b"FAKE_VIDEO_STREAM_BYTES_FOR_UPLOAD_TEST"
    status, up_res = req(
        "/media/upload",
        method="POST",
        body=test_bytes,
        content_type="application/octet-stream",
        extra_headers={"X-File-Name": "uploaded_test.mp4"}
    )
    assert status == 200, f"/media/upload failed: {up_res}"
    assert os.path.exists(up_res["filePath"]), "Uploaded file not saved"
    print("✓ POST /media/upload passed:", up_res["filePath"])

    # 6. Test /media/check
    status, chk_res = req("/media/check", method="POST", body={"filename": "uploaded_test.mp4", "size": len(test_bytes)})
    assert status == 200 and chk_res.get("exists") is True, f"/media/check failed: {chk_res}"
    print("✓ POST /media/check passed: file detected on disk without re-upload!")

    # 7. Test /media/locate
    status, loc_res = req("/media/locate", method="POST", body={"filenames": ["uploaded_test.mp4", "non_existent_12345.mp4"]})
    assert status == 200, f"/media/locate failed: {loc_res}"
    assert "uploaded_test.mp4" in loc_res.get("results", {}), "Expected uploaded_test.mp4 to be located"
    print("✓ POST /media/locate passed: host search successfully resolved media file!")

    server.shutdown()
    print("\n=========================================")
    print("ALL HTTP LOOPBACK ENDPOINTS TESTED AND VERIFIED!")
    print("=========================================")

if __name__ == "__main__":
    main()
