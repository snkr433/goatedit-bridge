"""
The audio half of a frame-stream render, end to end.

The mix used to travel inline as base64 in the /render/start body. A stereo WAV
is 192 KB per second, so ten minutes of timeline is 115 MB and half again as
much once base64'd — past the bridge's request cap, which the browser reported
as a bare "Failed to fetch" with no status behind it. It now goes up as a file
and /render/start is handed the path.
"""
import json, os, subprocess, tempfile, threading, urllib.request, urllib.error
from goatedit_bridge.jobs import JobStore
from goatedit_bridge.server import BridgeConfig, make_server

PORT, TOKEN, ORIGIN = 8801, "t" * 16, "https://ai.goatedit.com"
tmp = tempfile.mkdtemp(prefix="audiopath_probe_")
srv = make_server(BridgeConfig(port=PORT, token=TOKEN, origins=[ORIGIN]), JobStore(work_dir=tmp))
threading.Thread(target=srv.serve_forever, daemon=True).start()


def req(path, method="GET", body=None, ctype="application/json", extra=None):
    headers = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, "Host": f"127.0.0.1:{PORT}"}
    if extra:
        headers.update(extra)
    data = None
    if body is not None:
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        headers["Content-Type"] = "application/json" if isinstance(body, dict) else ctype
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


# A real WAV, so FFmpeg has something to mux.
wav = os.path.join(tmp, "mix.wav")
subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=f=440:d=2", wav],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open(wav, "rb") as f:
    wav_bytes = f.read()

status, up = req("/media/upload", "POST", wav_bytes, "application/octet-stream",
                 {"X-File-Name": "mix.wav"})
assert status == 200, up
print("✓ uploaded mix:", up["filePath"], up["size"], "bytes")

status, session = req("/render/start", "POST", {
    "width": 320, "height": 180, "fps": 30, "totalFrames": 60,
    "codec": "h264_hw", "filename": "with_audio.mp4", "outputDir": tmp,
    "audioPath": up["filePath"],
})
assert status == 201, session
sid = session["sessionId"]
print("✓ session started, encoder:", session["encoder"])

frame = bytes(320 * 180 * 4)
for _ in range(60):
    status, _r = req(f"/render/frame?sessionId={sid}", "POST", frame,
                     "application/octet-stream", {"X-Session-Id": sid})
    assert status == 200, _r

status, done = req("/render/finish", "POST", {"sessionId": sid})
assert status == 200 and done["state"] == "ready", done
print("✓ finished:", done["outputPath"], done["fileSizeBytes"], "bytes")

streams = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", done["outputPath"]],
    capture_output=True, text=True).stdout.split()
assert "audio" in streams, f"no audio stream in the master: {streams}"
print("✓ master carries audio:", streams)

# A missing path has to be an error, not a silent master.
status, err = req("/render/start", "POST", {
    "width": 320, "height": 180, "fps": 30, "totalFrames": 1,
    "codec": "h264_hw", "filename": "nope.mp4", "outputDir": tmp,
    "audioPath": os.path.join(tmp, "does_not_exist.wav"),
})
assert status == 500 and "not found" in err.get("error", ""), (status, err)
print("✓ missing audioPath refused:", err["error"])

# An oversized JSON body gets a 413 with a message, not a reset socket.
huge = json.dumps({"audioBase64": "A" * (65 * 1024 * 1024)}).encode()
status, err = req("/resolve", "POST", huge)
assert status == 413, (status, err)
print("✓ oversized body answered:", err["error"])

srv.shutdown()
print("\nALL AUDIO-PATH CHECKS PASSED")


def check_compiler_reports_ffmpeg_stderr():
    """
    A failed native compile has to say why.

    The progress reader owns FFmpeg's stderr pipe and consumes it a character at
    a time. It used to keep only the `time=` ticks, so by the time the failure
    path went looking for an explanation the pipe was empty and the toast read
    "Compilation failed with code 221:" with nothing after the colon.
    """
    import shutil

    work = tempfile.mkdtemp(prefix="compile_fail_")
    srv2 = make_server(BridgeConfig(port=8802, token=TOKEN, origins=[ORIGIN]), JobStore(work_dir=work))
    threading.Thread(target=srv2.serve_forever, daemon=True).start()

    src = os.path.join(work, "clip.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:d=1:r=30",
                    "-c:v", "libx264", src],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def req2(path, method="GET", body=None):
        headers = {"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN, "Host": "127.0.0.1:8802"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        r = urllib.request.Request(f"http://127.0.0.1:8802{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())

    # Aim the output at a directory. FFmpeg refuses at the point of opening it,
    # which is exactly the class of failure that used to arrive unexplained.
    blocked = os.path.join(work, "in_the_way.mov")
    os.makedirs(blocked, exist_ok=True)

    status, job = req2("/timeline/render", "POST", {
        "width": 320, "height": 180, "fps": 30, "duration": 1.0,
        "codec": "prores_422", "filename": "in_the_way.mov", "outputDir": work,
        "tracks": [{"id": "v1", "type": "video", "clips": [{
            "id": "c1", "mediaId": "m1", "filePath": src,
            "startTime": 0.0, "duration": 1.0, "sourceIn": 0.0, "sourceOut": 1.0,
        }]}],
    })
    assert status == 201, job

    import time as _time
    for _ in range(100):
        _time.sleep(0.05)
        _, st = req2(f"/timeline/status/{job['jobId']}")
        if st["state"] in ("ready", "error"):
            break

    assert st["state"] == "error", f"expected the compile to fail, got {st['state']}"
    error = st.get("error") or ""
    # The exit code alone is what this test exists to reject.
    assert error.split(":", 1)[-1].strip(), f"failure carried no explanation: {error!r}"
    print("✓ a failed compile reports FFmpeg's own words:", error[:160])

    srv2.shutdown()
    shutil.rmtree(work, ignore_errors=True)


check_compiler_reports_ffmpeg_stderr()
print("ALL RENDER-AUDIO CHECKS PASSED")
