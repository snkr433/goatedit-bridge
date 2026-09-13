"""The loopback HTTP server the plugin panel talks to.

Threat model, because a server on localhost is a server every page in the
browser can try to reach:

  * The socket binds 127.0.0.1, so nothing off this machine can connect.
  * Every request must carry `Authorization: Bearer <token>`. The token is
    generated per run and printed once; the user pastes it into the plugin.
  * Every browser request must carry an `Origin` we were started with
    (https://ai.goatedit.com by default). That is what stops evil.com from
    firing no-preflight requests at us, and — together with the Host check —
    what stops DNS rebinding.
  * The `Host` header must itself be loopback, so a rebound name that resolves
    to 127.0.0.1 still gets turned away.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import platform
import re
import secrets
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__
from .ai_engine import detect_silences
from .jobs import PREVIEW_MAX_DURATION, JobStore, _clock, ffmpeg_available, probe, storyboard
from .localfs import LocalFs, LocalFsError, contains
from .proxies import needs_proxy, proxy_manager
from .render import probe_hardware, render_manager
from .timeline_compiler import ClipSpec, TimelineSpec, TrackSpec, timeline_manager

MAX_REQUEST_BODY = 64 * 1024 * 1024
FILE_CHUNK = 1024 * 1024

ALLOWED_SCHEMES = ("http", "https")


def _section(body: dict[str, Any]) -> tuple[float, float] | None:
    """The (start, end) the panel asked for, or None for the whole video.

    Both ends have to be real numbers and in order, because they are handed to
    ffmpeg as a seek range: a reversed or non-finite pair produces a file that
    is empty or never finishes rather than an error the user can read.
    """
    start_raw, end_raw = body.get("start"), body.get("end")
    if start_raw is None and end_raw is None:
        return None
    try:
        start_at, end_at = float(start_raw or 0), float(end_raw)
    except (TypeError, ValueError):
        raise ValueError("start and end must be numbers of seconds") from None
    if not (math.isfinite(start_at) and math.isfinite(end_at)):
        raise ValueError("start and end must be finite")
    if start_at < 0 or end_at <= start_at:
        raise ValueError("end must be greater than start, and start cannot be negative")
    return (start_at, end_at)


def _max_height(body: dict[str, Any]) -> int:
    """The tallest picture the caller will accept, or 0 for no ceiling.

    A caller that knows the shot is going to be six muted seconds of b-roll can
    say so and skip the 4K master. Rejected rather than clamped when it is not a
    sane number: silently downloading something other than what was asked for is
    how a caller ends up trusting a cap that never applied.
    """
    raw = body.get("maxHeight")
    if raw is None:
        return 0
    try:
        height = int(raw)
    except (TypeError, ValueError):
        raise ValueError("maxHeight must be a number of pixels") from None
    if height < 0:
        raise ValueError("maxHeight cannot be negative")
    return height


# Directories we have already resolved media out of. Ranked ahead of the generic
# home scan and walked deeper, because the folder that held one clip almost
# always holds the rest of the project. Bounded and LRU-evicted: an unbounded
# set would make every later request walk every folder ever touched.
_MAX_KNOWN_DIRS = 64
_KNOWN_PROJECT_DIRS: "OrderedDict[str, None]" = OrderedDict()

# The walk is the expensive part, so its result is reused for a short window.
# A project directory that appears mid-request invalidates the cache, which is
# what keeps a freshly discovered folder from being missed by the next call.
_INDEX_TTL_SECONDS = 60.0
_INDEX_LOCK = threading.Lock()
_INDEX_CACHE: dict[str, Any] = {"key": None, "built_at": 0.0, "index": None}

_EXT_CLASSES: dict[str, frozenset[str]] = {
    "video": frozenset({".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}),
    "audio": frozenset({".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg"}),
    "image": frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff"}),
    "doc": frozenset({".html", ".htm"}),
}
_MEDIA_EXTENSIONS = frozenset().union(*_EXT_CLASSES.values())

_SKIP_DIR_NAMES = frozenset({
    "Library", "Applications", ".git", "node_modules",
    "DerivedData", "Pictures", "Music", "System",
    # Proxies sit beside the footage they came from, so a media walk would
    # otherwise index them as findable sources. Locate has to answer with the
    # original: relinking a project to its own proxies would silently pin the
    # edit to 540p, including on export.
    "Proxies",
})


def _ext_class(ext: str) -> str | None:
    for cls, exts in _EXT_CLASSES.items():
        if ext in exts:
            return cls
    return None


def _normalize_stem(stem: str) -> str:
    """Reduces a filename stem to comparable letters and digits."""
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def _known_dirs() -> list[str]:
    with _INDEX_LOCK:
        return list(_KNOWN_PROJECT_DIRS)


def _remember_dir(dir_path: str) -> bool:
    """Promotes a directory to the priority set. True when it was not already there."""
    with _INDEX_LOCK:
        if dir_path in _KNOWN_PROJECT_DIRS:
            _KNOWN_PROJECT_DIRS.move_to_end(dir_path)
            return False
        _KNOWN_PROJECT_DIRS[dir_path] = None
        while len(_KNOWN_PROJECT_DIRS) > _MAX_KNOWN_DIRS:
            _KNOWN_PROJECT_DIRS.popitem(last=False)
        return True


class _FileIndex:
    """Filename lookup built from a single directory walk.

    Two tiers, because a real filename must never lose to some other file's
    tolerant alias: `exact` holds names as they are on disk, `fuzzy` holds the
    forgiving variants (hyphen-stripped, timestamp prefix, upload-hash suffix).
    Inside a tier the first writer wins, so directories walked earlier — the
    project folders we already pulled media from — outrank the generic scan.
    """

    def __init__(self) -> None:
        self.exact: dict[str, str] = {}
        self.fuzzy: dict[str, str] = {}
        # class -> [(normalized stem, basename, path)], for the last-resort match.
        self.stems: dict[str, list[tuple[str, str, str]]] = {}

    def add(self, name: str, path: str) -> None:
        ext = os.path.splitext(name)[1].lower()
        cls = _ext_class(ext)
        if cls is None:
            return

        self.exact.setdefault(name, path)
        self.exact.setdefault(name.lower(), path)
        self.fuzzy.setdefault(name.replace("-", "").lower(), path)

        if "_" in name:
            # "20240612_103500_take1.mov" answers to its leading timestamp,
            # and "<uploadhash>_clip.mov" answers to the name it was uploaded as.
            prefix = "_".join(name.split("_")[:3])
            if prefix:
                self.fuzzy.setdefault(prefix, path)
            suffix = name.split("_", 1)[1]
            if suffix:
                self.fuzzy.setdefault(suffix, path)
                self.fuzzy.setdefault(suffix.lower(), path)
                self.fuzzy.setdefault(suffix.replace("-", "").lower(), path)

        stem = _normalize_stem(os.path.splitext(name)[0])
        if stem:
            self.stems.setdefault(cls, []).append((stem, name, path))

    def lookup(self, clean_name: str) -> str | None:
        lower = clean_name.lower()
        prefix = "_".join(clean_name.split("_")[:3]) if "_" in clean_name else ""
        return (
            self.exact.get(clean_name)
            or self.exact.get(lower)
            or self.fuzzy.get(clean_name)
            or self.fuzzy.get(lower)
            or self.fuzzy.get(lower.replace("-", ""))
            or (self.fuzzy.get(prefix) if prefix else None)
        )

    def match_stem(self, clean_name: str) -> str | None:
        """Last resort: closest stem among files of the same media class.

        Covers the names an editor invents that no exact rule can reach — a
        label prefix the timeline added ("HTML-quiz.html" for `quiz.html`), or a
        sound effect stored under a longer name than the timeline remembers.
        Scored and sorted rather than first-hit, so the answer does not depend
        on dictionary order.
        """
        raw_stem, ext = os.path.splitext(clean_name)
        cls = _ext_class(ext.lower())
        if cls is None:
            return None

        variants = {_normalize_stem(raw_stem)}
        for sep in ("-", "_"):
            if sep in raw_stem:
                variants.add(_normalize_stem(raw_stem.split(sep, 1)[1]))
        variants = {v for v in variants if len(v) >= 3}
        if not variants:
            return None

        matches: list[tuple[int, str, str]] = []
        for cand_stem, base, path in self.stems.get(cls, ()):
            score = 0
            for v in variants:
                # Partial rules need real overlap on both sides: a one-letter
                # stem like "q" is a prefix of half the library.
                overlap = min(len(v), len(cand_stem))
                if cand_stem == v:
                    score = max(score, 3)
                elif overlap < 4:
                    continue
                elif cand_stem.startswith(v) or v.startswith(cand_stem):
                    score = max(score, 2)
                elif v in cand_stem or cand_stem in v:
                    score = max(score, 1)
            if score:
                matches.append((score, base, path))
        if not matches:
            return None
        # Highest score, then the shortest (least embellished) name, then stable.
        matches.sort(key=lambda m: (-m[0], len(m[1]), m[1], m[2]))
        return matches[0][2]


def _index_directory(
    dir_path: str,
    index: _FileIndex,
    max_depth: int = 2,
    current_depth: int = 0,
) -> None:
    """Recursively scans a candidate directory into an in-memory index."""
    if current_depth > max_depth or not os.path.isdir(dir_path):
        return
    try:
        with os.scandir(dir_path) as entries:
            for entry in entries:
                if entry.name.startswith("."):
                    continue
                if entry.is_file():
                    index.add(entry.name, entry.path)
                elif entry.is_dir() and entry.name not in _SKIP_DIR_NAMES:
                    _index_directory(entry.path, index, max_depth, current_depth + 1)
    except Exception:
        pass


def _build_file_index(priority_dirs: list[str], candidate_dirs: list[str]) -> _FileIndex:
    """Walks the search roots once, reusing the previous walk while it is fresh."""
    key = (tuple(priority_dirs), tuple(sorted(candidate_dirs)))
    now = time.time()
    with _INDEX_LOCK:
        cached = _INDEX_CACHE.get("index")
        if (
            cached is not None
            and _INDEX_CACHE.get("key") == key
            and now - float(_INDEX_CACHE.get("built_at") or 0.0) < _INDEX_TTL_SECONDS
        ):
            return cached  # type: ignore[return-value]

    index = _FileIndex()
    # Priority dirs first and deeper: first writer wins, so they win collisions.
    for d in priority_dirs:
        _index_directory(d, index, max_depth=3)
    priority_set = set(priority_dirs)
    for d in candidate_dirs:
        if d not in priority_set:
            _index_directory(d, index, max_depth=2)

    with _INDEX_LOCK:
        _INDEX_CACHE["key"] = key
        _INDEX_CACHE["built_at"] = time.time()
        _INDEX_CACHE["index"] = index
    return index


def _invalidate_file_index() -> None:
    with _INDEX_LOCK:
        _INDEX_CACHE["key"] = None


class BridgeConfig:
    def __init__(
        self,
        port: int,
        token: str,
        origins: list[str],
        localfs: LocalFs | None = None,
    ) -> None:
        self.port = port
        self.token = token
        self.origins = origins
        # None when the operator shared no folders. Every local-file route then
        # answers 403: the capability is absent rather than empty, so it cannot
        # be turned on by asking nicely.
        self.localfs = localfs


def _host_is_loopback(host_header: str) -> bool:
    host = host_header.rsplit(":", 1)[0].strip("[]").lower()
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = f"goatedit-bridge/{__version__}"
    protocol_version = "HTTP/1.1"

    config: BridgeConfig
    jobs: JobStore

    # Set by _read_json when a body exceeds MAX_REQUEST_BODY, so the route can
    # answer 413 rather than treat an unread body as an empty one.
    _body_too_large: bool = False

    # --- plumbing ------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # One tidy line per request instead of BaseHTTPRequestHandler's stderr spray.
        print(f"  {self.command} {self.path} → {args[1] if len(args) > 1 else ''}")

    def _origin_ok(self) -> str | None:
        """Returns the Origin to echo back, or None when it isn't allowed."""
        origin = self.headers.get("Origin")
        if origin is None:
            # Not a browser request (curl, a health probe). The token still gates it.
            return ""
        return origin if origin in self.config.origins else None

    def _send_json(self, status: int, payload: dict[str, Any], origin: str = "") -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors(origin)
        self.end_headers()
        self.wfile.write(body)

    def _send_cors(self, origin: str) -> None:
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        # Every custom header the editor sends has to be named here or the
        # browser refuses the request before it is made, and the only thing the
        # page sees is a bare "Failed to fetch" with no status behind it.
        # /media/upload sends X-File-Name; the frame stream sends X-Session-Id.
        self.send_header(
            "Access-Control-Allow-Headers",
            "authorization, content-type, x-session-id, x-file-name, x-frame-count",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        # A local downloader has no business being framed or sniffed.
        self.send_header("X-Content-Type-Options", "nosniff")

    def _guard(self) -> str | None:
        """Runs every check a request must pass. Returns the Origin, or None if it was answered with an error."""
        if not _host_is_loopback(self.headers.get("Host", "")):
            self._send_json(403, {"error": "This bridge only answers on 127.0.0.1"})
            return None
        origin = self._origin_ok()
        if origin is None:
            self._send_json(403, {"error": "That origin is not paired with this bridge"})
            return None
        auth = self.headers.get("Authorization", "")
        expected = f"Bearer {self.config.token}"
        # Constant-time: the token is the only secret here.
        if secrets.compare_digest(auth, expected):
            return origin

        # A <video src="..."> cannot send an Authorization header, and a proxy
        # exists precisely to be played by one. Without this, every proxy URL we
        # hand the editor answers 401 to the element that has to play it, while
        # working fine from fetch() — which is exactly how it shipped past a
        # unit test and was caught only end to end.
        #
        # Accepted on GET only, so nothing that changes state can be triggered
        # by a URL alone, and still compared in constant time. The token is
        # already a loopback-only secret; putting it in a query string on
        # 127.0.0.1 does not widen who can reach it.
        if self.command == "GET":
            query = parse_qs(urlparse(self.path).query)
            supplied = (query.get("token") or [""])[0]
            if supplied and secrets.compare_digest(supplied, self.config.token):
                return origin

        self._send_json(401, {"error": "Missing or wrong bridge token"}, origin)
        return None

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_REQUEST_BODY:
            # Drain before answering. Replying to a body still being uploaded
            # and then closing resets the connection, and a reset is all the
            # browser reports: `fetch` rejects with a bare "Failed to fetch",
            # no status, nothing to show the person exporting. Reading it to
            # the end costs a moment and buys a real 413.
            remaining = length
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, FILE_CHUNK))
                if not chunk:
                    break
                remaining -= len(chunk)
            self._body_too_large = True
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # --- routes --------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        origin = self._origin_ok()
        if origin is None:
            self._send_json(403, {"error": "That origin is not paired with this bridge"})
            return
        self.send_response(204)
        self._send_cors(origin)
        # Private Network Access: Chrome treats a public page reaching 127.0.0.1
        # as a private-network request and preflights it with this header. Without
        # the matching opt-in the browser drops the request before we ever see it.
        # It is not a weakening of anything — the Origin and token checks below
        # still decide who gets an answer.
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        # /health is the pairing probe: it answers without a token so the panel
        # can say "bridge found, paste your token" instead of "unreachable".
        # It leaks only that a bridge is running, and only to a paired origin.
        if path == "/health":
            origin = self._origin_ok()
            if origin is None:
                self._send_json(403, {"error": "That origin is not paired with this bridge"})
                return
            self._send_json(200, {
                "name": "goatedit-bridge",
                "version": __version__,
                "ffmpeg": ffmpeg_available(),
            }, origin)
            return

        origin = self._guard()
        if origin is None:
            return

        if path.startswith("/job/"):
            job = self.jobs.get(path[len("/job/"):])
            if not job:
                self._send_json(404, {"error": "No such job"}, origin)
                return
            self._send_json(200, job.public(), origin)
            return

        if path.startswith("/local/file/"):
            self._serve_handle(path[len("/local/file/"):], origin)
            return

        if path == "/local/roots":
            fs = self.config.localfs
            self._send_json(200, {
                "enabled": bool(fs and fs.roots),
                "roots": list(fs.roots) if fs else [],
                "framesDir": fs.frames_dir if fs else "",
            }, origin)
            return

        if path.startswith("/file/"):
            self._serve_file(path[len("/file/"):], origin)
            return

        if path.startswith("/file-info/"):
            job_id = path[len("/file-info/"):]
            path_disk = None
            job = self.jobs.get(job_id)
            if job and job.state == "ready" and job.filepath:
                path_disk = job.filepath
            else:
                pjob = proxy_manager.get(job_id)
                if pjob and pjob.state == "ready" and pjob.target_path:
                    path_disk = pjob.target_path
            if path_disk and os.path.exists(path_disk):
                self._send_json(200, {"filePath": path_disk, "size": os.path.getsize(path_disk)}, origin)
            else:
                self._send_json(404, {"error": "File info not found"}, origin)
            return

        if path == "/hardware/info":
            self._send_json(200, probe_hardware(), origin)
            return

        if path.startswith("/render/status/"):
            session_id = path[len("/render/status/"):]
            session = render_manager.get(session_id)
            if not session:
                self._send_json(404, {"error": "No such render session"}, origin)
                return
            self._send_json(200, session.public(), origin)
            return

        if path.startswith("/timeline/status/"):
            job_id = path[len("/timeline/status/"):]
            job = timeline_manager.get(job_id)
            if not job:
                self._send_json(404, {"error": "No such timeline job"}, origin)
                return
            self._send_json(200, job.public(), origin)
            return

        if path.startswith("/proxy/status/"):
            job_id = path[len("/proxy/status/"):]
            job = proxy_manager.get(job_id)
            if not job:
                self._send_json(404, {"error": "No such proxy job"}, origin)
                return
            self._send_json(200, job.public(), origin)
            return

        self._send_json(404, {"error": f"No route for {path}"}, origin)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        origin = self._guard()
        if origin is None:
            return
        self._body_too_large = False

        if path == "/shutdown":
            # Stopping the bridge from the page it serves. The only way to stop
            # it used to be Ctrl-C in whichever terminal it was started from,
            # which is fine when you remember where that is and unhelpful
            # otherwise — the process outlives the tab that needed it.
            #
            # POST, so no URL alone can trigger it, and behind the same guard as
            # every other route: loopback host, paired origin, constant-time
            # token. A page that can already ask this bridge to transcode is not
            # being handed anything new by being able to ask it to stop.
            body = self._read_json()
            busy = self.jobs.busy()
            if busy and not body.get("force"):
                # Work in flight is worth one round trip. The caller decides;
                # it just does not get to decide by accident.
                self._send_json(409, {
                    "error": "Jobs are still running",
                    "busy": busy,
                }, origin)
                return
            self._send_json(200, {"stopping": True, "interrupted": busy}, origin)
            # After the reply is on the wire, never before: shutdown() blocks
            # until the serve loop exits, and calling it inline would deadlock
            # this very request.
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        if path == "/render/frame":
            session_id = self.headers.get("X-Session-Id") or parse_qs(urlparse(self.path).query).get("sessionId", [None])[0]
            if not session_id:
                self._send_json(400, {"error": "Missing X-Session-Id header or query parameter"}, origin)
                return
            session = render_manager.get(session_id)
            if not session:
                self._send_json(404, {"error": "Session not found or expired"}, origin)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send_json(400, {"error": "Empty frame body"}, origin)
                return
            frame_bytes = self.rfile.read(length)
            # An encoded chunk can stand for more than one frame, and only the
            # caller knows how many; a raw frame is always exactly one.
            try:
                frames = max(1, int(self.headers.get("X-Frame-Count") or 1))
            except ValueError:
                frames = 1
            try:
                written = session.push_frame(frame_bytes, frames)
                self._send_json(200, {"framesWritten": written, "progress": session.progress}, origin)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path == "/media/upload":
            raw_filename = self.headers.get("X-File-Name") or parse_qs(urlparse(self.path).query).get("filename", ["upload.mp4"])[0]
            clean_name = os.path.basename(raw_filename)
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._send_json(400, {"error": "Empty upload payload"}, origin)
                return
            upload_dir = os.path.join(self.jobs.work_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            dest_path = os.path.join(upload_dir, f"{secrets.token_hex(8)}_{clean_name}")
            try:
                with open(dest_path, "wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(remaining, FILE_CHUNK))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                self._send_json(200, {"filePath": dest_path, "size": os.path.getsize(dest_path)}, origin)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path in ("/local/open", "/local/frame"):
            fs = self.config.localfs
            if fs is None or not fs.roots:
                self._send_json(403, {
                    "error": "This bridge shares no folders. Restart it with --allow-dir <folder> "
                             "for each folder the editor may read.",
                }, origin)
                return

            body = self._read_json()
            if self._body_too_large:
                self._body_too_large = False
                self._send_json(413, {"error": "That request body is too large."}, origin)
                return

            try:
                if path == "/local/open":
                    # The caller names a file, never a directory. What comes
                    # back is a handle plus the numbers the bin needs — not the
                    # path, which the browser has no use for and should not be
                    # taught to pass around.
                    found = fs.resolve(str(body.get("filename", "")))
                    ctype = mimetypes.guess_type(found)[0] or "application/octet-stream"
                    token = fs.mint(found, ctype)
                    self._send_json(200, {
                        "handle": token,
                        "url": f"http://127.0.0.1:{self.config.port}/local/file/{token}",
                        "name": os.path.basename(found),
                        "size": os.path.getsize(found),
                        "contentType": ctype,
                    }, origin)
                    return

                written = fs.write_frame(
                    str(body.get("data", "")),
                    str(body.get("contentType", "image/jpeg")),
                    str(body.get("label", "")),
                )
                # The path IS the point here: it is what the agent reads back
                # with its own file tools instead of carrying the image through
                # the transcript.
                self._send_json(200, {
                    "path": written.path,
                    "size": written.size,
                    "contentType": written.content_type,
                }, origin)
                return
            except LocalFsError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return

        if path == "/media/check":
            body = self._read_json()
            filename = os.path.basename(str(body.get("filename", "")).strip())
            size = int(body.get("size", 0))
            upload_dir = os.path.join(self.jobs.work_dir, "uploads")
            if os.path.exists(upload_dir) and filename:
                for entry in os.listdir(upload_dir):
                    if entry.endswith(f"_{filename}") or entry == filename:
                        fp = os.path.join(upload_dir, entry)
                        if os.path.isfile(fp):
                            if size <= 0 or os.path.getsize(fp) == size:
                                self._send_json(200, {"exists": True, "filePath": fp, "size": os.path.getsize(fp)}, origin)
                                return
            self._send_json(200, {"exists": False}, origin)
            return

        if path == "/media/locate":
            body = self._read_json()
            raw_filenames = body.get("filenames")
            if not raw_filenames:
                single = body.get("filename")
                raw_filenames = [single] if single else []

            # 1. Harvest caller-supplied search directories (directory affinity)
            search_dirs = body.get("search_dirs") or body.get("searchDirs") or []
            if isinstance(search_dirs, str):
                search_dirs = [search_dirs]
            for s_dir in search_dirs:
                if s_dir and os.path.isdir(s_dir):
                    _remember_dir(os.path.abspath(s_dir))

            results: dict[str, str] = {}
            priority_dirs = _known_dirs()
            candidate_dirs: set[str] = set(priority_dirs)
            home = os.path.expanduser("~")

            # Common media roots
            for d in (
                os.path.join(home, "Downloads"),
                os.path.join(home, "Movies"),
                os.path.join(home, "Desktop"),
                os.path.join(home, "Documents"),
                self.jobs.work_dir,
                os.path.join(self.jobs.work_dir, "uploads"),
            ):
                if os.path.isdir(d):
                    candidate_dirs.add(d)

            # Discover 1-level user folders in home
            try:
                for item in os.listdir(home):
                    if item.startswith(".") or item in _SKIP_DIR_NAMES:
                        continue
                    p = os.path.join(home, item)
                    if os.path.isdir(p):
                        candidate_dirs.add(p)
            except Exception:
                pass

            # 2. One walk of every search root, reused across nearby requests
            file_index = _build_file_index(priority_dirs, sorted(candidate_dirs))

            # 3. In-memory resolution for all requested files
            unresolved_media: list[str] = []
            for raw_name in raw_filenames:
                clean_name = os.path.basename(str(raw_name).strip())
                if not clean_name:
                    continue

                ext = os.path.splitext(clean_name)[1].lower()
                found_path = file_index.lookup(clean_name) or file_index.match_stem(clean_name)

                if found_path and os.path.isfile(found_path):
                    results[clean_name] = found_path
                    parent_dir = os.path.dirname(found_path)
                    if _remember_dir(parent_dir):
                        # Fold the rest of this project in now; the cached walk
                        # predates the discovery, so retire it for the next call.
                        _index_directory(parent_dir, file_index, max_depth=2)
                        _invalidate_file_index()
                elif ext in _MEDIA_EXTENSIONS:
                    unresolved_media.append(clean_name)

            # 4. Spotlight search on macOS (ONLY for actual missing media files, capped)
            if unresolved_media and platform.system() == "Darwin":
                for clean_name in unresolved_media[:10]:
                    # mdfind's query language is its own; a quote or backslash in
                    # the name would otherwise truncate the predicate.
                    escaped = clean_name.replace("\\", "\\\\").replace('"', '\\"')
                    try:
                        mdfind_out = subprocess.check_output(
                            ["mdfind", f'kMDItemFSName == "{escaped}"'],
                            text=True,
                            timeout=1,
                        ).strip().split("\n")
                        for candidate_path in mdfind_out:
                            candidate_path = candidate_path.strip()
                            if candidate_path and os.path.isfile(candidate_path):
                                results[clean_name] = candidate_path
                                pdir = os.path.dirname(candidate_path)
                                if _remember_dir(pdir):
                                    _index_directory(pdir, file_index, max_depth=2)
                                    _invalidate_file_index()
                                break
                    except Exception:
                        pass

            self._send_json(200, {"results": results}, origin)
            return

        body = self._read_json()
        if self._body_too_large:
            self._send_json(
                413,
                {"error": f"Request body is larger than the {MAX_REQUEST_BODY // (1024 * 1024)} MB limit. "
                          "Upload large payloads to /media/upload and send the path instead."},
                origin,
            )
            return
        url = str(body.get("url") or "").strip()
        if path in ("/resolve", "/download", "/storyboard", "/preview"):
            if not url or urlparse(url).scheme not in ALLOWED_SCHEMES:
                self._send_json(400, {"error": "Give me an http(s) URL to work on"}, origin)
                return

        if path == "/resolve":
            try:
                self._send_json(200, probe(url), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
            return

        if path == "/storyboard":
            try:
                self._send_json(200, storyboard(url), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
            return

        if path == "/preview":
            # Fetching a small rendition so the panel can play the video before
            # the user commits to a quality or a range. An ordinary job in every
            # respect — the panel polls /job/<id> and then reads it off /file/,
            # exactly as it would a real download; only the reaper treats it
            # differently, deleting it even when downloads are being kept.
            try:
                # Cached from the /resolve the panel already did, so this is a
                # dictionary lookup rather than another minute of player API.
                duration = int(probe(url).get("duration") or 0)
            except Exception as exc:  # noqa: BLE001
                self._send_json(502, {"error": str(exc)[:500]}, origin)
                return
            if duration > PREVIEW_MAX_DURATION:
                self._send_json(413, {
                    "error": (
                        f"Too long to preview — {_clock(duration)} would mean fetching a "
                        "few hundred MB to pick two timestamps. Use the filmstrip."
                    ),
                    "maxDuration": PREVIEW_MAX_DURATION,
                }, origin)
                return
            job = self.jobs.start(url, None, False, None, preview=True)
            self._send_json(202, job.public(), origin)
            return

        if path == "/download":
            format_id = body.get("formatId")
            try:
                section = _section(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return
            try:
                max_height = _max_height(body)
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)}, origin)
                return
            job = self.jobs.start(
                url,
                str(format_id) if format_id else None,
                bool(body.get("audioOnly")),
                section,
                max_height=max_height,
            )
            self._send_json(202, job.public(), origin)
            return

        if path == "/render/start":
            try:
                width = int(body.get("width", 1920))
                height = int(body.get("height", 1080))
                fps = float(body.get("fps", 30))
                total_frames = int(body.get("totalFrames", 0))
                codec_id = str(body.get("codec", "h264_hw"))
                filename = str(body.get("filename", "export.mp4"))
                output_dir = body.get("outputDir")
                bitrate = body.get("bitrate")
                # "rawvideo" (the default) means the browser sends uncompressed
                # RGBA and this end encodes. "h264"/"hevc" means it already
                # encoded, with the same hardware, and this end only muxes.
                stream_format = str(body.get("streamFormat", "rawvideo"))

                # The mix arrives as a file the editor uploaded first. It used to
                # come inline as base64 in this very request, which meant a WAV
                # of the whole timeline — 192 KB per second of stereo, then a
                # third bigger again as base64 — had to fit in one JSON body. Ten
                # minutes of audio is over 150 MB, which no request cap is going
                # to accommodate and no JSON parser should be asked to.
                #
                # audioBase64 is still read, for a caller that has not moved over
                # and a mix small enough to carry.
                audio_path = None
                uploaded_audio = body.get("audioPath")
                audio_base64 = body.get("audioBase64")
                if uploaded_audio:
                    candidate = os.path.abspath(os.path.expanduser(str(uploaded_audio)))
                    if not os.path.isfile(candidate):
                        raise ValueError(f"Audio file not found: {candidate}")
                    audio_path = candidate
                elif audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_audio.write(audio_bytes)
                    temp_audio.close()
                    audio_path = temp_audio.name

                session = render_manager.create_session(
                    width=width,
                    height=height,
                    fps=fps,
                    total_frames=total_frames,
                    codec_id=codec_id,
                    output_filename=filename,
                    output_dir=output_dir,
                    audio_path=audio_path,
                    bitrate=bitrate,
                    stream_format=stream_format,
                )
                self._send_json(201, session.public(), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path == "/render/finish":
            session_id = str(body.get("sessionId", ""))
            session = render_manager.get(session_id)
            if not session:
                self._send_json(404, {"error": "No such render session"}, origin)
                return
            result = session.finish()
            self._send_json(200, result, origin)
            return

        if path == "/render/cancel":
            session_id = str(body.get("sessionId", ""))
            session = render_manager.get(session_id)
            if not session:
                self._send_json(404, {"error": "No such render session"}, origin)
                return
            session.cancel()
            self._send_json(200, session.public(), origin)
            return

        if path == "/timeline/render":
            try:
                width = int(body.get("width", 1920))
                height = int(body.get("height", 1080))
                fps = float(body.get("fps", 30))
                duration = float(body.get("duration", 0))
                codec_id = str(body.get("codec", "h264_hw"))
                filename = str(body.get("filename", "timeline_export.mp4"))
                output_dir = body.get("outputDir")
                bg_color = str(body.get("backgroundColor", "#000000"))
                bitrate = body.get("bitrate")

                if not output_dir:
                    output_dir = probe_hardware()["defaultExportDir"]
                target_path = os.path.join(output_dir, filename)

                raw_tracks = body.get("tracks", [])
                tracks: list[TrackSpec] = []
                for rt in raw_tracks:
                    clips: list[ClipSpec] = []
                    for rc in rt.get("clips", []):
                        clips.append(ClipSpec(
                            clip_id=str(rc.get("id", "")),
                            media_id=str(rc.get("mediaId", "")),
                            file_path=str(rc.get("filePath", "")),
                            start_time=float(rc.get("startTime", 0)),
                            duration=float(rc.get("duration", 0)),
                            source_in=float(rc.get("sourceIn", rc.get("sourceInPoint", 0))),
                            source_out=float(rc.get("sourceOut", rc.get("sourceOutPoint", 0))),
                            track_type=str(rt.get("type", "video")),
                            speed=float(rc.get("speed", 1.0)),
                            volume=float(rc.get("volume", 1.0)),
                            opacity=float(rc.get("opacity", 1.0)),
                            text=str(rc["text"]) if rc.get("text") is not None else None,
                            font_size=int(rc.get("fontSize", 54)),
                            font_color=str(rc.get("textColor", rc.get("fontColor", "#ffffff"))),
                        ))
                    tracks.append(TrackSpec(
                        track_id=str(rt.get("id", "")),
                        track_type=str(rt.get("type", "video")),
                        clips=clips,
                    ))

                spec = TimelineSpec(
                    width=width,
                    height=height,
                    fps=fps,
                    duration=duration,
                    codec_id=codec_id,
                    output_path=target_path,
                    background_color=bg_color,
                    bitrate=bitrate,
                    tracks=tracks,
                )

                job = timeline_manager.create_job(spec)
                self._send_json(201, job.public(), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path == "/timeline/cancel":
            job_id = str(body.get("jobId", ""))
            job = timeline_manager.get(job_id)
            if not job:
                self._send_json(404, {"error": "No such timeline job"}, origin)
                return
            job.cancel()
            self._send_json(200, job.public(), origin)
            return

        if path == "/proxy/plan":
            # Answers "should this file be proxied?" without transcoding anything,
            # so the editor can skip the work for footage the browser can already
            # hardware-decode instead of proxying a whole bin on import.
            try:
                source_path = str(body.get("sourcePath", ""))
                self._send_json(200, needs_proxy(source_path), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path == "/proxy/generate":
            try:
                source_path = str(body.get("sourcePath", ""))
                height = int(body.get("height", 540))
                codec_id = str(body.get("codec", "h264_hw"))
                proxy_job = proxy_manager.create_proxy(source_path, height=height, codec_id=codec_id)
                self._send_json(200, proxy_job.public(), origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)}, origin)
            return

        if path == "/ai/silence-detect":
            try:
                file_path = str(body.get("filePath", ""))
                threshold = float(body.get("noiseThresholdDb", -30.0))
                min_dur = float(body.get("minDurationSec", 0.4))
                padding = float(body.get("paddingSec", 0.08))
                result = detect_silences(file_path, threshold, min_dur, padding)
                self._send_json(200, result, origin)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"error": str(exc)}, origin)
            return

        self._send_json(404, {"error": f"No route for {path}"}, origin)

    # --- file delivery -------------------------------------------

    def _serve_handle(self, token: str, origin: str) -> None:
        """Serve a file by handle.

        The handle was minted only after the path passed containment, so there
        is nothing to re-authorise here beyond checking the file is still
        there. This is why the URL carries a handle and not a path: a path in
        a URL is an invitation to edit it.
        """
        fs = self.config.localfs
        if fs is None:
            self._send_json(403, {"error": "This bridge shares no folders"}, origin)
            return
        try:
            handle = fs.lookup(token)
        except LocalFsError as exc:
            self._send_json(404, {"error": str(exc)}, origin)
            return

        if not os.path.isfile(handle.path):
            self._send_json(404, {"error": "That file has moved or been deleted"}, origin)
            return

        size = os.path.getsize(handle.path)
        self.send_response(200)
        self.send_header("Content-Type", handle.content_type or "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", "inline")
        self._send_cors(origin)
        self.end_headers()
        with open(handle.path, "rb") as fh:
            while chunk := fh.read(FILE_CHUNK):
                self.wfile.write(chunk)

    def _serve_file(self, job_id: str, origin: str) -> None:
        path = None
        job = self.jobs.get(job_id)
        if job and job.state == "ready" and job.filepath:
            path = job.filepath
        else:
            pjob = proxy_manager.get(job_id)
            if pjob and pjob.state == "ready" and pjob.target_path:
                path = pjob.target_path

        if not path or not os.path.exists(path):
            self._send_json(404, {"error": "That file is not ready or does not exist"}, origin)
            return

        # Security check: must sit inside work_dir or proxy cache_dir.
        #
        # commonpath, not startswith, and realpath, not abspath. startswith
        # treats "/tmp/work-elsewhere" as inside "/tmp/work", and abspath
        # leaves a symlink pointing out of the directory intact.
        abs_path = os.path.realpath(path)
        valid_dir = any(
            contains(os.path.realpath(root), abs_path)
            for root in (self.jobs.work_dir, proxy_manager.cache_dir)
        )
        # Proxies are written beside the footage they came from, so a directory
        # test cannot authorise them any more. `is_known_target` authorises the
        # exact files this process produced — one path per job, never a folder,
        # so a media directory does not become browsable.
        if not valid_dir and proxy_manager.is_known_target(abs_path):
            valid_dir = True
        if not valid_dir:
            self._send_json(403, {"error": "Refusing to serve a file outside approved directories"}, origin)
            return

        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "video/mp4"

        # Range support, because the consumer is a <video> element. Without a 206
        # the browser has to pull the whole file before it can show frame one and
        # cannot seek inside it at all — which is the entire point of a proxy.
        start, end = 0, size - 1
        status = 200
        rng = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)\s*$", rng) if rng else None
        if match:
            first, last = match.group(1), match.group(2)
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            elif last:
                # "bytes=-500" means the LAST 500 bytes, not the first 500.
                start = max(0, size - int(last))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self._send_cors(origin)
                self.end_headers()
                return
            end = min(end, size - 1)
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Disposition", "inline")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self._send_cors(origin)
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(FILE_CHUNK, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # A <video> abandons ranges constantly while seeking. Normal.
                    return
                remaining -= len(chunk)


def make_server(config: BridgeConfig, jobs: JobStore) -> ThreadingHTTPServer:
    handler = type("BoundBridgeHandler", (BridgeHandler,), {"config": config, "jobs": jobs})
    httpd = ThreadingHTTPServer(("127.0.0.1", config.port), handler)
    httpd.daemon_threads = True
    return httpd
