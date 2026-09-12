"""Bounded local-filesystem access for the editor.

Two capabilities the browser cannot have and the editor needs:

  IMPORT — a file on this machine, brought into the bin without a round trip
    through a CDN. The tab cannot read a path; this can.

  FRAMES — preview frames written to disk instead of returned as base64. A
    frame is ~200KB of JPEG and every one of them used to travel through the
    transcript and through Supabase's cache. Writing it here and handing back
    a path costs a filename.

Both are dangerous in the obvious way, so neither takes a path from the caller
and does what it is told:

  * READS resolve a *basename* against an allowlist of roots. The caller never
    names a directory, so there is no path for it to traverse out of. A file
    outside the roots is not "forbidden", it is invisible.

  * Containment is checked with realpath + commonpath, never startswith.
    `startswith` says /home/me/work contains /home/me/workspace-elsewhere, and
    it follows a symlink straight out of the allowlist without noticing.

  * Only media extensions resolve. An agent talked into asking for
    `id_rsa`, `.env` or `credentials` gets nothing, because nothing without a
    media extension is ever a candidate — and the private files worth stealing
    do not have one.

  * WRITES never take a path at all. The caller supplies bytes and a label;
    the name is generated here, inside one directory.

  * Serving is by opaque handle, not by path. The path is verified once, when
    the handle is minted, and the URL the browser holds cannot be edited into
    a different file.

The transport in front of all of this (loopback bind, per-run bearer token,
Origin allowlist, Host check) lives in server.py and applies to these routes
exactly as it applies to every other.
"""

from __future__ import annotations

import base64
import os
import secrets
import threading
import time
from dataclasses import dataclass

# Deliberately the same set the media index already walks: this widens what can
# be *fetched*, never what can be *found*. Note what is absent — no .env, no
# .pem, no .json, no .txt. A secret does not have one of these extensions.
READABLE_EXTENSIONS = frozenset({
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm",
    ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".tiff",
})

# A handle is single-file and short-lived. The window only has to cover the
# browser's own fetch, which follows immediately.
HANDLE_TTL_SECONDS = 15 * 60
MAX_HANDLES = 256

# One frame of 1080p JPEG is well under this; the cap is here so a bug upstream
# cannot fill the disk one POST at a time.
MAX_FRAME_BYTES = 12 * 1024 * 1024

_FRAME_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class LocalFsError(Exception):
    """Refused. The message is safe to show the caller."""


@dataclass(frozen=True)
class Handle:
    path: str
    size: int
    content_type: str
    created_at: float


def _real(path: str) -> str:
    """Absolute, symlinks resolved. Every comparison below happens on this form."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def contains(root: str, candidate: str) -> bool:
    """True when `candidate` is inside `root`, both fully resolved.

    commonpath compares whole path segments, which is the difference that
    matters: startswith("/home/me/work") is also true of
    "/home/me/workspace-elsewhere", and that is a real directory someone can
    create. realpath is applied by the caller before this, so a symlink
    pointing out of the root fails here rather than being followed.
    """
    try:
        return os.path.commonpath([root, candidate]) == root
    except ValueError:
        # Different drives on Windows: not the same tree by definition.
        return False


class LocalFs:
    """Allowlisted reads, one output directory for writes, handles for serving."""

    def __init__(self, roots: list[str], frames_dir: str) -> None:
        seen: list[str] = []
        for root in roots:
            resolved = _real(root)
            if os.path.isdir(resolved) and resolved not in seen:
                seen.append(resolved)
        self.roots = seen

        self.frames_dir = _real(frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)

        self._handles: dict[str, Handle] = {}
        self._lock = threading.Lock()

    # --- reading -------------------------------------------------

    def resolve(self, filename: str) -> str:
        """Find a file by name inside the allowed roots.

        Takes a basename, never a path. `os.path.basename` is applied rather
        than trusted: it turns "../../.ssh/id_rsa" into "id_rsa", which then
        fails the extension check, and would fail the search anyway.
        """
        name = os.path.basename(str(filename or "").strip())
        if not name:
            raise LocalFsError("No filename given.")

        ext = os.path.splitext(name)[1].lower()
        if ext not in READABLE_EXTENSIONS:
            raise LocalFsError(
                f'"{name}" is not a media file. This bridge reads video, audio and images '
                "and nothing else, whatever the path."
            )

        if not self.roots:
            raise LocalFsError(
                "No folders are shared with the editor. Restart the bridge with "
                "--allow-dir <folder> for each folder it may read."
            )

        for root in self.roots:
            for dirpath, dirnames, filenames in os.walk(root):
                # Dot-directories hold the things worth stealing and none of the
                # things worth importing.
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                if name in filenames:
                    found = _real(os.path.join(dirpath, name))
                    # The walk started inside a root, but a symlinked entry can
                    # still land outside it, so the result is checked and not
                    # assumed.
                    if contains(root, found) and os.path.isfile(found):
                        return found

        raise LocalFsError(
            f'"{name}" was not found in the folders shared with the editor '
            f"({', '.join(self.roots)})."
        )

    # --- handles -------------------------------------------------

    def mint(self, path: str, content_type: str = "") -> str:
        """Issue an opaque handle for an already-verified path."""
        resolved = _real(path)
        if not any(contains(root, resolved) for root in (*self.roots, self.frames_dir)):
            raise LocalFsError("That file is outside the folders shared with the editor.")
        if not os.path.isfile(resolved):
            raise LocalFsError("That file no longer exists.")

        token = secrets.token_urlsafe(18)
        handle = Handle(
            path=resolved,
            size=os.path.getsize(resolved),
            content_type=content_type,
            created_at=time.time(),
        )
        with self._lock:
            self._prune_locked()
            self._handles[token] = handle
        return token

    def lookup(self, token: str) -> Handle:
        with self._lock:
            self._prune_locked()
            handle = self._handles.get(token)
        if handle is None:
            raise LocalFsError("That link has expired. Ask for the file again.")
        return handle

    def _prune_locked(self) -> None:
        cutoff = time.time() - HANDLE_TTL_SECONDS
        for token in [t for t, h in self._handles.items() if h.created_at < cutoff]:
            del self._handles[token]
        while len(self._handles) > MAX_HANDLES:
            oldest = min(self._handles, key=lambda t: self._handles[t].created_at)
            del self._handles[oldest]

    # --- writing -------------------------------------------------

    def write_frame(self, data_b64: str, content_type: str, label: str = "") -> Handle:
        """Write one image into the frames directory. The caller names nothing.

        `label` only reaches the filename through a character filter, and only
        as a middle segment: the directory is fixed, the extension comes from
        the content type, and a random segment keeps two frames from colliding.
        There is no input here that can steer the write somewhere else.
        """
        suffix = _FRAME_SUFFIXES.get(content_type)
        if suffix is None:
            raise LocalFsError(
                f'"{content_type}" is not an image type this writes. '
                f"Use one of: {', '.join(sorted(_FRAME_SUFFIXES))}."
            )

        try:
            raw = base64.b64decode(data_b64, validate=True)
        except Exception as exc:
            raise LocalFsError(f"That is not valid base64 ({exc}).") from exc

        if not raw:
            raise LocalFsError("No image data.")
        if len(raw) > MAX_FRAME_BYTES:
            raise LocalFsError(
                f"That frame is {len(raw) // (1024 * 1024)} MB, over the "
                f"{MAX_FRAME_BYTES // (1024 * 1024)} MB limit."
            )

        safe = "".join(c for c in str(label)[:40] if c.isalnum() or c in "-_") or "frame"
        name = f"{safe}-{secrets.token_hex(4)}{suffix}"
        target = os.path.join(self.frames_dir, name)

        # Belt and braces: the name is generated, but the check is cheap and
        # this is the one place bytes hit the disk.
        if not contains(self.frames_dir, _real(target)):
            raise LocalFsError("Refusing to write outside the frames directory.")

        with open(target, "wb") as fh:
            fh.write(raw)

        return Handle(
            path=target,
            size=len(raw),
            content_type=content_type,
            created_at=time.time(),
        )
