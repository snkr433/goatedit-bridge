"""Download jobs: yt-dlp runs here, on a worker thread, one job per request."""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import yt_dlp

# The plugin polls /job/<id>, so a job id ends up in a URL path. Only hex.
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Finished files are kept so the editor can pull them, then reaped. Two hours
# is long enough for a user who wandered off mid-download.
JOB_TTL_SECONDS = 2 * 60 * 60

# Extracting a URL costs ~1.8s of player-API and JS-challenge work, and the
# panel asks for the same URL twice: once to fill the quality dropdown, then
# again the moment you press download. The second one is the same answer to the
# same question, so /resolve leaves its result here for /download to pick up.
# Short-lived on purpose — the media URLs inside carry their own expiry, and a
# stale hit only costs us the extraction we were trying to skip.
_INFO_TTL_SECONDS = 10 * 60
_INFO_CACHE_MAX = 32
_info_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_info_lock = threading.Lock()


def _cache_info(url: str, info: dict[str, Any]) -> None:
    now = time.time()
    with _info_lock:
        for key, (stamp, _) in list(_info_cache.items()):
            if now - stamp > _INFO_TTL_SECONDS:
                _info_cache.pop(key, None)
        if len(_info_cache) >= _INFO_CACHE_MAX:
            oldest = min(_info_cache, key=lambda k: _info_cache[k][0])
            _info_cache.pop(oldest, None)
        _info_cache[url] = (now, info)


def _cached_info(url: str) -> dict[str, Any] | None:
    with _info_lock:
        hit = _info_cache.get(url)
        if not hit:
            return None
        stamp, info = hit
        if time.time() - stamp > _INFO_TTL_SECONDS:
            _info_cache.pop(url, None)
            return None
    # Processing mutates the dict it is handed, so the cache keeps the original
    # and every job works on its own copy.
    return copy.deepcopy(info)


_EXT_TO_TYPE = {
    "mp4": "video", "mkv": "video", "webm": "video", "mov": "video",
    "m4a": "audio", "mp3": "audio", "opus": "audio", "wav": "audio", "flac": "audio",
}


@dataclass
class Job:
    id: str
    url: str
    state: str = "queued"          # queued | downloading | processing | ready | error
    progress: float = 0.0          # 0..100
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None
    filepath: str | None = None
    media: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # A preview is scratch: it is a small rendition fetched so the panel can
    # play the video before deciding what to keep, so it never gets a friendly
    # name and it never survives the reaper, even in the user's own folder.
    ephemeral: bool = False
    # What the preview arrived as, before _make_playable had its say. Kept so
    # that a preview which still will not play can be reported as the codec it
    # is, rather than as an unexplained "Format error".
    source_codecs: str = ""

    def public(self) -> dict[str, Any]:
        return {
            "jobId": self.id,
            "state": self.state,
            "progress": round(self.progress, 1),
            "downloadedBytes": self.downloaded_bytes,
            "totalBytes": self.total_bytes,
            "error": self.error,
            "media": self.media or None,
            # Where the file actually landed. Only the paired origin, holding the
            # token, ever sees this — and it names a file on the user's own disk
            # that they just asked us to write. Without it the panel can only say
            # "downloaded" and leave them hunting through a temp directory.
            "filepath": self.filepath,
            "sourceCodecs": self.source_codecs or None,
        }


def ffmpeg_available() -> bool:
    """yt-dlp needs ffmpeg to mux the separate 1080p+ video and audio streams."""
    return shutil.which("ffmpeg") is not None


def _base_opts() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # No player_client override on purpose. YouTube retires clients faster
        # than we can re-pick one: the pinned android_vr/android/tv/ios list
        # still probed fine but every media fetch came back 403, because those
        # clients now need a GVS PO token we do not have. yt-dlp tracks which
        # client works this week; the same reasoning that keeps the yt-dlp
        # dependency unpinned in pyproject.toml keeps this unset.
    }


def probe(url: str) -> dict[str, Any]:
    """Metadata + selectable formats for one URL. No bytes are fetched."""
    opts = _base_opts() | {"skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    _cache_info(url, info)

    def rank(f: dict[str, Any]) -> tuple[int, float]:
        # YouTube offers most heights twice: once as a single progressive or
        # DASH file, once as an HLS ladder. They look alike here but do not
        # behave alike — HLS carries no filesize, so the panel loses its size
        # label, and it arrives as hundreds of fragments, which turns a two
        # second download into a minute. Take the non-HLS twin whenever there
        # is one, and only then break ties on bitrate.
        is_hls = (f.get("protocol") or "").startswith("m3u8")
        return (0 if is_hls else 1, f.get("tbr") or 0)

    heights: dict[int, dict[str, Any]] = {}
    for f in info.get("formats") or []:
        height = f.get("height")
        if not height or f.get("vcodec") in (None, "none"):
            continue
        best = heights.get(height)
        if not best or rank(f) > rank(best):
            heights[height] = f

    formats = [
        {
            "formatId": f["format_id"],
            "label": f"{h}p60" if (f.get("fps") or 0) > 45 else f"{h}p",
            "height": h,
            "fps": int(f.get("fps") or 0),
            "ext": f.get("ext") or "mp4",
            "hasAudio": f.get("acodec") not in (None, "none"),
            "sizeBytes": int(f.get("filesize") or f.get("filesize_approx") or 0),
        }
        for h, f in sorted(heights.items(), key=lambda kv: kv[0], reverse=True)
    ]

    # Plenty of sites publish exactly one rendition and do not tag it with a
    # height — Instagram and Pinterest usually do not. That left the quality
    # dropdown empty, which reads as "nothing to download" even though asking
    # for no format at all gets yt-dlp's best pick, which is the only pick.
    if not formats and (info.get("formats") or info.get("url")):
        formats = [{
            "formatId": "",
            "label": "Best available",
            "height": int(info.get("height") or 0),
            "fps": int(info.get("fps") or 0),
            "ext": info.get("ext") or "mp4",
            "hasAudio": True,
            "sizeBytes": int(info.get("filesize") or info.get("filesize_approx") or 0),
        }]

    thumbnails = info.get("thumbnails") or []
    return {
        "title": info.get("title") or "Video",
        "duration": int(info.get("duration") or 0),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "thumbnail": thumbnails[-1]["url"] if thumbnails else (info.get("thumbnail") or ""),
        "formats": formats,
        "audioOnly": True,
        "ffmpeg": ffmpeg_available(),
    }


# A filmstrip has to fit through the panel's mediated fetch as base64 text, so
# the tier we pick is a byte budget as much as a picture-quality one. sb0 is
# sharp but costs 1.3 MB and 36 requests on a long video; sb1 covers the same
# video in 493 KB and 13. Anything past this budget gets a coarser tier rather
# than a slow panel.
_STORYBOARD_TIERS = ("sb1", "sb0", "sb2", "sb3")
_STORYBOARD_MAX_BYTES = 600 * 1024
_STORYBOARD_MAX_FRAGMENTS = 20


def storyboard(url: str) -> dict[str, Any]:
    """A scrub filmstrip for `url`, as sprite sheets the panel can slice in CSS.

    The sheets are sent whole rather than cut into individual thumbnails: it
    keeps a hard image dependency out of the bridge, and 13 sheets travel a lot
    lighter than the 325 tiles printed on them.
    """
    info = _cached_info(url)
    if info is None:
        opts = _base_opts() | {"skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        _cache_info(url, info)

    duration = float(info.get("duration") or 0)
    by_id = {f.get("format_id"): f for f in (info.get("formats") or [])}
    for tier in _STORYBOARD_TIERS:
        board = by_id.get(tier)
        if not board:
            continue
        fragments = board.get("fragments") or []
        if not fragments or len(fragments) > _STORYBOARD_MAX_FRAGMENTS:
            continue

        # One request per sheet, and a long video has a dozen of them; serially
        # that was 7.8s of staring at an empty strip.
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            sheets = list(pool.map(lambda f: _fetch_sheet(f["url"]), fragments))
        if any(sheet is None for sheet in sheets):
            continue
        if sum(len(sheet) for sheet in sheets) > _STORYBOARD_MAX_BYTES:
            continue

        columns = int(board.get("columns") or 0)
        rows = int(board.get("rows") or 0)
        if not columns or not rows:
            continue
        return {
            "tier": tier,
            "duration": duration,
            "tileWidth": int(board.get("width") or 0),
            "tileHeight": int(board.get("height") or 0),
            "columns": columns,
            "rows": rows,
            "count": len(fragments) * columns * rows,
            "sheets": [
                {
                    "start": float(sum(float(f.get("duration") or 0) for f in fragments[:i])),
                    "duration": float(fragments[i].get("duration") or 0),
                    "dataUri": "data:image/jpeg;base64," + base64.b64encode(sheet).decode(),
                }
                for i, sheet in enumerate(sheets)
            ],
        }

    # Plenty of sites publish no storyboard at all. That is not an error — the
    # panel falls back to its numeric in/out fields, which are what actually
    # decide the cut anyway.
    return {"duration": duration, "sheets": [], "count": 0}


def _fetch_sheet(url: str) -> bytes | None:
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            return ydl.urlopen(url).read()
    except Exception:  # noqa: BLE001 — a missing sheet just means no filmstrip
        return None


def _clock(seconds: float) -> str:
    """A timestamp that survives being part of a filename — no colons."""
    whole = int(seconds)
    return f"{whole // 3600:02d}h{whole % 3600 // 60:02d}m{whole % 60:02d}s" if whole >= 3600 \
        else f"{whole // 60:02d}m{whole % 60:02d}s"


# What the preview player plays. Small on purpose — the point is to see the
# shot, not to grade it — so the ladder starts at 480p and only widens when a
# site has nothing that low.
#
# The obvious pick would be an already-muxed rendition, and on YouTube that is
# format 18. It is also the one format YouTube now answers with 403 unless you
# carry a PO token, so preferring it would make the preview fail exactly where
# it is most wanted. Take video and audio separately and let ffmpeg put them
# together, the same way an ordinary download already does, and keep the muxed
# form only as the fallback for sites that offer nothing else.
# avc1 ahead of everything, and not for quality: YouTube's 480p is usually AV1,
# which Safari and anything but a recent Chrome cannot decode — a preview that
# hands back a black frame is worse than no preview. H.264 at this size is
# universal and costs a few hundred KB more.
PREVIEW_FORMAT = (
    "bv*[height<=480][vcodec^=avc1]+ba[ext=m4a]"
    "/bv*[height<=480][ext=mp4]+ba[ext=m4a]"
    "/bv*[height<=480]+ba"
    "/bv*[vcodec^=avc1]+ba[ext=m4a]"
    "/bv*+ba"
    "/b[height<=480]"
    "/b"
)

# Above this, fetching a whole low-quality copy costs more than the scrubbing it
# buys: 360p runs a couple of MB a minute, so an hour-long source is a few
# hundred MB fetched to pick two timestamps off a filmstrip that already works.
# The panel is told the number so it can say why rather than just not appear.
PREVIEW_MAX_DURATION = 20 * 60

# What every browser can actually decode. The format ladder above asks for
# H.264 first, but a site that has none leaves us holding VP9, AV1 or Opus, and
# those play in some browsers and raise a bare "Format error" in the rest —
# which the panel cannot tell apart from a blocked fetch. So rather than trust
# the ladder, look at the file that arrived and fix the ones that would not
# play. Only the preview does this; a real download is the user's file and gets
# handed over exactly as the site served it.
PLAYABLE_VCODECS = ("h264",)
PLAYABLE_ACODECS = ("aac", "mp3")


def _ffprobe_codecs(path: str) -> tuple[str, str]:
    """(video codec, audio codec) of a local file; empty when ffprobe cannot say."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "", ""
    codecs = {"video": "", "audio": ""}
    for line in out.splitlines():
        name, _, kind = line.partition(",")
        if kind in codecs and not codecs[kind]:
            codecs[kind] = name
    return codecs["video"], codecs["audio"]


def _make_playable(path: str) -> tuple[str, str]:
    """A browser-playable copy of a preview, plus what the original was.

    Returns the path unchanged when it already holds H.264 in MP4. Otherwise
    remuxes if only the container is wrong, and re-encodes if a codec is. The
    note comes back either way so a failure downstream can still say what the
    file was, instead of leaving the panel to guess.
    """
    vcodec, acodec = _ffprobe_codecs(path)
    note = f"{vcodec or '?'}/{acodec or '?'}"
    if not vcodec or not ffmpeg_available():
        # Nothing to decide with, or nothing to decide it with. Leave it alone:
        # an unplayable preview is a worse outcome than none, but a mangled one
        # is worse still.
        return path, note

    is_mp4 = os.path.splitext(path)[1].lower() == ".mp4"
    video_ok = vcodec in PLAYABLE_VCODECS
    # No audio at all is fine — plenty of sources have none, and a silent
    # preview still shows the shot.
    audio_ok = not acodec or acodec in PLAYABLE_ACODECS
    if is_mp4 and video_ok and audio_ok:
        return path, note

    out = os.path.splitext(path)[0] + ".play.mp4"
    args = ["ffmpeg", "-y", "-v", "error", "-i", path]
    args += ["-c:v", "copy"] if video_ok else [
        # Scrubbing quality, not grading quality. veryfast keeps a 20 minute
        # source — the longest a preview is allowed to be — inside a couple of
        # minutes, and 480p is all the strip ever shows.
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-vf", "scale=-2:min(480\,ih)",
    ]
    if not acodec:
        args += ["-an"]
    else:
        args += ["-c:a", "copy"] if audio_ok else ["-c:a", "aac", "-b:a", "128k"]
    # Without this the moov atom lands at the end of the file and the element
    # has to range-request its way backwards before it can show a single frame.
    args += ["-movflags", "+faststart", out]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=900, check=False)
    except (OSError, subprocess.SubprocessError):
        return path, note
    if done.returncode != 0 or not os.path.exists(out) or not os.path.getsize(out):
        # A half-written re-encode is not scratch anyone will come back for, and
        # the reaper only knows about the path the job is holding.
        try:
            os.remove(out)
        except OSError:
            pass
        return path, note
    # The source was scratch either way; only one of the two is worth keeping.
    try:
        os.remove(path)
    except OSError:
        pass
    return out, note


def _format_selector(format_id: str | None, audio_only: bool) -> str:
    if audio_only:
        return "ba[ext=m4a]/ba/b"
    if format_id:
        # Pair the chosen video-only rendition with the best m4a; yt-dlp falls
        # back to the bare format when it already carries audio.
        return f"{format_id}+ba[ext=m4a]/{format_id}"
    if ffmpeg_available():
        return "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b"
    # No ffmpeg means no muxing, so only already-combined streams will play.
    return "b[ext=mp4]/b"


class JobStore:
    """Every job this bridge has run, plus the directory their files live in.

    `keep_files` is on when the user pointed `--dir` at a folder of their own:
    the files are theirs now, named after the video, and neither the reaper nor
    shutdown may touch them. Without it the dir is a temp one we made and wipe.
    """

    def __init__(self, work_dir: str, keep_files: bool = False) -> None:
        self.work_dir = work_dir
        self.keep_files = keep_files
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._name_lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        if not JOB_ID_RE.match(job_id):
            return None
        with self._lock:
            return self._jobs.get(job_id)

    def start(
        self, url: str, format_id: str | None, audio_only: bool,
        section: tuple[float, float] | None = None, preview: bool = False,
    ) -> Job:
        self._reap()
        job = Job(id=uuid.uuid4().hex, url=url, ephemeral=preview)
        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(
            target=self._run, args=(job, format_id, audio_only, section, preview), daemon=True,
        ).start()
        return job

    def _reap(self) -> None:
        """Drops jobs past their TTL and deletes the files they were holding."""
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            stale = [j for j in self._jobs.values() if j.created_at < cutoff]
            for job in stale:
                self._jobs.pop(job.id, None)
        if self.keep_files:
            # The job record expires so /file/<id> stops answering, but the file
            # itself is in a folder the user chose. Deleting it would be theft.
            # Previews are the exception: the user never asked for that file and
            # would only find it as litter beside the ones they did ask for.
            stale = [j for j in stale if j.ephemeral]
        for job in stale:
            if job.filepath and os.path.exists(job.filepath):
                try:
                    os.remove(job.filepath)
                except OSError:
                    pass

    def _rehome(
        self, path: str, info: dict[str, Any],
        section: tuple[float, float] | None, audio_only: bool,
    ) -> str:
        """Moves a finished download to a name the user will recognise.

        The name has to say what makes this file different from the others, or
        clipping three ranges out of one video just overwrites one file three
        times. Quality and the in/out points are exactly that difference, so
        they go in the name.
        """
        title = info.get("title") or "video"
        video_id = info.get("id") or ""
        parts = [f"{title} [{video_id}]" if video_id else title]
        if audio_only:
            parts.append("audio")
        else:
            height = info.get("height") or (info.get("requested_downloads") or [{}])[0].get("height")
            if height:
                parts.append(f"{int(height)}p")
        if section:
            parts.append(f"{_clock(section[0])}-{_clock(section[1])}")

        ext = os.path.splitext(path)[1]
        head = yt_dlp.utils.sanitize_filename(parts[0], restricted=False)
        tail = yt_dlp.utils.sanitize_filename(" ".join(parts[1:]), restricted=False)
        # Same path budget as trim_file_name. What gets cut matters: the tail is
        # the quality and the timecodes, the only thing telling two clips of one
        # video apart, so the title gives up its characters first. Trimming the
        # other way round turned 00m10s-00m20s and 00m15s-00m25s into the same
        # "00m1" and put the collision straight back.
        room = max(24, 120 - len(ext) - len(tail) - 1)
        stem = (head[:room] + " " + tail).strip() if tail else head[:room]

        # Two jobs can want one name at the same moment, so picking it and
        # claiming it happen together rather than one after the other.
        with self._name_lock:
            target = os.path.join(self.work_dir, stem + ext)
            n = 2
            while os.path.exists(target):
                target = os.path.join(self.work_dir, f"{stem} ({n}){ext}")
                n += 1
            try:
                os.replace(path, target)
            except OSError:
                # A rename that fails leaves a perfectly good file where it is;
                # losing the download over its name would be the worse outcome.
                return path
        return target

    @staticmethod
    def _extract(ydl: yt_dlp.YoutubeDL, url: str) -> dict[str, Any]:
        """The formats for `url`, downloaded — reusing /resolve's work if it is
        still around. Format selection happens here rather than at extraction,
        so the cached dict serves any quality the user ends up picking.

        A cached dict can go stale in ways that are hard to predict per site, so
        any failure re-extracts from scratch instead of surfacing as a download
        error the user cannot act on.
        """
        cached = _cached_info(url)
        if cached:
            try:
                return ydl.process_video_result(cached, download=True)
            except Exception:  # noqa: BLE001 — a stale hit is not worth failing over
                pass
        return ydl.extract_info(url, download=True)

    def _run(
        self, job: Job, format_id: str | None, audio_only: bool,
        section: tuple[float, float] | None = None, preview: bool = False,
    ) -> None:
        def hook(d: dict[str, Any]) -> None:
            if d.get("status") == "downloading":
                job.state = "downloading"
                job.downloaded_bytes = int(d.get("downloaded_bytes") or 0)
                job.total_bytes = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
                if job.total_bytes:
                    job.progress = min(99.0, job.downloaded_bytes / job.total_bytes * 100)
            elif d.get("status") == "finished":
                # Merging/remuxing happens after the last stream lands.
                job.state = "processing"
                job.progress = 99.0

        # Always download to the job id, even in the user's own folder. Naming
        # by title used to mean two jobs on one video wrote the same path and
        # silently stomped each other — which in-and-out points turned from an
        # edge case into the normal one, since every clip of a video shares its
        # title. The friendly name is put on afterwards, where a collision can
        # be settled instead of raced.
        name_tmpl = f"{job.id}.%(ext)s"
        opts = _base_opts() | {
            "format": PREVIEW_FORMAT if preview else _format_selector(format_id, audio_only),
            "outtmpl": os.path.join(self.work_dir, name_tmpl),
            # yt-dlp measures trim_file_name against the whole path, not the
            # basename its docstring names, so a --dir a few folders deep spends
            # the budget on the directory and leaves a stub: a 117-character
            # folder turned a 92-character title into "Nai.mp4". Budget from the
            # end of work_dir instead, and keep enough room that the longest
            # title still survives on a shallow folder.
            "trim_file_name": len(os.path.join(self.work_dir, "")) + 120,
            "progress_hooks": [hook],
            # A job is one file; a URL that expands to a playlist is user error.
            "playlist_items": "1",
            # Fragmented formats otherwise arrive one piece at a time. Eight is
            # where the gain flattened out in testing; sixteen was no better.
            "concurrent_fragment_downloads": 8,
        }
        if not audio_only and ffmpeg_available():
            opts["merge_output_format"] = "mp4"
        if section:
            # ffmpeg range-seeks the remote stream, so a segment costs about the
            # same wall time whatever the source length — ~15s either way in
            # testing. That is a loss on a short video and an enormous win on a
            # long one, which is why the panel only offers it as a choice.
            start_at, end_at = section
            opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(start_at, end_at)])
            # Without this the cut lands on the nearest keyframe, which can be
            # seconds adrift of the in-point the user actually dragged to.
            opts["force_keyframes_at_cuts"] = True

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = self._extract(ydl, job.url)
            requested = (info.get("requested_downloads") or [{}])[0]
            path = requested.get("filepath") or ydl.prepare_filename(info)
            if not path or not os.path.exists(path):
                raise RuntimeError("yt-dlp finished but produced no file")

            if preview:
                path, job.source_codecs = _make_playable(path)
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            if self.keep_files and not preview:
                path = self._rehome(path, info, section, audio_only)
            job.filepath = path

            # Sites that hand back one pre-muxed file — Instagram and Pinterest
            # among them — leave the top-level info dict without width, height,
            # fps or duration; those only ever appear on the format that was
            # actually downloaded. Read through to it before giving up on a
            # field, and report 0 rather than a guess when nobody knows: the
            # editor probes the file itself and 0 is how it is told to.
            def dimension(key: str) -> float:
                for source in (info, requested):
                    value = source.get(key)
                    if value:
                        return float(value)
                return 0.0

            job.media = {
                "name": info.get("title") or os.path.basename(path),
                "type": "audio" if audio_only else _EXT_TO_TYPE.get(ext, "video"),
                "ext": ext,
                "sizeBytes": os.path.getsize(path),
                # info carries the whole video's duration even when only a
                # section of it was written, which would tell the editor a 15
                # second clip is three minutes long. The section length is the
                # one fact we know better than the info dict here.
                "duration": (section[1] - section[0]) if section else dimension("duration"),
                "width": int(dimension("width")),
                "height": int(dimension("height")),
                "fps": dimension("fps"),
                "filepath": path,
            }
            job.progress = 100.0
            job.state = "ready"
        except Exception as exc:  # noqa: BLE001 — the panel shows whatever went wrong
            job.error = str(exc)[:500]
            job.state = "error"
