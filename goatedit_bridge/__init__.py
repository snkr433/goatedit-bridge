"""GoatEdit local yt-dlp bridge.

A tiny loopback HTTP server the GoatEdit yt-dlp plugin talks to. It runs
yt-dlp on the user's own machine — their IP, their cookies — and hands the
finished file to the editor tab, which imports it straight into the media bin.

Nothing here is reachable from the internet: the socket binds 127.0.0.1, every
request must carry the session token printed at startup, and the Origin header
must match the editor the user paired with.
"""

from importlib.metadata import PackageNotFoundError, version as _installed_version

# Read from the installed distribution rather than repeating the number here.
# The two spellings drifted once already: 0.1.1 shipped /storyboard and in/out
# points while this file still said 0.1.0, so /health under-reported and the
# panel hid the trim strip from a bridge that supported it.
try:
    __version__ = _installed_version("goatedit-bridge")
except PackageNotFoundError:  # running from a source checkout, not installed
    __version__ = "0.0.0+source"
