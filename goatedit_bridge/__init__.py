"""GoatEdit local yt-dlp bridge.

A tiny loopback HTTP server the GoatEdit yt-dlp plugin talks to. It runs
yt-dlp on the user's own machine — their IP, their cookies — and hands the
finished file to the editor tab, which imports it straight into the media bin.

Nothing here is reachable from the internet: the socket binds 127.0.0.1, every
request must carry the session token printed at startup, and the Origin header
must match the editor the user paired with.
"""

__version__ = "0.1.0"
