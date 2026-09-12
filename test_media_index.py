"""Tests for /media/locate's filename index.

The matching rules here have broken twice: once when a rewrite dropped the
upload-hash suffix, and once when project-directory priority was inverted by
last-writer-wins. Both are cheap to assert directly against the index.
"""

from __future__ import annotations

import os
import tempfile

from goatedit_bridge.server import _FileIndex, _index_directory


def _index_of(*names: str, root: str = "/proj") -> _FileIndex:
    idx = _FileIndex()
    for name in names:
        idx.add(name, os.path.join(root, name))
    return idx


def _resolve(idx: _FileIndex, query: str) -> str | None:
    return idx.lookup(query) or idx.match_stem(query)


def test_exact_and_tolerant_lookups():
    idx = _index_of(
        "interview_final.mov",
        "b-roll-city.mp4",
        "4eaeed8aa5b69778_uploaded_test.mp4",
        "20240612_103500_take1.mov",
    )
    assert _resolve(idx, "interview_final.mov") == "/proj/interview_final.mov"
    assert _resolve(idx, "INTERVIEW_FINAL.MOV") == "/proj/interview_final.mov"
    # Hyphens the editor dropped somewhere between project and timeline.
    assert _resolve(idx, "brollcity.mp4") == "/proj/b-roll-city.mp4"
    # An upload keeps its content hash on disk but not in the timeline.
    assert _resolve(idx, "uploaded_test.mp4") == "/proj/4eaeed8aa5b69778_uploaded_test.mp4"
    # A camera timestamp prefix is enough to identify the take.
    assert _resolve(idx, "20240612_103500_take1.mov") == "/proj/20240612_103500_take1.mov"
    print("✓ exact, hyphen, upload-hash and timestamp lookups resolve.")


def test_label_prefix_and_longer_name_match():
    idx = _index_of("quiz.html", "whoosh-sfx-01.mp3")
    # The timeline labels generated clips by kind; the file on disk does not.
    assert _resolve(idx, "HTML-quiz.html") == "/proj/quiz.html"
    # A sound effect filed under a longer name than the timeline remembers.
    assert _resolve(idx, "whoosh.mp3") == "/proj/whoosh-sfx-01.mp3"
    print("✓ label-prefixed and longer-on-disk names resolve without hardcoded names.")


def test_no_false_positives():
    idx = _index_of("q.html", "clip.mp4")
    # A one-letter stem must not swallow every query that contains it.
    assert _resolve(idx, "HTML-quiz.html") is None
    assert _resolve(idx, "zzz.mp4") is None
    # An audio request never resolves onto a video file.
    assert _resolve(idx, "clip.mp3") is None
    print("✓ short stems, unrelated names and cross-class matches stay unresolved.")


def test_first_indexed_directory_wins():
    idx = _FileIndex()
    idx.add("clip.mov", "/project/clip.mov")
    idx.add("clip.mov", "/Downloads/clip.mov")
    # Project dirs are walked first, so they must not lose to the generic scan.
    assert idx.lookup("clip.mov") == "/project/clip.mov"
    print("✓ the directory indexed first wins a filename collision.")


def test_exact_name_beats_another_files_alias():
    idx = _FileIndex()
    # "hash_take.mov" registers "take.mov" as a tolerant alias...
    idx.add("hash_take.mov", "/uploads/hash_take.mov")
    # ...but a real file of that name must still answer for itself.
    idx.add("take.mov", "/project/take.mov")
    assert idx.lookup("take.mov") == "/project/take.mov"
    print("✓ a real filename outranks another file's alias.")


def test_index_directory_walks_and_skips():
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "footage"))
        os.makedirs(os.path.join(root, "node_modules"))
        os.makedirs(os.path.join(root, ".hidden"))
        for rel in ("footage/a.mov", "node_modules/b.mov", ".hidden/c.mov", "notes.txt"):
            with open(os.path.join(root, rel), "wb") as fh:
                fh.write(b"x")

        idx = _FileIndex()
        _index_directory(root, idx, max_depth=2)

        assert idx.lookup("a.mov") == os.path.join(root, "footage", "a.mov")
        assert idx.lookup("b.mov") is None, "node_modules must not be walked"
        assert idx.lookup("c.mov") is None, "dot-directories must not be walked"
        assert idx.lookup("notes.txt") is None, "non-media files are not indexed"
    print("✓ directory walk descends into media folders and skips the rest.")


def main():
    test_exact_and_tolerant_lookups()
    test_label_prefix_and_longer_name_match()
    test_no_false_positives()
    test_first_indexed_directory_wins()
    test_exact_name_beats_another_files_alias()
    test_index_directory_walks_and_skips()
    print("\nALL MEDIA INDEX TESTS PASSED!")


if __name__ == "__main__":
    main()
