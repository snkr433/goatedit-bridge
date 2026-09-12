"""What the local filesystem bridge must refuse.

Most of these are written from the attacker's side rather than the user's: the
threat is not a person typing a path, it is an agent that has read a poisoned
web page and is now asking, in good faith, for ~/.ssh/id_rsa.
"""

from __future__ import annotations

import base64
import os

import pytest

from goatedit_bridge.localfs import (
    LocalFs,
    LocalFsError,
    contains,
)


@pytest.fixture()
def fs(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "clip.mp4").write_bytes(b"video bytes")
    (shared / "notes.txt").write_text("not media")

    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "id_rsa").write_text("PRIVATE KEY")
    (secret / "stolen.mp4").write_bytes(b"outside the root")

    frames = tmp_path / "frames"
    return LocalFs([str(shared)], str(frames)), shared, secret, frames


# --- containment ------------------------------------------------

def test_contains_does_not_fall_for_a_shared_prefix(tmp_path):
    # The bug this exists to prevent: "/work" startswith-matches
    # "/workspace-elsewhere", which is a directory anyone can create next to
    # the real one.
    root = str(tmp_path / "work")
    os.makedirs(root)
    sibling = str(tmp_path / "workspace-elsewhere")
    os.makedirs(sibling)
    assert contains(root, os.path.join(root, "a.mp4"))
    assert not contains(root, os.path.join(sibling, "a.mp4"))


def test_symlink_out_of_the_root_is_not_followed(fs):
    local, shared, secret, _ = fs
    os.symlink(secret / "stolen.mp4", shared / "innocent.mp4")
    # The walk finds the name inside the root; realpath then puts it outside,
    # and the containment check is what catches it.
    with pytest.raises(LocalFsError, match="not found"):
        local.resolve("innocent.mp4")


# --- reading ----------------------------------------------------

def test_finds_a_media_file_inside_a_shared_root(fs):
    local, shared, _, _ = fs
    assert local.resolve("clip.mp4") == os.path.realpath(str(shared / "clip.mp4"))


def test_refuses_a_non_media_extension_even_inside_the_root(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="not a media file"):
        local.resolve("notes.txt")


def test_refuses_a_private_key_by_extension_before_it_ever_searches(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="not a media file"):
        local.resolve("id_rsa")


def test_traversal_is_flattened_to_a_basename(fs):
    local, _, _, _ = fs
    # basename("../../secret/id_rsa") is "id_rsa", which then fails on extension.
    with pytest.raises(LocalFsError, match="not a media file"):
        local.resolve("../../secret/id_rsa")


def test_a_media_file_outside_every_root_is_invisible(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="not found"):
        local.resolve("stolen.mp4")


def test_absolute_path_to_a_real_file_outside_the_root_still_fails(fs):
    local, _, secret, _ = fs
    with pytest.raises(LocalFsError, match="not found"):
        local.resolve(str(secret / "stolen.mp4"))


def test_dot_directories_are_not_walked(fs):
    local, shared, _, _ = fs
    hidden = shared / ".ssh"
    hidden.mkdir()
    (hidden / "backup.mp4").write_bytes(b"x")
    with pytest.raises(LocalFsError, match="not found"):
        local.resolve("backup.mp4")


def test_no_roots_configured_says_so_rather_than_searching_everything(tmp_path):
    local = LocalFs([], str(tmp_path / "frames"))
    with pytest.raises(LocalFsError, match="No folders are shared"):
        local.resolve("clip.mp4")


# --- handles ----------------------------------------------------

def test_handle_round_trips_a_verified_path(fs):
    local, shared, _, _ = fs
    token = local.mint(local.resolve("clip.mp4"))
    handle = local.lookup(token)
    assert handle.path == os.path.realpath(str(shared / "clip.mp4"))
    assert handle.size == len(b"video bytes")


def test_minting_a_path_outside_the_roots_is_refused(fs):
    local, _, secret, _ = fs
    with pytest.raises(LocalFsError, match="outside the folders"):
        local.mint(str(secret / "stolen.mp4"))


def test_an_unknown_handle_is_not_served(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="expired"):
        local.lookup("not-a-real-handle")


def test_handles_are_unguessable_and_distinct(fs):
    local, _, _, _ = fs
    path = local.resolve("clip.mp4")
    tokens = {local.mint(path) for _ in range(20)}
    assert len(tokens) == 20
    assert all(len(t) > 20 for t in tokens)


# --- writing ----------------------------------------------------

def _png() -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 32).decode()


def test_writes_a_frame_into_the_frames_directory(fs):
    local, _, _, frames = fs
    handle = local.write_frame(_png(), "image/png", "preview")
    assert os.path.isfile(handle.path)
    assert contains(os.path.realpath(str(frames)), handle.path)
    assert handle.path.endswith(".png")


def test_a_label_cannot_steer_the_write_out_of_the_directory(fs):
    local, _, _, frames = fs
    handle = local.write_frame(_png(), "image/png", "../../../../etc/passwd")
    # Every separator and dot is filtered out of the label, so what is left is
    # a flat name inside the one directory.
    assert contains(os.path.realpath(str(frames)), handle.path)
    assert "etc" not in os.path.dirname(handle.path)


def test_two_frames_with_the_same_label_do_not_collide(fs):
    local, _, _, _ = fs
    first = local.write_frame(_png(), "image/png", "same")
    second = local.write_frame(_png(), "image/png", "same")
    assert first.path != second.path


def test_refuses_a_content_type_it_does_not_write(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="not an image type"):
        local.write_frame(_png(), "application/x-sh", "shell")


def test_refuses_bytes_that_are_not_base64(fs):
    local, _, _, _ = fs
    with pytest.raises(LocalFsError, match="not valid base64"):
        local.write_frame("!!!! not base64 !!!!", "image/png")


def test_refuses_a_frame_over_the_size_cap(fs, monkeypatch):
    local, _, _, _ = fs
    monkeypatch.setattr("goatedit_bridge.localfs.MAX_FRAME_BYTES", 16)
    with pytest.raises(LocalFsError, match="over the"):
        local.write_frame(base64.b64encode(b"0" * 64).decode(), "image/png")


def test_a_written_frame_can_be_served_by_handle(fs):
    local, _, _, _ = fs
    written = local.write_frame(_png(), "image/png", "preview")
    handle = local.lookup(local.mint(written.path, "image/png"))
    assert handle.path == written.path
