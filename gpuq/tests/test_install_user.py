"""install_user.sh: ~/.local/bin/gpuq points at the root-owned binary (todo E2)."""
import os
import subprocess
from pathlib import Path

GPUQ = Path(__file__).resolve().parent.parent
SCRIPT = GPUQ / "install_user.sh"


def run(tmp_path, *args):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    sysbin = tmp_path / "usr-local-bin-gpuq"
    if not sysbin.exists():
        sysbin.write_text("#!/bin/sh\n")
        sysbin.chmod(0o755)
    env = dict(os.environ, HOME=str(home), GPUQ_SYSTEM_BIN=str(sysbin))
    r = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)
    return r, home / ".local/bin/gpuq", sysbin


def test_the_default_links_to_the_root_owned_binary(tmp_path):
    r, link, sysbin = run(tmp_path)
    assert r.returncode == 0, r.stderr
    assert os.readlink(link) == str(sysbin)


def test_an_old_link_to_the_shared_copy_is_replaced(tmp_path):
    link = tmp_path / "home/.local/bin/gpuq"
    link.parent.mkdir(parents=True)
    link.symlink_to("/var/lib/gpu_queue/gpuq.py")
    r, link, sysbin = run(tmp_path)
    assert os.readlink(link) == str(sysbin)


def test_retired_flags_say_so_and_still_link_to_the_binary(tmp_path):
    for flag in ("--publish-shared", "--symlink-shared", "--copy-shared"):
        r, link, sysbin = run(tmp_path, flag)
        assert r.returncode == 0, r.stderr
        assert f"{flag} is retired" in r.stderr
        assert os.readlink(link) == str(sysbin)


def test_no_code_line_refers_to_the_queue_dir():
    # The only way to touch the shared copy would be to name it.
    code = [l for l in SCRIPT.read_text().splitlines() if not l.lstrip().startswith("#")]
    assert not [l for l in code if "/var/lib/gpu_queue" in l]


def test_copy_from_repo_is_a_private_copy(tmp_path):
    r, link, _ = run(tmp_path, "--copy-from-repo")
    assert r.returncode == 0, r.stderr
    assert not link.is_symlink()
    assert link.read_bytes() == (GPUQ / "userspace.py").read_bytes()


def test_a_missing_binary_is_reported_not_linked(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), GPUQ_SYSTEM_BIN=str(tmp_path / "absent"))
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "install_system.sh first" in r.stderr
    assert not (home / ".local/bin/gpuq").is_symlink()
