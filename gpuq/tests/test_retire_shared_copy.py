"""retire_shared_copy.sh (todo E2), run into a temp prefix.

Fake getent lists the accounts; fake runuser records who each command ran as
and then runs it (the tests are not root). Link targets are compared as real
paths, so every link here points at /var/lib/gpu_queue/gpuq.py or
/usr/local/bin/gpuq exactly as on the host.
"""
import os
import subprocess
from pathlib import Path

GPUQ = Path(__file__).resolve().parent.parent
SCRIPT = GPUQ / "retire_shared_copy.sh"
SHARED = "/var/lib/gpu_queue/gpuq.py"
SYSTEM = "/usr/local/bin/gpuq"


def host(tmp_path, runuser_fails_for=""):
    root = tmp_path / "root"
    q = root / "var/lib/gpu_queue"
    (q / "__pycache__").mkdir(parents=True)
    (q / "gpuq.py").write_text("# shared copy\n")
    (q / "__pycache__/gpuq.cpython-312.pyc").write_text("")
    (q / "running.json").write_text("[]")
    sysbin = root / "usr/local/bin/gpuq"
    sysbin.parent.mkdir(parents=True)
    sysbin.write_text("#!/bin/sh\n")
    sysbin.chmod(0o755)
    homes = {}
    for user in ["alice", "bob", "carol", "dave", "erin"]:
        (root / "home" / user / ".local/bin").mkdir(parents=True)
        homes[user] = root / "home" / user / ".local/bin/gpuq"
    homes["alice"].symlink_to(SHARED)
    homes["bob"].symlink_to(SYSTEM)
    homes["carol"].write_text("# private copy\n")
    homes["erin"].symlink_to("/opt/elsewhere/gpuq")
    passwd = "".join(f"{u}:x:{1000 + i}:{1000 + i}::/home/{u}:/bin/bash\n"
                     for i, u in enumerate(homes))
    passwd += "daemon:x:1:1::/usr/sbin:/usr/sbin/nologin\nroot:x:0:0::/:/bin/bash\n"
    getent = tmp_path / "getent"
    getent.write_text(f"#!/bin/sh\ncat <<'EOF'\n{passwd}EOF\n")
    runuser = tmp_path / "runuser"
    log = tmp_path / "runuser.log"
    runuser.write_text(
        '#!/bin/bash\n'
        f'echo "$*" >> {log}\n'
        f'[[ "$2" == "{runuser_fails_for}" ]] && exit 1\n'
        'shift 3\nexec "$@"\n')
    for f in (getent, runuser):
        f.chmod(0o755)
    env = dict(os.environ, GPUQ_DESTDIR=str(root), GETENT=str(getent), RUNUSER=str(runuser))
    return root, homes, env, log


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)


def test_the_default_is_a_dry_run_that_changes_nothing(tmp_path):
    root, homes, env, log = host(tmp_path)
    r = run(env)
    assert r.returncode == 0, r.stderr
    assert "would re-point  alice" in r.stdout and "would remove" in r.stdout
    assert os.readlink(homes["alice"]) == SHARED
    assert (root / "var/lib/gpu_queue/gpuq.py").exists() and not log.exists()


def test_apply_repoints_only_links_to_the_shared_copy_then_removes_it(tmp_path):
    root, homes, env, log = host(tmp_path)
    r = run(env, "--apply")
    assert r.returncode == 0, r.stderr
    assert os.readlink(homes["alice"]) == SYSTEM
    assert os.readlink(homes["bob"]) == SYSTEM
    assert homes["carol"].read_text() == "# private copy\n"
    assert not homes["dave"].exists() and not homes["dave"].is_symlink()
    assert os.readlink(homes["erin"]) == "/opt/elsewhere/gpuq"
    q = root / "var/lib/gpu_queue"
    assert not (q / "gpuq.py").exists() and not (q / "__pycache__").exists()
    assert (q / "running.json").exists()
    assert "left alone      carol" in r.stdout and "left alone      erin" in r.stdout


def test_each_link_is_changed_as_its_owner_never_as_root(tmp_path):
    _, _, env, log = host(tmp_path)
    run(env, "--apply")
    calls = log.read_text().splitlines()
    assert len(calls) == 1 and calls[0].startswith("-u alice -- ln -sfn /usr/local/bin/gpuq ")


def test_a_failed_repoint_keeps_the_shared_copy(tmp_path):
    root, homes, env, _ = host(tmp_path, runuser_fails_for="alice")
    r = run(env, "--apply")
    assert r.returncode == 1
    assert "FAILED          alice" in r.stderr and "KEPT" in r.stderr
    assert os.readlink(homes["alice"]) == SHARED
    assert (root / "var/lib/gpu_queue/gpuq.py").exists()


def test_other_python_files_in_the_queue_dir_are_reported_not_touched(tmp_path):
    root, _, env, _ = host(tmp_path)
    planted = root / "var/lib/gpu_queue/json.py"
    planted.write_text("")
    r = run(env, "--apply")
    assert "NOTE: other .py files" in r.stdout and "json.py" in r.stdout
    assert planted.exists()


def test_nothing_happens_without_the_root_owned_binary(tmp_path):
    root, homes, env, _ = host(tmp_path)
    (root / "usr/local/bin/gpuq").unlink()
    r = run(env, "--apply")
    assert r.returncode == 1 and "install_system.sh first" in r.stderr
    assert os.readlink(homes["alice"]) == SHARED


def test_without_the_test_prefix_it_needs_root():
    if os.geteuid() == 0:
        return
    env = {k: v for k, v in os.environ.items() if k != "GPUQ_DESTDIR"}
    r = run(env)
    assert r.returncode == 1 and "Needs root" in r.stderr
