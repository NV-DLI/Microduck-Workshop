"""Check the installed ROS 2 Nav2 packages against the course lock and record them.

Fails when a direct package is missing or its upstream version differs from
lab/nav2/image-lock.json. Exact package identities (with build stamps) are
compared too and recorded; they match only on the architecture of the lock.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream(version):
    return version.rsplit("-", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--packages", type=Path, required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text())
    wanted = a.packages.read_text().split()
    if sorted(wanted) != sorted(lock["direct_packages"]):
        raise SystemExit("packages.txt differs from the lock's direct packages")
    arch = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
    inventory = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${Architecture}\n"], text=True)
    installed = {}
    for line in inventory.splitlines():
        name, version, _ = line.split("\t")
        installed[name.split(":")[0]] = version
    direct = {}
    for name in wanted:
        if name not in installed:
            raise SystemExit("Missing direct package: " + name)
        locked = lock["direct_packages"][name]
        direct[name] = dict(installed=installed[name], locked=locked,
                            upstream_equal=upstream(installed[name]) == upstream(locked),
                            identical=installed[name] == locked)
    bad = [n for n, d in direct.items() if not d["upstream_equal"]]
    if bad:
        raise SystemExit("Upstream version differs from the lock: " + ", ".join(bad))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "dpkg.tsv").write_text(inventory)
    record = dict(
        schema="microduck.nav2-container-install.v1",
        written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        architecture=arch,
        source=f"http://snapshots.ros.org/jazzy/{a.snapshot}/ubuntu noble main",
        key_sha256=sha(a.key),
        key_fingerprint="4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA",
        lock_sha256=sha(a.lock),
        lock_architecture=lock["architecture"],
        direct_packages=direct,
        all_direct_identical_to_lock=all(d["identical"] for d in direct.values()),
        ros_packages=sum(1 for n in installed if n.startswith("ros-jazzy-")),
        dpkg_inventory_sha256=sha(a.output / "dpkg.tsv"),
        dpkg_packages=len(installed),
    )
    (a.output / "install-record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("architecture", "all_direct_identical_to_lock", "ros_packages")}))


if __name__ == "__main__":
    main()
