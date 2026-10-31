"""Run the scene Nav2 planning service inside the course container, without Docker.

The course container has no Docker CLI or socket, so the onboard football route
cannot start the separate Nav2 image that lab/scene_nav2/owner.py creates on a host
install. LocalServiceOwner runs the same verified lab/scene_nav2/service.py (same
Navfn planner, nav2.yaml, file protocol, run nonce, request/response checks and
bounded service lifetime) as a CPU-only child process of the course container, using
the ROS 2 Jazzy packages installed by docker/nav2/install.sh.

Isolation in place of the container flags: GPUs hidden; a clean environment; one
CPU (taskset prefix, so nothing runs in the multithreaded Kit parent between fork
and exec); ROS discovery limited to this host on a ROS domain that this owner holds
with an exclusive lock until close() and whose DDS ports are not bound by anyone
else; UDP transport only (no shared-memory files, since the course container
shares the host IPC namespace); a memory and wall-time watchdog. Every child
carries an ownership token in its environment; close() stops the service, then
terminates any process that still carries the token and checks that none remain,
even when an earlier cleanup stage fails.

  python3 lab/course_labs/nav2_local.py check --output <new folder>

runs the preflight, starts the service, replays one recorded onboard query (path
compared with the recorded one), a no-path query, an invalid request and one more
valid query, then shuts down and checks cleanup. It needs no simulator and no GPU.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
from scene_nav2.contract import atomic, read  # noqa: E402
from scene_nav2.owner import verify  # noqa: E402

ROS = Path("/opt/ros/jazzy")
LOCK = LAB / "nav2/image-lock.json"
PREFLIGHT = Path(__file__).with_name("nav2_local_preflight.py")
INSTALL_RECORD = Path("/usr/local/share/microduck-nav2/install-record.json")
MEMORY_LIMIT_BYTES = 384 * 2**20  # same budget as the container owner
PREFERRED_DOMAIN = 179  # the container owner's ROS domain
TOKEN_VARIABLE = "MICRODUCK_NAV2_OWNER_TOKEN"
TASKSET = Path("/usr/bin/taskset")  # util-linux
# Shared by every process of the course container (all lab shells use one container).
LOCK_DIR = Path(os.environ.get("MICRODUCK_NAV2_LOCK_DIR", "/tmp/microduck-nav2-domains"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nav2_backend(environ=os.environ):
    """`local` in the course container (MICRODUCK_NAV2_BACKEND=local), `docker` on a host install."""
    value = environ.get("MICRODUCK_NAV2_BACKEND")
    if value in ("local", "docker"):
        return value
    if value:
        raise ValueError("MICRODUCK_NAV2_BACKEND must be local or docker")
    if shutil.which("docker") is None and (ROS / "setup.bash").is_file():
        return "local"
    return "docker"


def bound_udp_ports(paths=("/proc/net/udp", "/proc/net/udp6")):
    ports = set()
    for path in paths:
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            ports.add(int(line.split()[1].rsplit(":", 1)[1], 16))
    return ports


def domain_ports(domain):
    base = 7400 + 250 * domain
    return set(range(base, base + 250))


def reserve_domain(token, lock_dir=LOCK_DIR, preferred=PREFERRED_DOMAIN, ports=bound_udp_ports):
    """Hold the first ROS domain (preferred, then 180..232, 1..178) that is free.

    Free means: no other owner holds its exclusive lock file, and, checked after the
    lock is taken, none of its DDS ports (7400 + 250 d ... + 249) is bound by any
    process. Selection and holding are one step, so two owners starting together
    never share a domain. Returns (domain, descriptor); the lock lasts until the
    descriptor is closed by every process holding it.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    for domain in [preferred, *range(preferred + 1, 233), *range(1, preferred)]:
        fd = os.open(lock_dir / f"domain-{domain:03d}.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        if ports() & domain_ports(domain):
            os.close(fd)
            continue
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"owner_pid": os.getpid(), "ownership_token": token}).encode() + b"\n")
        return domain, fd
    raise RuntimeError("No ROS domain with free DDS ports")


def token_pids(token, proc=Path("/proc")):
    needle = f"{TOKEN_VARIABLE}={token}".encode()
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if needle in (entry / "environ").read_bytes().split(b"\0"):
                found.append(int(entry.name))
        except OSError:
            continue
    return sorted(found)


def pss_bytes(pid):
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def shm_files():
    try:
        return sorted(p.name for p in Path("/dev/shm").iterdir() if "fastrtps" in p.name or "fastdds" in p.name)
    except OSError:
        return []


class LocalServiceOwner:
    """Same constructor, start() and close() contract as scene_nav2.owner.ServiceOwner."""

    def __init__(self, source_dir, exchange_dir, output_dir, run_nonce, max_wall_s=1200):
        self.source = Path(source_dir).resolve()
        self.exchange = Path(exchange_dir).resolve()
        self.output = Path(output_dir).resolve()
        self.nonce = run_nonce
        self.max_wall_s = max_wall_s
        self.token = uuid.uuid4().hex
        self.process = None
        self.log = None
        self.cleanup = None
        self.started = None
        self.domain = None
        self.domain_fd = None
        self.watch = {"peak_PSS_bytes": 0, "memory_limit_hit": False, "wall_limit_hit": False}
        self._stop = threading.Event()
        self._watchdog = None
        if not isinstance(run_nonce, str) or not 16 <= len(run_nonce) <= 128 or not 1 <= max_wall_s <= 1800:
            raise ValueError("Explicit nonce/bounded lifetime required")
        if (self.output == self.source or self.source in self.output.parents
                or self.exchange == self.source or self.source in self.exchange.parents):
            raise ValueError("Output outside source required")

    def environment(self, domain):
        return {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "HOME": str(self.output),
            "ROS_HOME": str(self.output / "ros-home"),
            "ROS_LOG_DIR": str(self.output / "ros-logs"),
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "PYTHONDONTWRITEBYTECODE": "1",
            "ROS_DOMAIN_ID": str(domain),
            "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
            "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            TOKEN_VARIABLE: self.token,
        }

    def _ros(self, argv):
        # CPU pinning is part of the command: the parent (a multithreaded Kit process)
        # must not run Python code between fork and exec, so no preexec_fn.
        script = 'source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 -B "$@"'
        return [str(TASKSET), "--cpu-list", str(self.cpu),
                "/bin/bash", "-c", script, "nav2-local", *map(str, argv)]

    def start(self, timeout_s=30):
        if not (ROS / "setup.bash").is_file():
            raise RuntimeError("ROS 2 Jazzy Nav2 is not installed in this container (docker/nav2/install.sh)")
        if not TASKSET.is_file():
            raise RuntimeError("taskset (util-linux) is required to pin the Nav2 service to one CPU")
        self.source_pin = verify(self.source)
        self.exchange.mkdir(parents=True, exist_ok=False)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "result").mkdir()
        try:
            self.domain, self.domain_fd = reserve_domain(self.token)
            self.cpu = max(os.sched_getaffinity(0))
            env = self.environment(self.domain)
            self.shm_before = shm_files()
            command = self._ros([self.source / "service.py", "--exchange", self.exchange,
                                 "--output", self.output / "result", "--run-nonce", self.nonce,
                                 "--max-wall-s", self.max_wall_s])
            atomic(self.output / "plan.json", {
                "kind": "persistent_Navfn_CPU_owner_local_v1",
                "backend": "local ROS 2 Jazzy child process in the course container",
                "source_sha256": self.source_pin,
                "install_record_sha256": sha(INSTALL_RECORD) if INSTALL_RECORD.is_file() else None,
                "ownership_token": self.token,
                "run_nonce": self.nonce,
                "max_wall_s": self.max_wall_s,
                "cpu_affinity": [self.cpu],
                "memory_watchdog_MiB": MEMORY_LIMIT_BYTES // 2**20,
                "ros_domain_id": self.domain,
                "ros_domain_lock": str(LOCK_DIR / f"domain-{self.domain:03d}.lock"),
                "discovery_range": "LOCALHOST",
                "dds_transport": "UDPv4 only",
                "GPU": False,
                "command": command,
                "environment": {k: v for k, v in env.items() if k != TOKEN_VARIABLE},
            })
            pre = subprocess.run(
                self._ros([PREFLIGHT, "--output", self.output / "image-preflight.json", "--lock", LOCK]),
                env=env, cwd=self.output, capture_output=True, text=True, timeout=60)
            if pre.returncode or read(self.output / "image-preflight.json")["status"] != "passed":
                raise RuntimeError("Nav2 preflight failed: " + (pre.stderr or "")[-400:])
            self.log = (self.output / "service.log").open("wb")
            self.started = time.monotonic()
            # The service inherits the domain lock, so the domain stays reserved while it
            # runs even if this process dies first.
            self.process = subprocess.Popen(command, env=env, cwd=self.output, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True,
                                            pass_fds=(self.domain_fd,))
            self._watchdog = threading.Thread(target=self._watch, daemon=True)
            self._watchdog.start()
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                p = self.exchange / "ready.json"
                if p.exists():
                    r = read(p)
                    if r.get("run_nonce") == self.nonce and r.get("accepting_requests"):
                        return r
                if self.process.poll() is not None:
                    raise RuntimeError("Service exited before ready")
                time.sleep(0.02)
            raise TimeoutError("Service readiness deadline")
        except BaseException:
            self.close()
            raise

    def _watch(self):
        while not self._stop.wait(0.25):
            total = sum(pss_bytes(p) for p in token_pids(self.token))
            self.watch["peak_PSS_bytes"] = max(self.watch["peak_PSS_bytes"], total)
            if total > MEMORY_LIMIT_BYTES:
                self.watch["memory_limit_hit"] = True
            elif time.monotonic() - self.started > self.max_wall_s + 20:
                self.watch["wall_limit_hit"] = True
            else:
                continue
            self._terminate(signal.SIGKILL)
            return

    def _terminate(self, sig):
        for pid in token_pids(self.token):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _graceful(self, result):
        if self.process is None or self.process.poll() is not None:
            return
        result["graceful_shutdown"] = False
        atomic(self.exchange / "shutdown.json", {"run_nonce": self.nonce})
        try:
            self.process.wait(timeout=15)
            result["graceful_shutdown"] = True
        except subprocess.TimeoutExpired:
            pass  # the process group is stopped next

    def _stop_group(self, result):
        if self.process is None or self.process.poll() is not None:
            return
        for sig, wait_s in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                continue

    def _sweep(self, result):
        leftover = token_pids(self.token)
        result["token_processes_before_cleanup"] = leftover
        if leftover:
            self._terminate(signal.SIGTERM)
            end = time.monotonic() + 5
            while token_pids(self.token) and time.monotonic() < end:
                time.sleep(0.05)
            self._terminate(signal.SIGKILL)
            end = time.monotonic() + 5
            while token_pids(self.token) and time.monotonic() < end:
                time.sleep(0.05)
        if self.process is not None and self.process.poll() is None:
            self.process.wait(timeout=5)  # reap the direct child after the token kill

    def close(self):
        """Stop the service and remove every owned process; each stage runs even if an earlier one fails."""
        if self.cleanup is not None:
            return self.cleanup
        result = {
            "backend": "local",
            "ownership_token": self.token,
            "service_pid": None if self.process is None else self.process.pid,
            "ros_domain_id": self.domain,
            "graceful_shutdown": None,
            "service_exit_code": None,
            "token_processes_before_cleanup": None,
            "owned_processes_zero": False,
            "service_status": None,
            "errors": [],
        }
        interrupted = []

        def stage(name, function, *args):
            try:
                function(*args)
            except BaseException as exc:
                result["errors"].append(f"{name}: {exc!r}")
                if not isinstance(exc, Exception):
                    interrupted.append(exc)

        def stop_watchdog():
            self._stop.set()
            if self._watchdog is not None:
                self._watchdog.join(timeout=2)
            result["watchdog"] = dict(self.watch)

        def owned_processes():
            remaining = token_pids(self.token)
            result.update(owned_processes_zero=not remaining, postscan_token_pids=remaining)
            if self.process is not None:
                result["service_exit_code"] = self.process.poll()

        def release_domain():
            # Kept while an owned process might survive; the lock then ends with this process.
            if self.domain_fd is not None and result["owned_processes_zero"]:
                os.close(self.domain_fd)
                self.domain_fd = None
            result["ros_domain_lock_released"] = self.domain_fd is None

        def shared_memory():
            if hasattr(self, "shm_before"):
                result["new_dds_shared_memory_files"] = sorted(set(shm_files()) - set(self.shm_before))

        def source_unchanged():
            if hasattr(self, "source_pin") and verify(self.source) != self.source_pin:
                raise RuntimeError("Bridge source changed during ownership")

        def service_summary():
            summary = self.output / "result/summary.json"
            if summary.exists():
                data = read(summary)
                result.update(service_status=data["status"], owned_planner_reaped=data.get("owned_planner_reaped"),
                              summary_sha256=sha(summary))

        def close_log():
            if self.log is not None:
                self.log.close()

        stage("graceful_shutdown", self._graceful, result)
        stage("stop_process_group", self._stop_group, result)
        stage("stop_watchdog", stop_watchdog)
        stage("token_sweep", self._sweep, result)
        stage("owned_process_scan", owned_processes)
        stage("release_domain", release_domain)
        stage("shared_memory_scan", shared_memory)
        stage("source_check", source_unchanged)
        stage("service_summary", service_summary)
        stage("close_log", close_log)
        if result["errors"]:
            result["error"] = result["errors"][0]  # same key as ServiceOwner.close()
        self.cleanup = result
        try:
            if self.output.exists():
                atomic(self.output / "cleanup.json", result)
        finally:
            if interrupted:
                raise interrupted[0]
        return result


def check(output):
    """Real start, four transactions and cleanup against the recorded onboard query."""
    from scene_nav2.client import BridgeClient
    from scene_nav2.contract import SCHEMA

    fixture = json.loads(Path(__file__).with_name("nav2_fixture").joinpath("recorded-transaction.json").read_text())
    output = Path(output).resolve()
    if output.exists():
        raise SystemExit("Choose a new output folder: " + str(output))
    nonce = uuid.uuid4().hex
    owner = LocalServiceOwner(LAB / "scene_nav2", output / "exchange", output / "service", nonce, 120)
    report = dict(schema="microduck.nav2-local-check.v1", backend=nav2_backend(), run_nonce=nonce, cases={})
    started = time.monotonic()
    try:
        report["ready"] = owner.start()
        client = BridgeClient(output / "exchange", nonce)
        t0 = fixture["sim_query_s"]
        # 1. Recorded query, replayed unchanged.
        r = client.query(fixture["case"], fixture["provenance"], 0, t0)
        recorded = fixture["recorded_response"]
        deviation = max((max(abs(a - b) for a, b in zip(p, q))
                         for p, q in zip(r["path_xy"], recorded["path_xy"])), default=None)
        report["cases"]["recorded_query"] = dict(
            status=r["status"], action_status=r["action_status"], error_code=r["error_code"],
            path_points=len(r["path_xy"]), recorded_points=len(recorded["path_xy"]),
            max_point_deviation_m=deviation, map_sha256_equal=r["map_sha256"] == recorded["map_sha256"],
            envelope=r["envelope"], planning_wall_s=r["wall_elapsed_s"],
            passed=(r["status"] == "complete" and len(r["path_xy"]) == len(recorded["path_xy"])
                    and deviation is not None and deviation < 1e-6 and r["envelope"]["accepted"]))
        # 2. Same recorded snapshot, goal moved off the surveyed field (unknown cells): no path.
        case = json.loads(json.dumps(fixture["case"]))
        case["query_s"] = t0 + 0.05  # recorded captures unchanged (50 ms old, inside the age limits)
        x, y, _ = case["odom"]["pose_xyyaw"]
        case["goal_body_xy"] = [0.2 - x, 0.85 - y]
        try:
            r = client.query(case, fixture["provenance"], 1, t0 + 0.05)
            report["cases"]["goal_off_field"] = dict(
                status=r["status"], action_status=r["action_status"], error_code=r["error_code"],
                path_points=len(r["path_xy"]), envelope=r["envelope"],
                passed=not r["path_xy"] and not r["envelope"]["accepted"])
        except Exception as exc:  # a failed transaction is also a refusal to plan
            response = read(output / "exchange/response.json")
            report["cases"]["goal_off_field"] = dict(client_error=repr(exc), status=response.get("status"),
                                                     error=response.get("error"),
                                                     passed=response.get("status") == "failed" and not response.get("path_xy"))
        # 3. Invalid request: wrong run nonce, written directly (the client refuses to send it).
        bad = dict(schema=SCHEMA, run_nonce=uuid.uuid4().hex, sequence=2, sim_query_s=t0 + 0.1,
                   sent_monotonic_ns=time.monotonic_ns(), case=fixture["case"], provenance=fixture["provenance"])
        atomic(output / "exchange/request.json", bad)
        end = time.monotonic() + 5
        response = {}
        while time.monotonic() < end:
            response = read(output / "exchange/response.json")
            if response.get("sequence") == 2:
                break
            time.sleep(0.02)
        report["cases"]["wrong_nonce"] = dict(status=response.get("status"), error=response.get("error"),
                                              path_points=len(response.get("path_xy", [])),
                                              passed=response.get("status") == "failed"
                                              and "nonce" in str(response.get("error")) and not response.get("path_xy"))
        # 4. The service keeps serving after the refused request: recorded snapshot queried
        # 80 ms later (inside the 100 ms odometry age limit).
        case = json.loads(json.dumps(fixture["case"]))
        case["query_s"] = t0 + 0.08
        r = client.query(case, fixture["provenance"], 3, t0 + 0.08)
        report["cases"]["after_invalid"] = dict(
            status=r["status"], path_points=len(r["path_xy"]), envelope=r["envelope"],
            passed=r["status"] == "complete" and len(r["path_xy"]) > 0 and r["envelope"]["accepted"])
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["cleanup"] = owner.close()
        report["wall_s"] = time.monotonic() - started
    cleanup = report["cleanup"]
    report["passed"] = ("error" not in report and len(report["cases"]) == 4
                        and all(c["passed"] for c in report["cases"].values())
                        and cleanup["owned_processes_zero"] and cleanup["ros_domain_lock_released"]
                        and not cleanup["errors"]
                        and cleanup["service_status"] == "complete"
                        and cleanup.get("owned_planner_reaped") is True
                        and not cleanup.get("new_dds_shared_memory_files"))
    atomic(output / "check.json", report)
    print(json.dumps(dict(passed=report["passed"], error=report.get("error"),
                          cases={k: v["passed"] for k, v in report["cases"].items()},
                          cleanup={k: cleanup.get(k) for k in ("owned_processes_zero", "service_status", "ros_domain_id")})))
    return 0 if report["passed"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    return check(a.output)


if __name__ == "__main__":
    raise SystemExit(main())
