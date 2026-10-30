#!/bin/bash
# Container entry: load course settings, prepare the course runtime once per
# start, then run the requested command (by default the container idles and
# learners open shells with `docker exec -it microduck-workshop bash`).
set -e

source /root/env 2>/dev/null || true

mkdir -p "${MICRODUCK_STATE:-/workspace/state}" /root/.cache/warp /root/.cache/ov /root/.nv/ComputeCache

# Bind the in-image Isaac Lab, detector and ffmpeg to the course modules. The
# relocated copy lives in $MICRODUCK_STATE, so a mounted repository never mixes
# host and container paths. Failures are reported, not hidden; the container
# keeps running so `python3 scripts/run.py doctor` can show the cause.
STATE="${MICRODUCK_STATE:-/workspace/state}"
rm -f "${STATE}/prepare.ok"
if python3 /workspace/Microduck-Workshop/scripts/container_runtime.py > "${STATE}/prepare.log" 2>&1; then
    touch "${STATE}/prepare.ok"
else
    echo "Course runtime preparation failed; see ${STATE}/prepare.log" >&2
    tail -n 5 "${STATE}/prepare.log" >&2 || true
fi

exec "$@"
