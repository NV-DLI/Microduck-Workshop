# Microduck Workshop image: Isaac Sim 6.2 + Isaac Lab (pinned commit) + Newton,
# the course detector environment and the GR00T learning environment.
#
# Layout follows the SO-101 sim-to-real workshop: one image for every lab, the
# course repository under /workspace/Microduck-Workshop, and outputs, datasets,
# checkpoints and caches mounted from the host at run time (docker/compose.yaml).
#
# Supported build platforms are the ones in lab/environment/uv.lock:
# linux/aarch64 (DGX Spark, the reference platform) and linux/x86_64.

FROM nvidia/cuda:13.0.0-base-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Both architectures (x86_64, aarch64) use the same pinned versions: uv.lock
# resolves wheels for both, PyTorch cu130 publishes both, and ffmpeg comes from
# apt. docker/prepare_host.sh rejects other architectures before a build.

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    git curl ca-certificates build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages uv==0.9.5

ARG ISAACLAB_COMMIT=5088b326c7fc81caa7899395b0176ad08b89e9e1
ARG ISAACLAB_ARCHIVE_SHA256=0f0817455bca9d5098a22e5591bea4faeae84b1af980f3c5e3c2682f365ec4ef

ENV UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=600 \
    UV_PYTHON_INSTALL_DIR=/workspace/python \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES

WORKDIR /workspace

# Isaac Lab at the course commit, verified against the archive digest that the
# host installer (scripts/setup_runtime.py) also checks.
RUN curl -fsSL -o isaaclab.tar.gz \
        "https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/${ISAACLAB_COMMIT}" \
    && echo "${ISAACLAB_ARCHIVE_SHA256}  isaaclab.tar.gz" | sha256sum -c - \
    && tar -xzf isaaclab.tar.gz \
    && mv "IsaacLab-${ISAACLAB_COMMIT}" IsaacLab \
    && rm isaaclab.tar.gz

# The course lock pins Isaac Sim 6.2.0.0, Newton 71eeba16, Warp 1.17.0,
# torch 2.11.0 (CUDA 13) and RSL-RL 5.5.1.
COPY lab/environment/pyproject.toml lab/environment/uv.lock /workspace/IsaacLab/
WORKDIR /workspace/IsaacLab
RUN uv sync --frozen --extra isaacsim --extra rsl-rl \
    && uv pip install --python .venv/bin/python \
        onnx==1.22.0 onnxruntime==1.26.0 tensorboard

# Detector environment: an overlay that reuses the Isaac Lab torch read-only.
COPY lab/runtime/simulation/core/perception/prepare_environment.py /tmp/prepare_detector.py
RUN /workspace/IsaacLab/.venv/bin/python /tmp/prepare_detector.py \
        --source-python /workspace/IsaacLab/.venv/bin/python \
        --venv /workspace/detector --install \
    && rm /tmp/prepare_detector.py

# GR00T learning environment: LeRobot at the commit that trained the course
# 8D walking checkpoint, with the exact package set of that environment.
COPY docker/groot-requirements.txt /tmp/groot-requirements.txt
RUN python3.12 -m venv /workspace/groot-env \
    && /workspace/groot-env/bin/pip install --no-cache-dir \
        torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cu130 \
    && /workspace/groot-env/bin/pip install --no-cache-dir -r /tmp/groot-requirements.txt \
    && /workspace/groot-env/bin/pip install --no-cache-dir --no-deps \
        "lerobot @ git+https://github.com/huggingface/lerobot.git@7e241bd630a3719a56157a497ce5d08f244784f1" \
    && /workspace/groot-env/bin/pip freeze > /workspace/groot-env/installed-packages.txt

RUN rm -rf /workspace/IsaacLab/docs /workspace/IsaacLab/.github

FROM nvidia/cuda:13.0.0-base-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Libraries for the Isaac Sim (Kit) window, RTX rendering and video export.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3.12 python3.12-venv libpython3.12t64 libatomic1 libgomp1 \
    ca-certificates curl git \
    libglvnd0 libgl1 libglx0 libopengl0 libegl1 libgles2 libvulkan1 libglu1-mesa \
    libx11-6 libxext6 libsm6 libxt6 libglib2.0-0 libnghttp2-14 \
    libxcursor1 libxrandr2 libxi6 libxinerama1 libxrender1 libxtst6 \
    libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 libnss3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# GPU driver hand-off for Vulkan/EGL, as in the Isaac Sim 6.2 container.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' \
        > /etc/vulkan/icd.d/nvidia_icd.json

# ROS 2 Jazzy Navfn planner (CPU only) for the onboard football route. The course
# container runs it as a child process (lab/course_labs/nav2_local.py) instead of a
# separate Docker container. Same steps on arm64 and amd64; the install record is
# written to /usr/local/share/microduck-nav2/.
COPY docker/nav2 /tmp/nav2
COPY lab/nav2/image-lock.json /tmp/nav2/image-lock.json
RUN NAV2_LOCK=/tmp/nav2/image-lock.json sh /tmp/nav2/install.sh && rm -rf /tmp/nav2

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /workspace /workspace

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    ISAACLAB_PATH=/workspace/IsaacLab \
    MICRODUCK_STATE=/workspace/state \
    MICRODUCK_GROOT_PYTHON=/workspace/groot-env/bin/python \
    MICRODUCK_NAV2_BACKEND=local \
    PYTHONPYCACHEPREFIX=/tmp/pycache

WORKDIR /workspace/Microduck-Workshop
COPY . /workspace/Microduck-Workshop

RUN mkdir -p /tmp/pycache /workspace/state \
    && chmod 1777 /tmp/pycache \
    && cat docker/utils.sh >> /root/.bashrc \
    && chmod +x docker/entrypoint.sh

LABEL org.opencontainers.image.title="Microduck Workshop" \
      org.opencontainers.image.source="https://github.com/NV-DLI/Microduck-Workshop" \
      org.microduck.isaaclab.commit="5088b326c7fc81caa7899395b0176ad08b89e9e1" \
      org.microduck.newton.commit="71eeba16a717d4dcf0b9dd7056a7b6f863054dc3"

# Healthy only after the course runtime was bound successfully at start-up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD test -f /workspace/state/prepare.ok

ENTRYPOINT ["/workspace/Microduck-Workshop/docker/entrypoint.sh"]
CMD ["sleep", "infinity"]
