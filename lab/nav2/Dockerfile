FROM ros@sha256:32114cd33511c91307d4f5dea184dcbfb9385860bee9839b772332c8fc103856
LABEL org.opencontainers.image.description="CPU-only Microduck Navfn probe runtime; official ROS Jazzy/Ubuntu apt packages"
ENV CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void PYTHONDONTWRITEBYTECODE=1
COPY deb-sha256.txt /usr/local/share/nav2-cpu/deb-sha256.txt
RUN --mount=type=bind,source=debs,target=/opt/probe-debs,readonly \
    cd /opt/probe-debs && sha256sum --check /usr/local/share/nav2-cpu/deb-sha256.txt && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends install /opt/probe-debs/*.deb && \
    dpkg-query -W -f='${binary:Package}\t${Version}\t${Architecture}\n' > /usr/local/share/nav2-cpu/dpkg.tsv && \
    rm -rf /var/lib/apt/lists/*
