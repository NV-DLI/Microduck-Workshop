// Recorded observations and their following labels; timestamps are display metadata.
(() => {
  const zh = document.body.dataset.lang === "zh";
  for (const figure of document.querySelectorAll("[data-demonstration]")) {
    const video = figure.querySelector("video");
    const stamp = figure.querySelector(".demonstration-time");
    const values = [...figure.querySelectorAll(".demonstration-action")];
    const sample = figure.querySelector(".demonstration-sample");
    fetch(figure.dataset.demonstration)
      .then((response) => {
        if (!response.ok) throw new Error("Sample trace unavailable");
        return response.json();
      })
      .then((trace) => {
        let shown = -1;
        function update(time = video.currentTime) {
          const index = Math.min(trace.rows.length - 1, Math.max(0, Math.floor(time * trace.sample_hz + 1e-6)));
          if (index === shown) return;
          const row = trace.rows[index];
          shown = index;
          figure.dataset.frame = String(row.frame);
          stamp.textContent = zh
            ? `第 ${row.frame} 帧 · 仿真时间 ${row.time.toFixed(2)} 秒 · 图像之后执行下面的动作`
            : `Frame ${row.frame} · simulation time ${row.time.toFixed(2)} s · the action follows this image`;
          values.forEach((cell, i) => { cell.textContent = row.action[i].toFixed(3); });
          sample.textContent = JSON.stringify({
            task: trace.task,
            "observation.state": row.state.map((value) => Number(value.toFixed(5))),
            action: row.action,
          }, null, 2);
        }
        for (const event of ["loadedmetadata", "timeupdate", "seeking", "seeked", "ended"])
          video.addEventListener(event, () => update());
        video.addEventListener("pause", () => update());
        if ("requestVideoFrameCallback" in video) {
          const onFrame = (_now, metadata) => {
            // A queued callback may describe the frame preceding a seek.
            // Paused scrubbing follows currentTime; playing follows decoded frames.
            if (!video.paused && !video.seeking && Math.abs(metadata.mediaTime - video.currentTime) < 0.12)
              update(metadata.mediaTime);
            video.requestVideoFrameCallback(onFrame);
          };
          video.requestVideoFrameCallback(onFrame);
        }
        update();
      })
      .catch(() => {
        stamp.textContent = zh
          ? "样本数据暂不可用；相机视频仍可独立播放。"
          : "Sample data unavailable; the camera video remains playable.";
      });
  }
})();
