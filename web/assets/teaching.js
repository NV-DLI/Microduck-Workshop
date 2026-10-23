(() => {
  const zh = document.body.dataset.lang === "zh";
  if (window.renderMathInElement)
    renderMathInElement(document.querySelector("main"), {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: true,
      strict: "error",
      trust: false,
    });
  document.addEventListener("click", (e) => {
    for (const menu of document.querySelectorAll(".language-menu[open]"))
      if (!menu.contains(e.target)) menu.open = false;
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape")
      for (const menu of document.querySelectorAll(".language-menu"))
        menu.open = false;
  });
  for (const f of document.querySelectorAll('[data-diagram="ppo"]')) {
    const q = (s) => f.querySelector(s);
    function update() {
      const a = Number(q(".advantage").value),
        ratio = Number(q(".prob-ratio").value),
        eps = Number(q(".clip-epsilon").value),
        low = a > 0 ? 0.4 : -1.6,
        high = a > 0 ? 1.6 : -0.4,
        x = (r) => 64 + (r - 0.5) * 510,
        y = (v) => 260 - ((v - low) / (high - low)) * 230,
        objective = (r) =>
          Math.min(r * a, Math.max(1 - eps, Math.min(1 + eps, r)) * a),
        path = (fn) =>
          Array.from({ length: 101 }, (_, i) => {
            const r = 0.5 + i / 100;
            return (i ? "L" : "M") + x(r) + "," + y(fn(r));
          }).join(" ");
      q(".ppo-original").setAttribute(
        "d",
        path((r) => r * a),
      );
      q(".ppo-clipped").setAttribute("d", path(objective));
      q(".ppo-point").setAttribute("cx", x(ratio));
      q(".ppo-point").setAttribute("cy", y(objective(ratio)));
      q(".clip-band").setAttribute("x", x(1 - eps));
      q(".clip-band").setAttribute("width", x(1 + eps) - x(1 - eps));
      q(".ppo-low").textContent = (1 - eps).toFixed(2);
      q(".ppo-high").textContent = (1 + eps).toFixed(2);
      q(".ppo-low").setAttribute("x", x(1 - eps) + 5);
      q(".ppo-high").setAttribute("x", x(1 + eps) + 5);
      q(".ppo-ymax").textContent = high.toFixed(1);
      q(".ppo-ymin").textContent = low.toFixed(1);
      q(".concept-output").textContent =
        `ρ = ${ratio.toFixed(2)}，ε = ${eps.toFixed(2)}；未裁剪目标 ${(ratio * a).toFixed(2)} → PPO 代理目标 ${objective(ratio).toFixed(2)}。${Math.abs(objective(ratio) - ratio * a) > 1e-8 ? "已进入平台区，继续沿这个方向改变概率，不再提高该样本的目标。" : "当前位置仍随概率比变化。"}`;
    }
    f.addEventListener("input", update);
    f.addEventListener("change", update);
    update();
  }
  for (const f of document.querySelectorAll(
    '[data-diagram="navigation-stance"]',
  )) {
    const q = (s) => f.querySelector(s);
    function update() {
      const d = Number(q(".nav-distance").value);
      const l = Number(q(".nav-offset").value);
      const b = [0, Number(q(".nav-ball-y").value)];
      const g = [0.68, Number(q(".nav-goal-y").value)];
      const length = Math.hypot(g[0] - b[0], g[1] - b[1]);
      const u = [(g[0] - b[0]) / length, (g[1] - b[1]) / length];
      const back = [b[0] - d * u[0], b[1] - d * u[1]];
      const p = [back[0] - l * u[1], back[1] + l * u[0]];
      const screen = (a) => [360 + 340 * a[0], 210 - 340 * a[1]];
      const path = (selector, points) =>
        q(selector).setAttribute(
          "d",
          points.map((a, i) => (i ? "L" : "M") + screen(a).join(",")).join(" "),
        );
      for (const [name, point] of [
        ["ball", b],
        ["goal", g],
        ["stance", p],
      ]) {
        const [x, y] = screen(point);
        q(".nav-" + name).setAttribute("cx", x);
        q(".nav-" + name).setAttribute("cy", y);
        q(".nav-" + name + "-label").setAttribute("x", x + 14);
        q(".nav-" + name + "-label").setAttribute(
          "y",
          y + (name === "stance" ? -17 : 24),
        );
      }
      path(".nav-shot", [back, g]);
      path(".nav-backoff", [back, b]);
      path(".nav-lateral", [back, p]);
      path(".nav-heading", [p, [p[0] + 0.12 * u[0], p[1] + 0.12 * u[1]]]);
      path(".nav-goal-frame", [
        [g[0] + 0.035, g[1] + 0.12],
        [g[0], g[1] + 0.12],
        [g[0], g[1] - 0.12],
        [g[0] + 0.035, g[1] - 0.12],
      ]);
      const mid = screen([(back[0] + b[0]) / 2, (back[1] + b[1]) / 2]);
      q(".nav-distance-label").setAttribute("x", mid[0]);
      q(".nav-distance-label").setAttribute("y", mid[1] + 24);
      for (const [selector, point, dx, dy] of [
        [
          ".nav-offset-label",
          [(back[0] + p[0]) / 2, (back[1] + p[1]) / 2],
          -22,
          5,
        ],
        [
          ".nav-direction-label",
          [(b[0] + g[0]) / 2, (b[1] + g[1]) / 2],
          0,
          -12,
        ],
        [".nav-heading-label", [p[0] + 0.14 * u[0], p[1] + 0.14 * u[1]], 3, -8],
      ]) {
        const [x, y] = screen(point);
        q(selector).setAttribute("x", x + dx);
        q(selector).setAttribute("y", y + dy);
      }
      const theta = (Math.atan2(u[1], u[0]) * 180) / Math.PI;
      f.dataset.targetX = p[0];
      f.dataset.targetY = p[1];
      f.dataset.heading = theta;
      q(".concept-output").textContent =
        `d = ${d.toFixed(2)} m, ℓ = ${l.toFixed(3)} m; B = (0, ${b[1].toFixed(2)}) m, G = (0.68, ${g[1].toFixed(2)}) m; P* = (${p[0].toFixed(3)}, ${p[1].toFixed(3)}) m; θ* = ${theta.toFixed(1)}°. ` +
        (zh
          ? "改变球或球门位置，观察目标坐标与朝向如何变化。"
          : "Move the ball or goal and observe the target position and heading.");
    }
    f.addEventListener("input", update);
    update();
  }
  for (const f of document.querySelectorAll('[data-diagram="stance"]')) {
    const q = (s) => f.querySelector(s);
    function update() {
      const d = Number(q(".stance-distance").value),
        l = Number(q(".stance-lateral").value),
        u = [0.92848, -0.37139],
        n = [-0.37139, -0.92848],
        x = 353 - 360 * d * u[0] + 360 * l * n[0],
        y = 149 - 360 * d * u[1] + 360 * l * n[1];
      q(".stance-robot").setAttribute("cx", x);
      q(".stance-robot").setAttribute("cy", y);
      q(".stance-label").setAttribute("x", Math.max(68, x - 85));
      q(".stance-label").setAttribute("y", Math.min(265, y + 32));
      q(".stance-offset").setAttribute(
        "d",
        `M${353 - 360 * d * u[0]},${149 - 360 * d * u[1]}L${x},${y}`,
      );
      q(".concept-output").textContent =
        `d = ${d.toFixed(2)} m，ℓ = ${l.toFixed(2)} m。滑块只改变示意站位；实际可用范围还受腿长、球的位置和踢球策略限制。`;
    }
    f.addEventListener("input", update);
    update();
  }
  for (const f of document.querySelectorAll('[data-diagram="terrain"]')) {
    const q = (s) => f.querySelector(s);
    function update() {
      const angle = Number(q(".terrain-angle").value),
        height = Math.tan((angle * Math.PI) / 180),
        y = 165 - height * 300;
      q(".terrain-surface").setAttribute("d", `M60 165H165L465 ${y}H585`);
      q(".terrain-fill").setAttribute("d", `M60 165H165L465 ${y}H585V265H60Z`);
      q(".terrain-height").setAttribute("d", `M475 165V${y}`);
      q(".terrain-height-label").setAttribute("y", (165 + y) / 2 - 7);
      q(".terrain-height-label").textContent =
        `${(height * 100).toFixed(1)} cm`;
      q(".terrain-landing").setAttribute("y", y - 16);
      q(".concept-output").textContent = zh
        ? `α = ${angle}°，坡道两端高差 ${height.toFixed(3)} m。示意图只改变几何，不预测策略是否能通过。`
        : `α = ${angle}°, height change ${height.toFixed(3)} m. This geometry diagram does not predict policy success.`;
    }
    f.addEventListener("input", update);
    update();
  }
  for (const link of document.querySelectorAll(".still-link"))
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const dialog = document.createElement("dialog");
      dialog.className = "image-dialog";
      const close = document.createElement("button");
      close.type = "button";
      close.textContent = "×";
      close.setAttribute("aria-label", zh ? "关闭图片" : "Close image");
      const img = new Image();
      img.src = link.href;
      img.alt = link.querySelector("img").alt;
      dialog.append(close, img);
      document.body.append(dialog);
      close.addEventListener("click", () => dialog.close());
      dialog.addEventListener("close", () => dialog.remove());
      dialog.addEventListener("click", (event) => {
        if (event.target === dialog) dialog.close();
      });
      dialog.showModal();
    });
  for (const video of document.querySelectorAll("video"))
    video.addEventListener("play", () => {
      for (const other of document.querySelectorAll("video"))
        if (other !== video) other.pause();
    });

  for (const video of document.querySelectorAll("video")) {
    if (
      !video
        .querySelector("source")
        ?.src.includes("autonomous-explained-hd.mp4")
    )
      continue;
    const panel = document.createElement("div");
    panel.className = "video-stage-caption";
    panel.setAttribute("aria-live", "polite");
    video.after(panel);
    fetch("../assets/media/autonomous-stages.json")
      .then((r) => {
        if (!r.ok) throw new Error("Stage descriptions unavailable");
        return r.json();
      })
      .then((stages) => {
        let shown;
        function updateStage() {
          const stage =
            stages.find(
              (s) => video.currentTime >= s.start && video.currentTime < s.end,
            ) || stages.at(-1);
          if (stage === shown) return;
          shown = stage;
          const title = document.createElement("strong");
          const detail = document.createElement("span");
          title.textContent = stage.title;
          detail.textContent = stage.detail;
          panel.replaceChildren(title, detail);
          panel.dataset.stage = stage.state;
        }
        for (const event of [
          "timeupdate",
          "seeking",
          "seeked",
          "loadedmetadata",
        ])
          video.addEventListener(event, updateStage);
        updateStage();
      })
      .catch(() => {
        panel.textContent = "请查看视频画面上方的阶段提示。";
      });
  }
})();
