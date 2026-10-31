// Open the interactive 3D anatomy viewer (web/assets/robot-viewer.js) on request:
// poster first, then download progress, then the model; a failed load keeps the
// poster and offers to try again.
(() => {
  const zh = document.body.dataset.lang === "zh";
  for (const viewer of document.querySelectorAll("[data-robot-viewer]")) {
    const button = viewer.querySelector(".load-robot");
    const label = button.querySelector(".load-robot-label");
    const launch = viewer.querySelector(".viewer-launch");
    const progress = viewer.querySelector(".viewer-progress");
    const bar = viewer.querySelector(".viewer-progress-bar");
    const text = viewer.querySelector(".viewer-progress-text");
    const status = viewer.querySelector(".viewer-status");
    const show = (fraction) => {
      const known = Number.isFinite(fraction);
      progress.classList.toggle("is-indeterminate", !known);
      bar.style.width = known ? `${Math.round(fraction * 100)}%` : "";
      text.textContent = `${text.dataset.label}… ${known ? Math.round(fraction * 100) + "%" : ""}`.trim();
    };
    button.addEventListener("click", async () => {
      viewer.dataset.state = "loading";
      launch.hidden = true;
      progress.hidden = false;
      status.textContent = "";
      show(NaN);
      try {
        const { mountRobot } = await import("../../assets/robot-viewer.js");
        await mountRobot(viewer, document.body.dataset.lang, show);
        viewer.dataset.state = "ready";
        launch.remove();
        progress.remove();
        viewer.querySelector(".robot-fallback").remove();
      } catch (error) {
        viewer.dataset.state = "error";
        progress.hidden = true;
        launch.hidden = false;
        label.textContent = button.dataset.labelRetry;
        status.textContent = zh
          ? "3D 模型未能加载，请检查网络后重试。下方说明与装配图仍可阅读。"
          : "The 3D model did not load. Check the connection and try again; the rendering and notes remain available.";
        button.focus();
        console.error(error);
      }
    });
  }
})();

// Walking-actor network: pointing at, focusing or tapping a layer column or a
// band of connections highlights it and shows its size below the figure; the
// input and output columns also show their interface tables. The explanation
// stays while the pointer moves from the diagram down to read it.
(() => {
  for (const figure of document.querySelectorAll("[data-network]")) {
    const readout = figure.querySelector(".network-readout");
    const panel = figure.querySelector(".network-panel");
    const details = [...figure.querySelectorAll(".network-detail")];
    const parts = [...figure.querySelectorAll(".network-part")];
    let pinned = null, hovered = null, timer = 0;
    const show = (part) => {
      clearTimeout(timer);
      hovered = part;
      parts.forEach((p) => p.classList.toggle("is-active", p === part));
      figure.classList.toggle("has-active", Boolean(part));
      readout.textContent = part ? part.dataset.info : readout.dataset.default;
      details.forEach((d) => (d.hidden = !part || d.dataset.for !== part.dataset.part));
    };
    // Leaving the diagram clears the hover after a short grace period, so the
    // pointer can reach the explanation below; entering the panel keeps it.
    const leaveSoon = () => {
      clearTimeout(timer);
      if (!pinned) timer = setTimeout(() => show(null), 450);
    };
    for (const part of parts) {
      part.addEventListener("pointerenter", (event) => { if (event.pointerType !== "touch" && !pinned) show(part); });
      part.addEventListener("pointerleave", (event) => { if (event.pointerType !== "touch") leaveSoon(); });
      part.addEventListener("focus", () => { if (!pinned) show(part); });
      part.addEventListener("blur", (event) => { if (!pinned && !figure.contains(event.relatedTarget)) show(null); });
      const toggle = () => { pinned = pinned === part ? null : part; show(pinned); };
      part.addEventListener("click", toggle);
      part.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(); }
      });
    }
    panel.addEventListener("pointerenter", () => { if (hovered) clearTimeout(timer); });
    panel.addEventListener("pointerleave", (event) => { if (event.pointerType !== "touch") leaveSoon(); });
    figure.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { pinned = null; show(null); }
    });
    document.addEventListener("click", (event) => {
      if (pinned && !figure.contains(event.target)) { pinned = null; show(null); }
    });
    show(null);
  }
})();

// "How often each module updates": recorded rounds k−1, k, k+1.
// Progress p runs 0–1 across the visible board. Every item carries data-t0/t1
// (active) and data-x (no longer in effect) and is marked future, active, live
// or expired. Dependency arrows are laid out from the rows they connect.
(() => {
  for (const root of document.querySelectorAll("[data-seq-timing]")) {
    const config = JSON.parse(root.dataset.config);
    const scrub = root.querySelector(".timing-scrub"), play = root.querySelector(".timing-replay");
    const say = root.querySelector(".timing-status");
    const radios = [...root.querySelectorAll(".timing-modes [data-mode]")];
    let mode = radios[0].dataset.mode, frame = 0, start = 0, running = false, shown = "";
    const board = () => root.querySelector(`.seq-board[data-board="${mode}"]`);
    function layout() {
      const b = board(), svg = b.querySelector(".seq-links"), box = svg.getBoundingClientRect();
      if (!box.width) return;
      svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
      const track = (key) => b.querySelector(`[data-row="${key}"] .seq-track`).getBoundingClientRect();
      for (const g of svg.querySelectorAll(".seq-link")) {
        const a = track(g.dataset.from), z = track(g.dataset.to);
        const x1 = a.left - box.left + Number(g.dataset.x1) * a.width, x2 = z.left - box.left + Number(g.dataset.x2) * z.width;
        const y1 = a.bottom - box.top, y2 = z.top - box.top - 1, ym = (y1 + y2) / 2;
        const line = x1 === x2 ? `M${x1} ${y1}V${y2 - 4}` : `M${x1} ${y1}V${ym}H${x2}V${y2 - 4}`;
        g.querySelector(".seq-link-line").setAttribute("d", line);
        g.querySelector(".seq-link-head").setAttribute("d", `M${x2 - 3.5} ${y2 - 5}L${x2 + 3.5} ${y2 - 5}L${x2} ${y2}Z`);
      }
    }
    function render() {
      const p = Number(scrub.value) / 1000;
      for (const b of root.querySelectorAll(".seq-board")) b.hidden = b.dataset.board !== mode;
      for (const el of root.querySelectorAll(".timing-facts [data-modes]")) el.hidden = !el.dataset.modes.split(" ").includes(mode);
      const active = board();
      let phase = null, text = null;
      for (const el of active.querySelectorAll("[data-t0]")) {
        const t0 = Number(el.dataset.t0), t1 = Number(el.dataset.t1), x = Number(el.dataset.x);
        const on = p >= t0 && (p < t1 || (p >= 1 && t1 >= 0.9999));
        const state = p < t0 ? "future" : on ? "active" : p < x ? "live" : "expired";
        if (el.dataset.state !== state) el.dataset.state = state;
        if (on && el.dataset.status) phase = el;
        if (on && el.dataset.say) text = el.dataset.say;
      }
      active.classList.toggle("is-inferring", phase?.dataset.status === "infer");
      for (const cursor of active.querySelectorAll(".seq-cursor")) cursor.style.left = `${p * 100}%`;
      if (text && text !== shown) { shown = text; say.textContent = text; }
    }
    function stop() { cancelAnimationFrame(frame); running = false; play.textContent = play.dataset.play; }
    function tick(now) {
      if (!running) return;
      scrub.value = Math.min(1000, ((now - start) / config.replay[mode]) * 1000);
      render();
      if (Number(scrub.value) >= 1000) stop(); else frame = requestAnimationFrame(tick);
    }
    function reset() { stop(); shown = ""; scrub.value = 0; render(); layout(); }
    play.addEventListener("click", () => {
      if (running) { stop(); return; }
      if (Number(scrub.value) >= 1000) scrub.value = 0;
      running = true;
      start = performance.now() - (Number(scrub.value) / 1000) * config.replay[mode];
      play.textContent = play.dataset.pause;
      frame = requestAnimationFrame(tick);
    });
    scrub.addEventListener("input", () => { stop(); render(); });
    const syncTabs = () => radios.forEach((r) => (r.tabIndex = r.dataset.mode === mode ? 0 : -1));
    for (const radio of radios) {
      radio.addEventListener("click", () => {
        mode = radio.dataset.mode;
        radios.forEach((r) => r.setAttribute("aria-checked", String(r === radio)));
        syncTabs();
        reset();
      });
    }
    root.querySelector(".timing-modes").addEventListener("keydown", (event) => {
      const i = radios.findIndex((r) => r.dataset.mode === mode);
      const next = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 0;
      if (!next) return;
      event.preventDefault();
      const target = radios[(i + next + radios.length) % radios.length];
      target.click(); target.focus();
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
    // Row heights change when fonts and KaTeX labels finish rendering.
    new ResizeObserver(layout).observe(root);
    window.addEventListener("load", layout);
    syncTabs();
    reset();
  }
})();

// Briefly tint the title text of the section an in-page link, the initial hash or back/forward
// leads to. Replays when the same contents entry is clicked again; a newer target cancels the
// previous tint. Scrolling, history and focus stay native.
(() => {
  const article = document.querySelector(".bd-article");
  if (!article) return;
  const isHeading = (el) => /^H[1-6]$/.test(el.tagName);
  const ownHeading = (section) => section.querySelector(":scope > :is(h1, h2, h3, h4, h5, h6)");
  const headingFor = (id) => {
    const el = document.getElementById(id);
    if (!el || !article.contains(el)) return null;
    if (isHeading(el)) return el;
    if (el.tagName === "SECTION") return ownHeading(el);
    if (el.tagName !== "SPAN" || el.childNodes.length) return null;
    // A MyST target span sits just before its heading or section.
    let next = el.nextElementSibling;
    while (next && next.tagName === "SPAN" && !next.childNodes.length) next = next.nextElementSibling;
    if (!next) return null;
    if (isHeading(next)) return next;
    return next.tagName === "SECTION" ? ownHeading(next) : null;
  };
  // Wrap the title (not the permalink) once, moving the existing nodes so math, inline markup
  // and their listeners stay as they are; the heading's own box and layout do not change.
  const titleOf = (heading) => {
    let text = heading.querySelector(":scope > .md-target-text");
    if (text) return text;
    text = document.createElement("span");
    text.className = "md-target-text";
    const permalink = heading.querySelector(":scope > .headerlink");
    for (const node of [...heading.childNodes]) {
      if (node === permalink) break;
      text.append(node);
    }
    heading.insertBefore(text, permalink);
    if (!text.textContent.trim()) {
      text.replaceWith(...text.childNodes);
      return null;
    }
    // Whitespace before the permalink stays outside so the tint ends at the last character.
    let trailing = text.lastChild;
    while (trailing && trailing.nodeType === Node.TEXT_NODE && !trailing.textContent.trim()) {
      heading.insertBefore(trailing, text.nextSibling);
      trailing = text.lastChild;
    }
    return text;
  };
  let current = null;
  let timer = 0;
  const flash = (hash) => {
    let id = hash.slice(1);
    try { id = decodeURIComponent(id); } catch { /* keep the raw id */ }
    clearTimeout(timer);
    if (current) current.classList.remove("md-target-flash");
    current = null;
    const heading = id && headingFor(id);
    const title = heading && titleOf(heading);
    if (!title) return;
    void title.offsetWidth; // restart the animation on the same title
    title.classList.add("md-target-flash");
    current = title;
    // One timer ends both the 1.2 s fade and the reduced-motion static tint.
    timer = setTimeout(() => { title.classList.remove("md-target-flash"); current = null; }, 1200);
  };
  let clicked = null;
  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest?.("a[href*='#']");
    if (!link || (link.target && link.target !== "_self")) return;
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin || url.pathname !== location.pathname || url.search !== location.search || !url.hash) return;
    // The hashchange this click causes (if the hash differs) must not flash again.
    clicked = url.hash === location.hash ? null : url.hash;
    flash(url.hash);
  });
  addEventListener("hashchange", () => {
    if (clicked !== null && clicked === location.hash) { clicked = null; return; }
    clicked = null;
    flash(location.hash);
  });
  if (location.hash) flash(location.hash);
})();

// Sections that moved to another page when the course was reorganized keep
// working: an old link such as 07-groot.html#lab-finetuning lands on the new place.
(() => {
  const moved = {
    "03-learning": {practice: "03-walk-training#train", "longer-practice": "03-walk-training#train", "training-time": "03-walk-training#watch", checkpoint: "03-walk-training#test", randomization: "03-walk-training#reality-gap"},
    "07-groot": {route: "08-groot-interface#task", actions: "08-groot-interface#why-high-level", "reuse-skills": "08-groot-interface#reuse-skills", "walking-interface": "08-groot-interface#inputs", "walking-finetuning-scope": "08-groot-interface#action-head", data: "08-groot-interface#data-design", "data-coverage": "08-groot-interface#data-design", "compare-policies": "07-groot#compare", "state-memory": "08-groot-interface#inputs", "lab-environment": "08-groot-lab#before", "lab-demonstrations": "08-groot-interface#data-design", "lab-finetuning": "08-groot-lab#lab-train", "lab-rollout": "08-groot-lab#lab-validate", correction: "08-groot-interface#improve-data", "lab-correction-collect": "08-groot-interface#improve-data", "lab-correction-train": "08-groot-interface#improve-data", "lab-correction-test": "08-groot-interface#improve-data", deployment: "08-groot-interface#deployment"},
    "09-worlds": {"lab-cosmos-source": "09-worlds#cosmos", "lab-cosmos-review": "09-worlds#cosmos", "lab-cosmos-train": "09-worlds#cosmos"},
  };
  const page = location.pathname.split("/").pop().replace(/\.html$/, "");
  const id = decodeURIComponent(location.hash.slice(1));
  const target = moved[page] && moved[page][id];
  if (target && !document.getElementById(id)) {
    const [file, anchor] = target.split("#");
    location.replace(file === page ? "#" + anchor : file + ".html#" + anchor);
  }
})();
