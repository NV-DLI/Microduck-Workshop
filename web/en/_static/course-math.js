// Render course math with local KaTeX and explain marked symbols.
//
// Formulas mark symbols at build time as \htmlData{sym=key}{...}; the page's
// {symbols} tables (JSON) give each key its TeX and explanation. Only that one
// HTML extension is trusted. Interaction follows WCAG 1.4.13 (content on hover
// or focus): the explanation can be hovered, persists until the pointer or focus
// leaves, and Escape dismisses it.
(() => {
  const zh = document.body.dataset.lang === "zh";
  const main = document.querySelector("main");
  if (!main || !window.renderMathInElement) return;

  const KEY = /^[A-Za-z][A-Za-z0-9_-]*$/;
  renderMathInElement(main, {
    delimiters: [
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: true,
    // Keep strict mode; allow only the HTML extension used for symbol marks.
    strict: (code) => (code === "htmlExtension" ? "ignore" : "error"),
    trust: (context) =>
      context.command === "\\htmlData" &&
      Object.keys(context.attributes).length === 1 &&
      KEY.test(context.attributes["data-sym"] || ""),
  });

  const table = {};
  for (const script of document.querySelectorAll("script.course-symbols")) {
    try { Object.assign(table, JSON.parse(script.textContent)); } catch (error) { console.error(error); }
  }
  const formulas = [...main.querySelectorAll(".katex")].filter((k) => k.querySelector("[data-sym]"));
  if (!formulas.length) return;

  // Shared tooltip, keyboard hint and announcement region.
  const tip = document.createElement("div");
  tip.id = "course-symbol-tip";
  tip.className = "symbol-tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  tip.innerHTML = '<div class="symbol-tip-tex"></div><p class="symbol-tip-text"></p>';
  const hint = document.createElement("span");
  hint.id = "course-symbol-hint";
  hint.hidden = true;
  hint.textContent = zh
    ? "公式中的变量带有含义说明：用左右方向键逐个查看，按 Esc 关闭。"
    : "Variables in this formula have explanations: use the arrow keys to step through them; Escape closes.";
  const live = document.createElement("span");
  live.className = "sr-only";
  live.setAttribute("aria-live", "polite");
  document.body.append(tip, hint, live);
  const tipTex = tip.querySelector(".symbol-tip-tex");
  const tipText = tip.querySelector(".symbol-tip-text");

  for (const root of formulas) {
    // The .katex wrapper holds the accessible MathML and is not aria-hidden,
    // so it is the focus target; marked symbols live in the visual layer.
    root.tabIndex = 0;
    root.classList.add("has-symbols");
    root.setAttribute("aria-describedby", hint.id);
  }

  let current = null; // { el, root, mode: "pointer" | "anchor" | "focus" }
  let hideTimer = 0;
  const margin = 8;

  function place(point) {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const vw = document.documentElement.clientWidth, vh = window.innerHeight;
    let x, y;
    if (point) {
      x = point.x + 14; y = point.y + 18;
      if (x + w > vw - margin) x = point.x - w - 14;
      if (y + h > vh - margin) y = point.y - h - 12;
    } else {
      const r = current.el.getBoundingClientRect();
      x = r.left + r.width / 2 - w / 2;
      y = r.bottom + margin;
      if (y + h > vh - margin) y = r.top - h - margin;
    }
    x = Math.min(Math.max(margin, x), Math.max(margin, vw - w - margin));
    y = Math.min(Math.max(margin, y), Math.max(margin, vh - h - margin));
    tip.style.left = `${Math.round(x)}px`;
    tip.style.top = `${Math.round(y)}px`;
  }

  function show(el, mode, point) {
    const entry = table[el.dataset.sym];
    if (!entry) return;
    clearTimeout(hideTimer);
    if (current?.el !== el) {
      current?.el.classList.remove("is-symbol-active");
      try { katex.render(entry.tex, tipTex, { throwOnError: false }); } catch { tipTex.textContent = entry.tex; }
      renderText(tipText, entry.text);
    }
    const root = el.closest(".katex");
    // Only the formula that currently shows the tooltip points to it.
    if (current && current.root !== root) current.root.setAttribute("aria-describedby", hint.id);
    current = { el, root, mode };
    el.classList.add("is-symbol-active");
    tip.hidden = false;
    root.setAttribute("aria-describedby", `${hint.id} ${tip.id}`);
    // Keyboard selection inside a long, horizontally scrolling formula stays in view.
    if (mode === "focus") el.scrollIntoView({ block: "nearest", inline: "nearest" });
    place(mode === "pointer" ? point : null);
  }

  // Explanations are plain text with optional $TeX$ segments.
  function renderText(node, text) {
    node.replaceChildren();
    text.split(/\$([^$]+)\$/).forEach((part, i) => {
      if (i % 2 === 0) { if (part) node.append(part); return; }
      const span = document.createElement("span");
      try { katex.render(part, span, { throwOnError: false }); } catch { span.textContent = part; }
      node.append(span);
    });
  }
  const spoken = (text) => text.replace(/\$([^$]+)\$/g, (_, tex) => tex.replace(/\\/g, ""));

  function hide() {
    clearTimeout(hideTimer);
    if (!current) return;
    current.el.classList.remove("is-symbol-active");
    current.root.setAttribute("aria-describedby", hint.id);
    current = null;
    tip.hidden = true;
  }

  const symbolAt = (target) => (target instanceof Element ? target.closest("[data-sym]") : null);
  const symbolsIn = (root) => [...root.querySelectorAll("[data-sym]")];

  // Mouse and pen: show near the pointer; the tooltip itself can be hovered.
  document.addEventListener("pointerover", (event) => {
    if (event.pointerType === "touch") return;
    const el = symbolAt(event.target);
    if (el && !(current?.mode === "anchor" && current.el === el)) {
      show(el, "pointer", { x: event.clientX, y: event.clientY });
    } else if (tip.contains(event.target)) {
      clearTimeout(hideTimer);
    }
  });
  document.addEventListener("pointerout", (event) => {
    if (event.pointerType === "touch" || !current || current.mode !== "pointer") return;
    const to = event.relatedTarget;
    if (to instanceof Node && (tip.contains(to) || symbolAt(to) === current.el)) return;
    if (symbolAt(event.target) || tip.contains(event.target)) hideTimer = setTimeout(hide, 300);
  });

  // Tap or click: pin the explanation under the symbol; tap again or elsewhere to close.
  document.addEventListener("click", (event) => {
    const el = symbolAt(event.target);
    if (el) {
      if (current?.el === el && current.mode === "anchor") hide();
      else show(el, "anchor");
    } else if (current && !tip.contains(event.target) && current.mode !== "focus") {
      hide();
    }
  });

  // Keyboard: one tab stop per formula; arrows step through its symbols.
  for (const root of formulas) {
    root.addEventListener("focus", () => {
      const first = symbolsIn(root)[0];
      if (first) { show(first, "focus"); announce(first); }
    });
    root.addEventListener("blur", () => { if (current?.root === root) hide(); });
    root.addEventListener("keydown", (event) => {
      const list = symbolsIn(root);
      let i = current?.root === root ? list.indexOf(current.el) : -1;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") i = (i + 1) % list.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") i = (i - 1 + list.length) % list.length;
      else if (event.key === "Home") i = 0;
      else if (event.key === "End") i = list.length - 1;
      else return;
      event.preventDefault();
      show(list[i], "focus");
      announce(list[i]);
    });
  }
  function announce(el) {
    const entry = table[el.dataset.sym];
    if (entry) live.textContent = spoken(entry.text);
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && current) { hide(); }
  });
  const follow = () => { if (current && current.mode !== "pointer") place(null); else if (current) hide(); };
  // Capture phase also catches horizontally scrolling formula containers.
  document.addEventListener("scroll", follow, { capture: true, passive: true });
  window.addEventListener("resize", follow);
})();
