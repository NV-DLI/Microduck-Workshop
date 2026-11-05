/* Course control cycles · control-cycle schematic (walking 2.2 / onboard 4.3 / GR00T 5.2).
   All times are CONFIGURED simulation periods; playback is teaching slow motion. */
document.querySelectorAll('[data-control-cycle]').forEach(host => {
  'use strict';
  const shadow = host.attachShadow({mode: 'open'});
  shadow.append(host.querySelector('template').content.cloneNode(true));

  const SCENARIOS = host.dataset.scenarios.split(',');
  const CAP_ON = [0, 200 / 3, 400 / 3];                 // onboard RGB + ToF, nominal 15 Hz
  const CAP_GR = [0, 200 / 3, 400 / 3, 200, 800 / 3];   // GR00T RGB, nominal 15 Hz
  const L = 'ABCDE';
  const WALK_CYCLES = ['t−1', 't', 't+1'];
  const TOTAL = { walking: 60, onboard: 200, groot: 320 };
  const PLAY_MS = { walking: 6000, onboard: 8000, groot: 8000 }; // real ms for the whole sim window
  const HOST_PLAY_MS = 3000;   // illustrative inference progress, NOT a timer
  const HOST_HOLD_MS = 700;    // pause on "16 requests ready" before execution starts
  const STEP_MS_REDUCED = 900;
  const EPS = 1e-6;
  // GR00T track layout (fractions of track width): host wait | axis break | simulation execution.
  const SEGS = { host: [0, 0.17], brk: [0.17, 0.215], exec: [0.215, 1] };
  const mqReduce = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

  let lang = host.dataset.lang;
  let theme = 'light';
  let cur = SCENARIOS[0];
  let playing = false;
  let raf = 0;
  let timer = 0;
  let lastTs = null;
  let model = null;
  let lastKey = '';
  let dom = {};

  const fresh = (sc) => (sc === 'groot'
    ? { phase: 'inference', host: 0, hold: 0, time: 0, ended: false }
    : { phase: 'cycle', host: 1, hold: 0, time: 0, ended: false });
  const store = {};
  SCENARIOS.forEach((s) => { store[s] = fresh(s); });

  const tr = (zh, en) => (lang === 'zh' ? zh : en);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const num = (v) => { const r = Math.round(v * 10) / 10; return Math.abs(r - Math.round(r)) < EPS ? String(Math.round(r)) : r.toFixed(1); };
  const ms = (v) => num(v) + ' ms';
  const span = (a, b) => `${num(a)}–${num(b)} ms`;
  const latest = (caps, t) => { let i = -1; for (let j = 0; j < caps.length; j += 1) if (caps[j] <= t + EPS) i = j; return i; };
  const steps = (a, b, d) => { const out = []; for (let v = a; v <= b + EPS; v += d) out.push(v); return out; };
  const uniq = (arr) => arr.slice().sort((x, y) => x - y).filter((v, i, a) => i === 0 || v - a[i - 1] > EPS);

  /* ------------------------------------------------------------------ models */
  function buildModel(sc) {
    if (sc === 'walking') {
      return {
        total: 60, ticks: [0, 20, 40, 60], minor: 5,
        aria: tr('行走控制周期示意：0、20、40 ms 各读取一次状态并输出 14 个关节目标；每组目标保持 20 ms，覆盖 4 个 5 ms 物理步。',
          'Walking cycle schematic: the state is read and 14 joint targets are output at 0, 20 and 40 ms; each set is held for 20 ms, covering four 5 ms physics steps.'),
        lanes: [
          { type: 'index', key: 'k', label: tr('控制周期序号', 'Control cycle no.'), sub: tr('t 为当前周期 · 每周期 20 ms', 't = current · 20 ms per cycle'),
            items: [0, 1, 2].map((i) => ({ s: i * 20, e: i * 20 + 20, text: WALK_CYCLES[i], currentCycle: i === 1 })) },
          { type: 'points', kind: 'update', key: 'upd', label: tr('关节目标更新时点', 'Joint-target update'), sub: tr('Actor 推理完成后', 'after Actor inference'),
            items: [0, 20, 40, 60].map((t, i) => ({ t, text: i < 3 ? ms(t) : ms(t) + tr(' · 下一次', ' · next'), ghost: i === 3 })) },
          { type: 'cells', role: 'hold', key: 'hold', label: tr('14 个关节目标', '14 joint targets'), sub: tr('每格 20 ms · 50 Hz', '20 ms per box · 50 Hz'),
            items: [0, 1, 2].map((i) => ({ s: i * 20, e: i * 20 + 20, text: '20 ms', alt: i % 2, currentCycle: i === 1 })) },
          { type: 'cells', role: 'phys', key: 'phys', label: tr('Newton 物理步', 'Newton physics steps'), sub: '200 Hz',
            items: steps(0, 55, 5).map((s, j) => ({ s, e: s + 5, text: '5 ms', alt: Math.floor(j / 4) % 2, currentCycle: Math.floor(j / 4) === 1 })) },
        ],
      };
    }
    if (sc === 'onboard') {
      const up = steps(0, 180, 20);
      return {
        total: 200, ticks: [0, 40, 80, 120, 160, 200], minor: 20,
        aria: tr('板载策略周期示意：RGB 与 ToF 在名义时刻 0、66.7、133.3 ms 采集（A、B、C），每次采集后运行 YOLO 推理并更新几何与依赖图像的阶段状态，结果沿用到下一次采集；每 20 ms 反馈控制逻辑读取身体反馈，结合最近视觉结果更新运动请求，随后底层 Actor 推理输出 14 个关节目标并保持 20 ms。',
          'Onboard cycle schematic: RGB and ToF are captured at nominal 0, 66.7 and 133.3 ms (A, B, C); after each capture YOLO inference runs and geometry and image-dependent stage state are updated, then reused until the next capture; every 20 ms feedback control logic reads body feedback and updates the motion request with the latest vision result, then low-level Actor inference outputs 14 joint targets held for 20 ms.'),
        lanes: [
          { type: 'points', kind: 'capture', key: 'cap', label: tr('RGB + ToF 采集', 'RGB + ToF capture'), sub: tr('名义 15 Hz', 'nominal 15 Hz'),
            items: CAP_ON.map((t, i) => ({ t, name: L[i], text: ms(t) }))
              .concat([{ t: 200, name: 'D', text: '200 ms', ghost: true }]) },
          { type: 'points', kind: 'vinfer', key: 'yolo', label: tr('YOLO 推理与位置估计', 'YOLO inference + localization'), sub: tr('每次新采集后 · 约 15 Hz', 'after each new capture · ~15 Hz'),
            items: CAP_ON.map((t) => ({ t, name: 'YOLO' })).concat([{ t: 200, name: 'YOLO', ghost: true }]) },
          { type: 'cells', role: 'vision', key: 'avail', label: tr('视觉结果（沿用）', 'Vision result (reused)'), sub: tr('沿用到下一次采集', 'held until the next capture'),
            items: CAP_ON.map((t, i) => { const e = i < 2 ? CAP_ON[i + 1] : 200; return { s: t, e, text: `${L[i]} · ${ms(e - t)}` }; }) },
          { type: 'index', key: 'k', label: tr('控制更新序号', 'Control update no.'), sub: tr('每 20 ms 更新', 'every 20 ms'),
            items: up.map((t, i) => ({ s: t, e: t + 20, text: '#' + (i + 1) })) },
          { type: 'points', kind: 'update', key: 'ctl', label: tr('反馈控制 → 运动请求', 'Feedback control → motion request'), sub: tr('身体反馈 · 50 Hz', 'body feedback · 50 Hz'),
            items: up.map((t) => ({ t })).concat([{ t: 200, ghost: true }]) },
          { type: 'cells', role: 'request', key: 'req', tall: true, label: tr('运动请求', 'Motion request'), sub: tr('每 20 ms 更新 · 50 Hz', 'updated every 20 ms · 50 Hz'),
            items: up.map((t) => ({ s: t, e: t + 20, text: '20 ms', small: L[latest(CAP_ON, t)] })) },
          { type: 'points', kind: 'update', key: 'upd', label: tr('底层 Actor 推理', 'Low-level Actor inference'), sub: tr('每 20 ms · 50 Hz', 'every 20 ms · 50 Hz'),
            items: up.map((t) => ({ t })).concat([{ t: 200, ghost: true }]) },
          { type: 'cells', role: 'actor', key: 'act', label: tr('14 个关节目标', '14 joint targets'), sub: tr('每格 20 ms · 50 Hz', '20 ms per box · 50 Hz'),
            items: up.map((t, i) => ({ s: t, e: t + 20, text: '20 ms', alt: i % 2 })) },
        ],
      };
    }
    return {
      total: 320, ticks: [0, 80, 160, 240, 320], minor: 20, segs: SEGS,
      aria: tr('GR00T 时序示意：0 ms 推理前快照（RGB A + 身体状态 + 指令）→ 主机上约 300 ms 的新推理，仿真暂停（与仿真时间不等比例）→ 断轴 → 仿真执行 0–320 ms：4 条请求各 80 ms，请求 1 来自该快照，80、160、240 ms 整理观察并同步取回缓存请求 2–4，每 20 ms 运行底层 Actor；320 ms 用新快照开始下一块推理。',
        'GR00T sequence schematic: a pre-inference snapshot at 0 ms (RGB A + body state + instruction) → about 300 ms of new inference on the host while simulation is paused (not to scale with simulation time) → axis break → simulation execution 0–320 ms: four 80 ms requests, request 1 from that snapshot, observations assembled at 80, 160 and 240 ms with cached requests 2–4 fetched synchronously, low-level Actor every 20 ms; at 320 ms a new snapshot starts the next chunk.'),
      lanes: [
        { type: 'points', kind: 'capture', key: 'cap', label: tr('RGB 采集', 'RGB capture'), sub: tr('名义 15 Hz', 'nominal 15 Hz'),
          items: [{ t: 0, pre: true, name: 'A', text: tr('0 ms · 推理前', '0 ms · initial') }]
            .concat(CAP_GR.slice(1).map((t, i) => ({ t, name: L[i + 1], text: ms(t) }))) },
        { type: 'points', kind: 'assemble', key: 'asm', label: tr('整理观察 → 同步请求', 'Assemble observation → sync request'), sub: tr('每 80 ms（含缓存步）', 'every 80 ms (incl. cached steps)'),
          items: [{ t: 0, pre: true, infer: true, name: tr('推理前快照', 'input snapshot'), text: '0 ms · RGB A' }]
            .concat([80, 160, 240].map((t, i) => ({ t, name: tr(`取缓存 #${i + 2}`, `cached #${i + 2}`), text: `RGB ${L[latest(CAP_GR, t)]}` })))
            .concat([{ t: 320, infer: true, ghost: true, name: tr('下一块：新推理', 'next chunk: infer'), text: `320 ms · RGB ${L[latest(CAP_GR, 320)]}` }]) },
        { type: 'cells', role: 'host', key: 'host', label: tr('GR00T 推理', 'GR00T inference'), sub: tr('主机时间 · 非等比例', 'host time · not to scale'),
          items: [{ seg: 'host', s: 0, e: 1, text: '≈ 300 ms', small: tr('主机计算', 'host compute') }],
          execNote: tr('使用预测缓存', 'Use cached predictions') },
        { type: 'index', key: 'q', label: tr('请求序号', 'Request no.'), sub: tr('共预测 16 条', '16 predicted'),
          items: [0, 1, 2, 3].map((i) => ({ s: i * 80, e: i * 80 + 80, text: tr(`#${i + 1} / 16`, `#${i + 1} of 16`) })) },
        { type: 'cells', role: 'request', key: 'req', pauseLabel: true, label: tr('高层请求', 'High-level request'), sub: tr('每条 80 ms · 8 个数', '80 ms each · 8 values'),
          items: [0, 1, 2, 3].map((i) => ({ s: i * 80, e: i * 80 + 80, text: '80 ms' })) },
        { type: 'points', kind: 'update', key: 'upd', label: tr('身体反馈 → 底层 Actor 推理', 'Body feedback → low-level Actor inference'), sub: tr('每 20 ms 更新', 'update every 20 ms'),
          items: steps(0, 300, 20).map((t) => ({ t })).concat([{ t: 320, ghost: true }]) },
        { type: 'cells', role: 'actor', key: 'act', label: tr('14 个关节目标', '14 joint targets'), sub: tr('每格 20 ms · 50 Hz', '20 ms per box · 50 Hz'),
          items: steps(0, 300, 20).map((s, i) => ({ s, e: s + 20, text: '20 ms', alt: Math.floor(i / 4) % 2 })) },
      ],
    };
  }

  /* --------------------------------------------------------------- templates */
  const ICON = {
    play: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>',
    pause: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>',
    replay: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5a5.5 5.5 0 1 1-5.3 4H1l2.6-3.2L6.3 6.5H4.3A3.8 3.8 0 1 0 8 4.2z" fill="currentColor"/></svg>',
  };

  // Position helpers: simulation time → % of track; host fraction → % of track.
  function positions(m) {
    const ex = m.segs ? m.segs.exec : [0, 1];
    const host = m.segs ? m.segs.host : [0, 0];
    return {
      t: (v) => (ex[0] + (v / m.total) * (ex[1] - ex[0])) * 100,
      h: (f) => (host[0] + f * (host[1] - host[0])) * 100,
    };
  }

  function laneInner(lane, P) {
    if (lane.type === 'cells' || lane.type === 'index') {
      return lane.items.map((it) => {
        const host = it.seg === 'host';
        const l = host ? P.h(it.s) : P.t(it.s);
        const w = (host ? P.h(it.e) : P.t(it.e)) - l;
        const small = it.small ? `<small class="bind">${esc(it.small)}</small>` : '';
        const tooltip = lane.role === 'actor' || lane.role === 'hold' ? ` title="${esc(span(it.s, it.e))}" aria-label="${esc(span(it.s, it.e))}"` : '';
        return `<span class="cell"${tooltip}${it.alt ? ' data-alt' : ''}${it.currentCycle ? ' data-current-cycle' : ''}${host ? ' data-seg="host"' : ''} data-s="${it.s}" data-e="${it.e}" style="left:${l}%;width:calc(${w}% - 2px)"><b>${esc(it.text)}</b>${small}</span>`;
      }).join('');
    }
    return lane.items.map((it) => {
      const p = it.pre ? P.h(0) : P.t(it.t);
      const edge = p < 4 ? ' edge-l' : p > 96 ? ' edge-r' : '';
      const label = (it.name || it.text)
        ? `<span class="pt-label">${it.name ? `<b>${esc(it.name)}</b>` : ''}${it.text ? `<small>${esc(it.text)}</small>` : ''}</span>` : '';
      return `<span class="pt${edge}${it.ghost ? ' ghost' : ''}${it.infer ? ' infer' : ''}"${it.pre ? ' data-pre' : ''} data-t="${it.t}" style="left:${p}%"><i aria-hidden="true"></i>${label}</span>`;
    }).join('');
  }

  function bands(m, lane) {
    if (!m.segs) return '';
    const S = m.segs;
    let out = '';
    if (!lane || lane.role !== 'host') {
      out += `<span class="band band-pause" style="left:0;width:${S.host[1] * 100}%">${lane && lane.pauseLabel ? `<em>${esc(tr('仿真暂停', 'sim paused'))}</em>` : ''}</span>`;
    }
    out += `<span class="band band-brk" style="left:${S.brk[0] * 100}%;width:${(S.brk[1] - S.brk[0]) * 100}%" aria-hidden="true"></span>`;
    if (lane && lane.execNote) {
      out += `<span class="exec-note" style="left:${S.exec[0] * 100}%;width:${(S.exec[1] - S.exec[0]) * 100}%">${esc(lane.execNote)}</span>`;
    }
    return out;
  }

  function timelineHTML(m) {
    const P = positions(m);
    const rows = m.lanes.map((lane) => {
      const cls = ['track', `track--${lane.type}`];
      if (lane.kind) cls.push(`kind-${lane.kind}`);
      if (lane.role) cls.push(`role-${lane.role}`);
      if (lane.role === 'actor' || lane.role === 'hold') cls.push('role-targets');
      if (lane.tall) cls.push('tall');
      return `<div class="lane-label"><span>${esc(lane.label)}</span><small>${esc(lane.sub)}</small></div>`
        + `<div class="${cls.join(' ')}" data-key="${lane.key}">${bands(m, lane)}${laneInner(lane, P)}</div>`;
    }).join('');
    let minor = '';
    for (let v = 0; v <= m.total + EPS; v += m.minor) {
      const major = m.ticks.some((t) => Math.abs(t - v) < EPS);
      minor += `<i${major ? ' class="major"' : ''} style="left:${P.t(v)}%"></i>`;
    }
    const labels = m.ticks.map((v) => {
      const p = P.t(v);
      const edge = (m.segs && v === 0) || p < 1 ? ' edge-l' : p > 99 ? ' edge-r' : '';
      return `<span class="tick${edge}" style="left:${p}%">${ms(v)}</span>`;
    }).join('');
    const hostAxis = m.segs
      ? `<span class="tick host-tick" style="left:0;width:${m.segs.host[1] * 100}%">${esc(tr('主机等待 · 非等比例', 'host wait · not to scale'))}</span>`
      : '';
    const axisLabel = m.segs ? tr('时间：主机 ‖ 仿真', 'Time: host ‖ simulation') : tr('仿真时间', 'Simulation time');
    return `<div class="tl" role="img" aria-label="${esc(m.aria)}" style="--grid-n:${m.total / m.minor}">${rows}`
      + `<div class="lane-label axis-label"><span>${esc(axisLabel)}</span></div>`
      + `<div class="track track--axis">${bands(m, null)}${minor}${hostAxis}${labels}</div></div>`;
  }

  function grootFigure(m) {
    const boxes = Array.from({ length: 16 }, (_, i) => `<span class="out ${i < 4 ? 'keep' : 'drop'}">#${i + 1}</span>`).join('');
    return `<p class="gr-line"><b>${esc(tr('起始观察', 'Initial observation'))}</b>：`
      + `${esc(tr('RGB A + 历史画面 · 身体状态 · 文字指令', 'RGB A + history frames · body state · text instruction'))}`
      + ` · <span class="off">${esc(tr('ToF 仅显示，不输入', 'ToF displayed only, not an input'))}</span></p>`
      + `<p class="clocks"><span class="simclock" data-ref="simclock"></span> <span class="sep">·</span> ${tr('// 分隔主机与仿真时间，两侧不等比例、不相加。',
        '// separates host and simulation time; scales differ and times are not added.')}</p>`
      + timelineHTML(m)
      + `<div class="gr-outputs"><div class="outs-title">${esc(tr('预测 16 条请求，执行前 4 条', 'Predict 16 requests; execute the first 4'))}</div>`
      + `<div class="outs" data-ref="outs">${boxes}</div>`
      + `<small class="outs-legend"><span><i class="sw"></i>${esc(tr('#1–#4：依次执行', '#1–#4: executed in turn'))}</span><span><i class="sw drop"></i>${esc(tr('#5–#16：不执行，随后重新预测', '#5–#16: not executed; replan next'))}</span></small></div>`;
  }

  function walkingComputeHTML() {
    return `<div class="actor-compute" role="group" aria-label="${esc(tr('每个控制周期的主机计算顺序', 'Host computation in each control cycle'))}">
      <div class="actor-compute-heading">${esc(tr('每个周期开始：先完成主机计算', 'At each cycle start: finish host computation first'))}</div>
      <div class="actor-compute-flow">
        <span>${esc(tr('读取状态', 'Read state'))}</span><b aria-hidden="true">→</b>
        <span class="actor-compute-infer"><strong>Actor ${esc(tr('推理', 'inference'))}</strong><small>${esc(tr('耗时随设备变化 · 非零', 'Device-dependent, nonzero duration'))}</small></span><b aria-hidden="true">→</b>
        <span>${esc(tr('生成关节目标', 'Produce joint targets'))}</span>
      </div>
      <p>${esc(tr('计算过程不按时间比例绘制；仿真时钟等待推理完成，再推进下方的 20 ms。', 'Computation is not drawn to scale. The simulation clock waits for inference, then advances through the 20 ms below.'))}</p>
    </div>`;
  }

  function legendHTML() {
    const li = (cls, text) => `<li><span class="lg ${cls}" aria-hidden="true"></span>${esc(text)}</li>`;
    const ph = li('lg-ph', tr('当前时刻', 'Current time'));
    let items;
    if (cur === 'walking') {
      items = [li('lg-update', tr('推理完成，更新目标', 'Inference done; update targets')),
        li('lg-target', tr('关节目标 · 每格 20 ms', 'Joint targets · 20 ms per box')),
        li('lg-cell phys', tr('物理步 5 ms', 'Physics step 5 ms')), ph];
    } else if (cur === 'onboard') {
      items = [li('lg-diamond', tr('采集（瞬时事件）', 'Capture (instant event)')),
        li('lg-vinfer', tr('感知计算', 'Perception compute')),
        li('lg-update', tr('控制 / 推理', 'Control / inference')),
        li('lg-cell vis', tr('视觉结果沿用期', 'Vision result reused')),
        li('lg-cell req', tr('A / B / C：请求采用的视觉结果', 'A / B / C: vision result used')),
        li('lg-bound', tr('虚线框：仍在沿用的观察', 'Dashed: earlier observation still used')),
        li('lg-target', tr('关节目标 · 每格 20 ms', 'Joint targets · 20 ms per box')), ph];
    } else {
      items = [li('lg-diamond', tr('RGB 采集（瞬时）', 'RGB capture (instant)')),
        li('lg-infer', tr('快照作为新推理输入', 'Snapshot feeds new inference')),
        li('lg-ring', tr('整理观察，取缓存请求', 'Assemble, fetch cached request')),
        li('lg-host', tr('主机推理（非等比例）', 'Host inference (not to scale)')),
        li('lg-pause', tr('仿真暂停', 'Simulation paused')),
        li('lg-cell req', tr('请求保持 80 ms', 'Request held 80 ms')),
        li('lg-update', tr('反馈 + Actor 推理', 'Feedback + Actor inference')), li('lg-target', tr('关节目标 · 每格 20 ms', 'Joint targets · 20 ms per box')), ph];
    }
    return `<ul class="legend" aria-label="${esc(tr('图例', 'Legend'))}">${items.join('')}</ul>`;
  }

  function detailsHTML() {
    let items;
    if (cur === 'walking') items = [
      tr('Actor 根据关节与 IMU 反馈输出 14 个关节目标，每 20 ms 更新一次。目标在四个 5 ms 物理步内保持。', 'The Actor uses joint and IMU feedback to output 14 joint targets every 20 ms, held across four 5 ms physics steps.'),
      tr('Actor 推理需要主机时间，具体耗时取决于设备和推理实现，尚未单独测量。上方方框表示计算顺序与非零耗时，宽度不对应毫秒；仿真等待输出后才推进。下方 20 ms 是仿真中的目标保持间隔，不是推理耗时。', 'Actor inference takes host time depending on the device and inference implementation; it has not been timed separately here. The boxes above show computation order and nonzero duration, not milliseconds to scale. Simulation advances after the output is ready. The 20 ms below is the simulated target-hold interval, not inference latency.')
    ];
    else if (cur === 'onboard') items = [
      tr('新采集后运行 YOLO、位置估计和依赖图像的阶段判断。每 20 ms，反馈控制结合最近视觉结果与新的身体反馈，更新运动请求，例如前进并减速；Actor 再将请求转成关节目标。', 'New captures trigger YOLO, localization and image-dependent stage decisions. Every 20 ms, feedback control uses the latest vision and fresh body feedback to update a motion request, such as moving forward while slowing down; the Actor turns it into joint targets.'),
      tr('反馈控制包括 PD 纠偏、制动与交接检查。阶段逻辑跨越感知和控制两种节奏，不能整体称为 15 Hz 或 50 Hz。', 'Feedback control includes PD correction, braking and handoff checks. Stage logic spans perception and control cadences; it is not uniformly 15 Hz or 50 Hz.'),
      tr('A、B、C 按名义 15 Hz 采集。66.7 ms 的新观察 B，到 80 ms 的控制更新才被采用；中间继续使用 A。', 'A, B and C use nominal 15 Hz capture times. Observation B arrives at 66.7 ms and is adopted at the 80 ms control update; A is used in between.'),
      tr('点表示计算事件，方格表示输出保持时间。计算并非零耗时或并行执行；仿真等待计算完成，50 Hz 仅表示配置的更新频率。', 'Points mark computation; blocks show output hold time. Computation is neither zero-cost nor concurrent: simulation waits for it. 50 Hz is the configured update rate.')
    ];
    else items = [
      tr('GR00T 一次预测 16 条 8 维高层请求，例如前进、转向、看球或踢球；先执行前 4 条。每条保持 80 ms，底层 Actor 用新的身体反馈更新四次，生成 14 个关节目标。', 'GR00T predicts sixteen 8-D high-level requests, such as moving, turning, looking at the ball or kicking. The first four execute for 80 ms each; the low-level Actor uses fresh body feedback four times per request to generate 14 joint targets.'),
      tr('约 300 ms 是 DGX Spark 服务端新推理的主机耗时中位数。20/80 ms 是仿真周期，两者分轴、不相加。等待期间仿真暂停，第一条请求使用推理前的同一份观察。', 'About 300 ms is the median host compute time for new inference on the DGX Spark server. 20/80 ms are simulation intervals: separate clocks, never added. Simulation pauses while waiting; request 1 uses the same pre-inference observation.'),
      tr('80、160、240 ms 整理观察并同步取回缓存请求，不重新推理。图中省略通信与解码等待；这些等待也会暂停仿真。320 ms 再用新观察预测。', 'At 80, 160 and 240 ms, observations are assembled and cached requests fetched synchronously, without new inference. Communication and decoding waits are omitted; they also pause simulation. A new prediction starts from the 320 ms observation.'),
      tr('RGB 按名义 15 Hz 采集；执行中的新图像进入历史，不改变当前缓存请求。ToF 只显示，不输入 GR00T。当前流程未使用 RTC。', 'RGB captures use nominal 15 Hz timing. New images enter history without changing cached requests. ToF is displayed only, not fed to GR00T. This flow does not use RTC.')
    ];
    return `<details><summary>${esc(tr('详解', 'Details'))}</summary><ul>${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul></details>`;
  }

  function headText() {
    if (cur === 'walking') {
      return {
        eyebrow: tr('2.2 行走', '2.2 Walking'),
        title: tr('控制周期 · 行走', 'Control cycle · walking'),
        summary: tr('每 20 ms 读取身体反馈、推理新动作，再保持关节目标。',
          'Every 20 ms, read body feedback, infer a new action, then hold the joint targets.'),
      };
    }
    if (cur === 'onboard') {
      return {
        eyebrow: tr('4.3 板载策略', '4.3 Onboard policy'),
        title: tr('控制周期 · 板载策略', 'Control cycle · onboard policy'),
        summary: tr('感知提供球位；反馈控制生成运动请求，底层 Actor 再推理出关节目标。',
          'Perception locates the ball; feedback control generates motion requests, then the low-level Actor infers joint targets.'),
      };
    }
    return {
      eyebrow: tr('5.2 GR00T', '5.2 GR00T'),
      title: tr('控制周期 · GR00T 分层执行', 'Control cycle · GR00T hierarchy'),
      summary: tr('先观察与推理，再执行前 4 条请求；随后重新观察，进入下一轮。',
        'Observe and infer, execute the first four requests, then observe again for the next round.'),
    };
  }

  /* ------------------------------------------------------------------ build */
  function build(focusSel) {
    model = buildModel(cur);
    const root = host;
    root.lang = lang === 'zh' ? 'zh-CN' : 'en';
    root.dataset.theme = theme;
    const h = headText();
    const scnLabels = { walking: tr('行走 · 2.2', 'Walking · 2.2'), onboard: tr('板载策略 · 4.3', 'Onboard · 4.3'), groot: 'GR00T · 5.2' };
    const isG = cur === 'groot';
    const app = shadow.getElementById('app');
    app.innerHTML = `
      <div class="bar" ${SCENARIOS.length===1?'hidden':''}>
        <div class="seg" role="group" aria-label="${esc(tr('选择控制周期', 'Choose a control cycle'))}">
          ${SCENARIOS.map((s) => `<button type="button" data-scn="${s}" aria-pressed="${s === cur}">${esc(scnLabels[s])}</button>`).join('')}
        </div>
      </div>
      <header>
        <h3>${esc(h.title)}</h3>
        <p class="summary">${esc(h.summary)}</p>
        <p class="note">${esc(cur === 'walking' ? tr('下方每格代表一组 20 ms 关节目标；频率为仿真配置，播放为教学慢放。', 'Each timeline box represents 20 ms of joint targets. Rates are simulation settings; playback is teaching slow motion.') : tr('每格代表一组 20 ms 关节目标，点标采集或计算；频率为仿真配置，播放为教学慢放。', 'Each box represents 20 ms of joint targets; points mark captures or computation. Rates are simulation settings; playback is teaching slow motion.'))}</p>
      </header>
      <figure class="diagram" data-scenario="${cur}" data-ref="fig">
        ${isG ? grootFigure(model) : (cur === 'walking' ? walkingComputeHTML() : '') + timelineHTML(model)}
        ${legendHTML()}
      </figure>
      <div class="controls" role="group" aria-label="${esc(tr('播放控制', 'Playback controls'))}">
        <button type="button" class="btn primary" data-act="play"></button>
        <div class="scrub">
          <label for="cyc-range">${esc(isG ? tr('执行段仿真时间', 'Execution sim time') : tr('仿真时间', 'Simulation time'))}</label>
          <input id="cyc-range" type="range" min="0" max="${model.total}" step="any" value="0">
          <output for="cyc-range" data-ref="out"></output>
        </div>
        <span class="status" data-ref="status"></span>
      </div>
      <section class="now" aria-labelledby="now-h" data-ref="now">
        <h2 id="now-h" data-ref="nowTitle"></h2>
        <ul data-ref="nowList"></ul>
      </section>
      ${detailsHTML()}
`;

    const ref = (n) => app.querySelector(`[data-ref="${n}"]`);
    dom = {
      app,
      fig: ref('fig'),
      out: ref('out'),
      status: ref('status'),
      now: ref('now'),
      nowTitle: ref('nowTitle'),
      nowList: ref('nowList'),
      range: app.querySelector('#cyc-range'),
      play: app.querySelector('[data-act="play"]'),
      cells: Array.from(app.querySelectorAll('.tl .cell')),
      ptLanes: Array.from(app.querySelectorAll('.tl .track--points')).map((el) => ({ el, key: el.dataset.key, pts: Array.from(el.querySelectorAll('.pt')) })),
      lane: {},
    };
    app.querySelectorAll('.tl .track').forEach((el) => {
      if (el.dataset.key) dom.lane[el.dataset.key] = { el, cells: Array.from(el.querySelectorAll('.cell')), pts: Array.from(el.querySelectorAll('.pt')) };
    });
    if (isG) {
      dom.outs = ref('outs');
      dom.outBoxes = Array.from(dom.outs.querySelectorAll('.out'));
      dom.simclock = ref('simclock');
    }
    lastKey = '';
    update(true);
    if (focusSel) { const f = app.querySelector(focusSel); if (f) f.focus(); }
  }

  /* ------------------------------------------------------------ explanation */
  function stageOf(st) {
    if (cur !== 'groot') return null;
    if (st.phase === 'inference') return st.host <= EPS && st.hold === 0 ? 1 : 2;
    return st.ended ? 4 : 3;
  }

  function explain() {
    const st = store[cur], t = st.time;
    if (cur === 'walking') {
      if (st.ended) return {key:'w-end',title:tr('已完成 · 60 ms','Completed · 60 ms'),lines:[tr('3 个控制周期，共 12 个物理步。','3 control cycles, 12 physics steps.')]};
      const k=Math.min(2,Math.floor(t/20+EPS)),p=Math.min(3,Math.floor((t-k*20)/5+EPS)),a=k*20+p*5;
      return {key:`w${k}-${p}`,title:tr(`控制周期 ${WALK_CYCLES[k]} · ${span(k*20,k*20+20)}`,`Control cycle ${WALK_CYCLES[k]} · ${span(k*20,k*20+20)}`),lines:[
        tr(`${ms(k*20)}：Actor 根据身体反馈生成 14 个关节目标。`,`${ms(k*20)}: Actor uses body feedback to generate 14 joint targets.`),
        tr(`保持目标；当前物理步 ${p+1}/4（${span(a,a+5)}）。`,`Hold targets; physics step ${p+1}/4 (${span(a,a+5)}).`)]};
    }
    if (cur === 'onboard') {
      if(st.ended)return {key:'o-end',title:tr('已完成 · 200 ms','Completed · 200 ms'),lines:[tr('3 次感知更新，10 次控制与 Actor 更新；下一窗口继续。','3 perception updates, 10 control and Actor updates; continue in the next window.')]};
      const k=Math.min(9,Math.floor(t/20+EPS)),kt=k*20,live=latest(CAP_ON,t),bound=latest(CAP_ON,kt);
      const use=live===bound
        ?tr(`本次请求采用观察 ${L[bound]} 与 ${ms(kt)} 的身体反馈。`,`This request uses observation ${L[bound]} and body feedback from ${ms(kt)}.`)
        :tr(`本次仍用观察 ${L[bound]}；新观察 ${L[live]} 在 ${ms(Math.ceil(CAP_ON[live]/20-EPS)*20)} 更新时采用。`,`Still using observation ${L[bound]}; new observation ${L[live]} is adopted at ${ms(Math.ceil(CAP_ON[live]/20-EPS)*20)}.`);
      return {key:`o${k}-${live}`,title:tr(`控制更新 #${k+1} · ${span(kt,kt+20)}`,`Control update #${k+1} · ${span(kt,kt+20)}`),lines:[
        tr(`最新感知：${L[live]}（${ms(CAP_ON[live])} 采集）。`,`Latest perception: ${L[live]} (captured at ${ms(CAP_ON[live])}).`),use,
        tr(`Actor 生成关节目标，保持到 ${ms(kt+20)}。`,`Actor generates joint targets, held until ${ms(kt+20)}.`)]};
    }
    if(st.phase==='inference'){
      if(st.host<=EPS&&st.hold===0)return {key:'g-obs',title:tr('起始观察 · 仿真 0 ms','Initial observation · sim 0 ms'),lines:[tr('整理 RGB A、历史画面、身体状态和指令，送入 GR00T。','Assemble RGB A, history frames, body state and instruction for GR00T.')]};
      if(st.host<1-EPS)return {key:'g-inf',title:tr('GR00T 推理中','GR00T inference'),lines:[tr('仿真停在 0 ms，等待新预测。','Simulation remains at 0 ms while waiting for a new prediction.'),tr('≈ 300 ms：DGX Spark 服务端推理中位数。','≈ 300 ms: median inference on the DGX Spark server.')]};
      return {key:'g-done',title:tr('推理完成 · 16 条请求','Inference complete · 16 requests'),lines:[tr('执行前 4 条：4 × 80 ms = 320 ms 仿真时间。','Execute the first 4: 4 × 80 ms = 320 ms of simulation time.')]};
    }
    if(st.ended)return {key:'g-end',title:tr('本段结束 · 320 ms','Segment complete · 320 ms'),lines:[tr('前 4 条已执行，其余不执行；整理新观察，再次推理。','The first 4 requests are complete; discard the rest, assemble a new observation and infer again.')]};
    const j=Math.min(3,Math.floor(t/80+EPS)),u=Math.min(15,Math.floor(t/20+EPS)),m=u-j*4,live=latest(CAP_GR,t),a=j*80;
    return {key:`g${u}-${live}`,title:tr(`请求 #${j+1} · ${span(a,a+80)}`,`Request #${j+1} · ${span(a,a+80)}`),lines:[
      j===0?tr('使用起始观察的预测，不重复采样。','Use the initial prediction; no repeated sampling.'):tr(`${ms(a)}：整理观察，取回缓存请求，不重新推理。`,`${ms(a)}: assemble observation and fetch the cached request, without new inference.`),
      tr(`底层 Actor 第 ${m+1}/4 次更新，目标保持到 ${ms(u*20+20)}。`,`Low-level Actor update ${m+1}/4; targets held until ${ms(u*20+20)}.`),
      tr(`最新 RGB：${L[live]}；当前请求保持不变。`,`Latest RGB: ${L[live]}; the current request stays unchanged.`)]};
  }

  /* ----------------------------------------------------------------- update */
  function ord(st) { return cur === 'groot' && st.phase === 'inference' ? st.host - 2 : st.time; }
  function atStart(st) { return cur === 'groot' ? (st.phase === 'inference' && st.host <= EPS && st.hold === 0) : st.time <= EPS; }

  function statusText(st) {
    if (st.ended) return tr('已完成', 'Completed');
    if (playing) {
      if (mqReduce.matches) return tr('逐步播放（减少动态效果）', 'Step-by-step playback (reduced motion)');
      if (cur === 'groot' && st.phase === 'inference') return tr('播放中 · 主机推理（进度仅示意）', 'Playing · host inference (progress illustrative)');
      const f = Math.round(PLAY_MS[cur] / TOTAL[cur]);
      return tr(`播放中 · 教学慢放（约慢 ${f} 倍）`, `Playing · teaching slow motion (~${f}× slower)`);
    }
    return atStart(st) ? tr('就绪', 'Ready') : tr('已暂停', 'Paused');
  }

  function update(force) {
    const st = store[cur];
    const total = TOTAL[cur];
    const t = st.time;
    const inf = cur === 'groot' && st.phase === 'inference';
    const execOn = !inf;

    let ph;
    if (cur === 'groot') {
      const S = SEGS;
      ph = inf ? (S.host[0] + Math.min(1, st.host) * (S.host[1] - S.host[0])) * 100
        : (S.exec[0] + (t / total) * (S.exec[1] - S.exec[0])) * 100;
    } else {
      ph = (t / total) * 100;
    }
    dom.fig.style.setProperty('--ph', `${ph}%`);
    dom.fig.classList.toggle('is-ended', st.ended);
    dom.fig.classList.toggle('is-inference', inf);
    dom.now.classList.toggle('is-ended', st.ended);

    dom.cells.forEach((c) => {
      if (c.dataset.seg === 'host') {
        c.classList.toggle('is-active', inf);
        c.classList.toggle('is-past', !inf);
        return;
      }
      const s = +c.dataset.s;
      const e = +c.dataset.e;
      const act = execOn && !st.ended && s <= t + EPS && t < e - EPS;
      const past = execOn && (st.ended || e <= t + EPS);
      c.classList.toggle('is-active', act);
      c.classList.toggle('is-past', past && !act);
    });
    dom.ptLanes.forEach((lane) => {
      let liveIdx = -1;
      lane.pts.forEach((p, i) => {
        const pre = p.hasAttribute('data-pre');
        const past = pre || (execOn && +p.dataset.t <= t + EPS);
        p.classList.toggle('is-past', past);
        if (past) liveIdx = i;
      });
      lane.pts.forEach((p, i) => p.classList.toggle('is-live', i === liveIdx));
    });

    if (cur === 'onboard') {
      const k = Math.min(9, Math.floor(t / 20 + EPS));
      const live = latest(CAP_ON, t);
      const b = st.ended ? -1 : latest(CAP_ON, k * 20);
      const marked = b !== live ? b : -1;
      dom.lane.cap.pts.forEach((p, i) => p.classList.toggle('is-bound', i === marked));
      dom.lane.avail.cells.forEach((c, i) => c.classList.toggle('is-bound-src', i === marked));
    }

    if (cur === 'groot') {
      dom.outs.classList.toggle('ready', !inf || st.host >= 1 - EPS);
      // Only #1–#4 can ever be active; at the end they are all completed and none is active.
      const j = inf || st.ended ? -1 : Math.min(3, Math.floor(t / 80 + EPS));
      dom.outBoxes.forEach((b, i) => {
        b.classList.toggle('is-active', i < 4 && i === j);
        b.classList.toggle('is-past', !inf && i < 4 && (st.ended || i < j));
      });
      dom.simclock.innerHTML = inf
        ? tr('仿真时钟 <b>0 ms</b> · 暂停（等待主机推理）', 'Sim clock <b>0 ms</b> · paused (waiting for host inference)')
        : tr(`仿真时钟 <b>${ms(t)}</b>`, `Sim clock <b>${ms(t)}</b>`);
    }

    // controls
    dom.range.value = String(t);
    dom.out.textContent = inf ? tr('推理中 · 0 ms', 'Inferring · 0 ms') : ms(t);
    dom.range.setAttribute('aria-valuetext', inf
      ? tr('推理阶段，仿真暂停在 0 毫秒', 'Inference stage, simulation paused at 0 milliseconds')
      : tr(`仿真时间 ${num(t)} 毫秒`, `Simulation time ${num(t)} milliseconds`));
    let label; let icon;
    if (playing) { label = tr('暂停', 'Pause'); icon = ICON.pause; }
    else if (st.ended) { label = tr('重新播放', 'Replay'); icon = ICON.replay; }
    else if (atStart(st)) { label = tr('播放', 'Play'); icon = ICON.play; }
    else { label = tr('继续', 'Resume'); icon = ICON.play; }
    if (dom.play.dataset.label !== label) { dom.play.innerHTML = `${icon}<span>${esc(label)}</span>`; dom.play.dataset.label = label; }
    const s = statusText(st);
    if (dom.status.textContent !== s) dom.status.textContent = s;

    const ex = explain();
    if (force || ex.key !== lastKey) {
      lastKey = ex.key;
      dom.nowTitle.textContent = ex.title;
      dom.nowList.innerHTML = ex.lines.map((x) => `<li>${esc(x)}</li>`).join('');
    }
  }

  function announce(msg) {
    const el = shadow.getElementById('announce');
    if (!el) return;
    el.textContent = '';
    window.setTimeout(() => { el.textContent = msg; }, 30);
  }
  function announceNow(prefix) {
    const ex = explain();
    announce((prefix ? prefix + ' ' : '') + ex.title + (lang === 'zh' ? '。' : '. ') + ex.lines[0]);
  }

  /* --------------------------------------------------------------- playback */
  function keyframes() {
    if (cur === 'walking') return steps(0, 60, 5).map((time) => ({ phase: 'cycle', time }));
    if (cur === 'onboard') return uniq(steps(0, 200, 20).concat(CAP_ON)).map((time) => ({ phase: 'cycle', time }));
    return [{ phase: 'inference', host: 0 }, { phase: 'inference', host: 0.5 }, { phase: 'inference', host: 1 }]
      .concat(uniq(steps(0, 320, 20).concat(CAP_GR)).map((time) => ({ phase: 'execution', time })));
  }
  const kord = (k) => (k.phase === 'inference' ? k.host - 2 : k.time);

  function applyKey(k) {
    const st = store[cur];
    st.phase = k.phase;
    st.host = k.phase === 'inference' ? k.host : 1;
    st.hold = 0;
    st.time = k.phase === 'inference' ? 0 : k.time;
    st.ended = k.phase !== 'inference' && st.time >= TOTAL[cur] - EPS;
  }

  function stepCore(dir) {
    const st = store[cur];
    const o = ord(st);
    const ks = keyframes();
    let target = null;
    if (dir > 0) target = ks.find((k) => kord(k) > o + EPS) || null;
    else for (let i = ks.length - 1; i >= 0; i -= 1) if (kord(ks[i]) < o - EPS) { target = ks[i]; break; }
    if (!target) return false;
    applyKey(target);
    return true;
  }

  function stopLoops() {
    if (raf) cancelAnimationFrame(raf);
    if (timer) clearInterval(timer);
    raf = 0; timer = 0; lastTs = null;
  }

  function finish() {
    playing = false;
    stopLoops();
    update();
    announceNow(tr('已完成：', 'Completed:'));
  }

  function advance(dt) {
    const st = store[cur];
    const total = TOTAL[cur];
    if (cur === 'groot' && st.phase === 'inference') {
      if (st.host < 1) { st.host = Math.min(1, st.host + dt / HOST_PLAY_MS); return; }
      st.hold += dt;
      if (st.hold >= HOST_HOLD_MS) { st.phase = 'execution'; st.time = 0; st.hold = 0; }
      return;
    }
    st.time += (dt * total) / PLAY_MS[cur];
    if (st.time >= total - EPS) { st.time = total; st.ended = true; }
  }

  function frame(ts) {
    if (!playing) return;
    if (lastTs === null) lastTs = ts;
    const dt = Math.min(ts - lastTs, 100);
    lastTs = ts;
    advance(dt);
    if (store[cur].ended) { finish(); return; }
    update();
    raf = requestAnimationFrame(frame);
  }

  function play() {
    const st = store[cur];
    if (playing) return;
    if (st.ended) Object.assign(st, fresh(cur));
    playing = true;
    lastTs = null;
    if (mqReduce.matches) {
      timer = window.setInterval(() => {
        if (!stepCore(1) || store[cur].ended) { finish(); return; }
        update();
      }, STEP_MS_REDUCED);
    } else {
      raf = requestAnimationFrame(frame);
    }
    update();
  }

  function pause(say) {
    if (!playing) return;
    playing = false;
    stopLoops();
    update();
    if (say) announceNow(tr('已暂停：', 'Paused:'));
  }

  function reset() {
    pause(false);
    Object.assign(store[cur], fresh(cur));
    update(true);
    announceNow(tr('已重置：', 'Reset:'));
  }

  function step(dir) {
    pause(false);
    if (stepCore(dir)) { update(); announceNow(); }
  }

  function seek(v, say) {
    pause(false);
    const st = store[cur];
    const total = TOTAL[cur];
    const t = Math.max(0, Math.min(total, Number(v) || 0));
    if (cur === 'groot') { st.phase = 'execution'; st.host = 1; st.hold = 0; }
    st.time = t;
    st.ended = t >= total - EPS;
    update();
    if (say) announceNow();
  }

  /* ------------------------------------------------------------ public API */
  function setScenario(name, focus) {
    if (!SCENARIOS.includes(name)) return;
    pause(false);
    cur = name;
    build(focus ? `[data-scn="${name}"]` : null);
  }
  function setLanguage(l, focus) {
    if (l !== 'zh' && l !== 'en') return;
    lang = l;
    build(focus ? '[data-act="lang"]' : null);
  }
  function setTheme(th) {
    if (th !== 'light' && th !== 'dark') return;
    theme = th;
    host.dataset.theme = theme;
    const b = shadow.querySelector('[data-act="theme"]');
    if (b) b.setAttribute('aria-pressed', String(theme === 'dark'));
  }
  function getState() {
    const st = store[cur];
    const inf = cur === 'groot' && st.phase === 'inference';
    return {
      scenario: cur,
      language: lang,
      theme,
      playing,
      phase: cur === 'groot' ? st.phase : 'cycle',
      time: inf ? 0 : st.time,
      total: TOTAL[cur],
      ended: st.ended,
      stage: stageOf(st),
      hostProgress: cur === 'groot' ? Math.min(1, st.host) : null,
    };
  }

  host.cyclePreview = {
    setScenario(n) { setScenario(n, false); return getState(); },
    setLanguage(l) { setLanguage(l, false); return getState(); },
    setTheme(th) { setTheme(th); return getState(); },
    getState,
    play() { play(); return getState(); },
    pause() { pause(false); return getState(); },
    reset() { reset(); return getState(); },
    seek(v) { seek(v, false); return getState(); },
    step(d) { step(d < 0 ? -1 : 1); return getState(); },
  };

  /* ----------------------------------------------------------------- events */
  function bind() {
    const app = shadow.getElementById('app');
    app.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b || !app.contains(b)) return;
      if (b.dataset.scn) { setScenario(b.dataset.scn, true); return; }
      if (b.getAttribute('aria-disabled') === 'true') return;
      switch (b.dataset.act) {
        case 'play': if (playing) pause(true); else play(); break;
        case 'lang': setLanguage(lang === 'zh' ? 'en' : 'zh', true); break;
        case 'theme': setTheme(theme === 'dark' ? 'light' : 'dark'); break;
        default: break;
      }
    });
    app.addEventListener('input', (e) => { if (e.target.id === 'cyc-range') seek(e.target.value, false); });
    app.addEventListener('change', (e) => { if (e.target.id === 'cyc-range') announceNow(); });
    app.addEventListener('keydown', (e) => {
      if (e.target.id !== 'cyc-range') return;
      const total = TOTAL[cur];
      const st = store[cur];
      const base = cur === 'groot' && st.phase === 'inference' ? 0 : st.time;
      const big = e.shiftKey ? 10 : 1;
      let v = null;
      switch (e.key) {
        case 'ArrowLeft': case 'ArrowDown': v = base - big; break;
        case 'ArrowRight': case 'ArrowUp': v = base + big; break;
        case 'PageDown': v = base - 20; break;
        case 'PageUp': v = base + 20; break;
        case 'Home': v = 0; break;
        case 'End': v = total; break;
        default: return;
      }
      e.preventDefault();
      seek(v, true);
    });
  }

  /* ------------------------------------------------------------------- init */
  const syncTheme = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');
  syncTheme();
  bind();
  build(null);
  new MutationObserver(syncTheme).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(false); });
  host.dataset.ready = 'true';
});
