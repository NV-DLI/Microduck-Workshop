import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// Same easing as the Blender scene the layout was exported from (sine in/out).
const ease = (u) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)));
const HIGHLIGHT = new THREE.Color(0x76b900);

// Interpolate [t, ...values] keys with the shared easing per segment.
function sample(keys, t) {
  if (!keys?.length) return null;
  if (t <= keys[0][0]) return keys[0].slice(1);
  for (let i = 1; i < keys.length; i++) {
    const a = keys[i - 1],
      b = keys[i];
    if (t <= b[0]) {
      const u = ease((t - a[0]) / (b[0] - a[0] || 1));
      return a.slice(1).map((v, k) => v + (b[k + 1] - v) * u);
    }
  }
  return keys[keys.length - 1].slice(1);
}

// onProgress(fraction) is optional and receives model download progress (0–1).
export async function mountRobot(root, lang, onProgress) {
  const zh = lang === "zh",
    text = (o) => (zh ? o.zh ?? o.label_zh : o.en ?? o.label_en),
    stage = root.querySelector(".viewer-stage"),
    status = root.querySelector(".viewer-status"),
    info = root.querySelector(".component-info"),
    modes = root.querySelector(".anatomy-modes"),
    explodeControl = root.querySelector(".explode-control"),
    explodeInput = root.querySelector(".explode"),
    assembly = root.querySelector(".assembly-controls"),
    playButton = root.querySelector(".assembly-play"),
    replayButton = root.querySelector(".assembly-replay"),
    scrub = root.querySelector(".assembly-scrub"),
    clock = root.querySelector(".assembly-time"),
    panel = root.querySelector(".assembly-panel"),
    stepsList = root.querySelector(".assembly-steps");
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.05;
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(20, 1, 0.005, 20),
    controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
  controls.maxDistance = 4;
  controls.minDistance = 0.03;
  // Soft product lighting: an image-based studio plus a key and two rims so dark
  // servos keep a readable silhouette on both page themes. No floor or backdrop.
  const pmrem = new THREE.PMREMGenerator(renderer),
    room = new RoomEnvironment(),
    environment = pmrem.fromScene(room, 0.03);
  scene.environment = environment.texture;
  room.dispose();
  pmrem.dispose();
  const hemisphere = new THREE.HemisphereLight(0xf4f7ff, 0x9a9184, 0.9);
  scene.add(hemisphere);
  // Camera-relative rig: key, fill, and two rims behind the model.
  const rig = new THREE.Group(),
    lights = [
      [0xfff6ea, 1.7, 1.7, [1.2, 1.6, 0.9]],
      [0xffffff, 0.8, 0.9, [-0.6, 0.4, 1.4]],
      [0xdde8ff, 1.1, 2.6, [-1.4, 0.9, -1.0]],
      [0xe8f0ff, 0.5, 2.2, [1.4, 0.3, -1.1]],
    ].map(([color, light, dark, pos]) => {
      const l = new THREE.DirectionalLight(color, light);
      l.position.set(...pos);
      l.userData.power = { light, dark };
      rig.add(l);
      return l;
    });
  scene.add(rig);
  // On the dark page theme, stronger rims and ambient keep black servos readable.
  const applyTheme = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    for (const l of lights) l.intensity = l.userData.power[dark ? "dark" : "light"];
    scene.environmentIntensity = dark ? 1.15 : 0.85;
    hemisphere.intensity = dark ? 1.25 : 0.9;
    edgeMaterial.opacity = dark ? 0.3 : 0.42;
  };
  const url = (p) => new URL(p, import.meta.url).href,
    json = (p) =>
      fetch(url(p)).then((r) => {
        if (!r.ok) throw Error(`Missing ${p}`);
        return r.json();
      });
  let loaded;
  try {
    loaded = await Promise.all([
      new GLTFLoader().loadAsync(url("./robot/microduck.glb"), (event) => {
        if (event.lengthComputable) onProgress?.(event.loaded / event.total);
      }),
      json("./robot/components.json"),
      json("./robot/anatomy-layout.json"),
    ]);
  } catch (error) {
    // Release the WebGL context so a retry starts clean.
    controls.dispose();
    environment.dispose();
    renderer.dispose();
    renderer.forceContextLoss();
    throw error;
  }
  const [{ scene: model }, meta, layout] = loaded;
  scene.add(model);
  model.updateMatrixWorld(true);
  const descriptions = new Map(meta.parts.map((p) => [p.id, p])),
    parts = [],
    byId = new Map();
  // Illustration look: semi-matte CAD materials plus restrained crease lines,
  // computed once per shared mesh (38 unique meshes for 70 nodes).
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x1b1d21, transparent: true, opacity: 0.42 }),
    edgeCache = new Map(),
    meshes = [];
  model.traverse((o) => o.isMesh && meshes.push(o));
  for (const o of meshes) {
    o.material = o.material.clone();
    o.material.roughness = Math.max(o.material.roughness ?? 0.5, 0.5);
    o.material.metalness = (o.material.metalness ?? 0) * 0.55;
    let edges = edgeCache.get(o.geometry.uuid);
    if (!edges) {
      edges = new THREE.EdgesGeometry(o.geometry, 38);
      edgeCache.set(o.geometry.uuid, edges);
    }
    const lines = new THREE.LineSegments(edges, edgeMaterial);
    lines.raycast = () => {};
    o.add(lines);
  }
  for (const o of meshes) {
    const data = descriptions.get(o.userData.id) || o.userData;
    o.userData = { ...o.userData, ...data };
    const part = {
      object: o,
      id: o.userData.id,
      position: o.position.clone(),
      quaternion: o.quaternion.clone(),
      material: o.material,
      emissive: o.material.emissive?.clone() || new THREE.Color(0),
      offset: new THREE.Vector3(...(layout.overview.explode[o.userData.id] || [0, 0, 0])),
      box: new THREE.Box3(),
    };
    parts.push(part);
    byId.set(part.id, part);
  }
  const demos = layout.demos;
  let mode = "overview",
    current = null,
    amount = 1,
    demo = null,
    time = 0,
    playing = false,
    last = 0,
    stepIndex = -1,
    boxesDirty = true;

  // ------------------------------------------------------------ camera framing
  // Fit a box seen from a direction, keeping a share of the width free (labels).
  function fit(box, direction, widthShare = 0.9, heightShare = 0.88) {
    const c = box.getCenter(new THREE.Vector3()),
      d = new THREE.Vector3(...direction).normalize(),
      right = new THREE.Vector3(0, 1, 0).cross(d).normalize(),
      up = new THREE.Vector3().crossVectors(d, right),
      tv = Math.tan((camera.fov * Math.PI) / 360),
      th = tv * camera.aspect;
    let distance = 0.05;
    for (const x of [box.min.x, box.max.x])
      for (const y of [box.min.y, box.max.y])
        for (const z of [box.min.z, box.max.z]) {
          const v = new THREE.Vector3(x, y, z).sub(c);
          distance = Math.max(
            distance,
            v.dot(d) + Math.max(Math.abs(v.dot(right)) / (th * widthShare), Math.abs(v.dot(up)) / (tv * heightShare)),
          );
        }
    camera.position.copy(c).addScaledVector(d, distance);
    controls.target.copy(c);
    camera.near = distance / 50;
    camera.far = distance * 10;
    camera.updateProjectionMatrix();
    controls.update();
  }
  const visibleBox = () => {
    model.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const p of parts) if (p.object.visible) box.union(p.box.setFromObject(p.object));
    return box;
  };
  // Frame what is on screen now. The default direction is used when a view is
  // (re)opened; resizing and the slider keep the learner's own viewing direction.
  function frameCurrent(defaultView = false) {
    if (current) {
      fit(new THREE.Box3().setFromObject(current.object), cameraDirection(), 0.6, 0.6);
    } else if (mode === "overview") {
      // Leave room at the sides for the callouts as the layout opens (continuous, no jump).
      const share = stage.clientWidth > 700 ? 0.85 - 0.27 * Math.min(1, amount / 0.5) : 0.85;
      fit(visibleBox(), defaultView ? layout.overview.view : cameraDirection(), share, 0.96);
    } else {
      const keep = time,
        box = new THREE.Box3();
      for (const t of [0, demo.steps[0].start, demo.steps[0].end, demo.motion.start, (demo.motion.start + demo.motion.end) / 2, demo.duration]) {
        animate(t);
        box.union(visibleBox());
      }
      animate(keep);
      fit(box, defaultView ? demo.view : cameraDirection(), 0.8, 0.84);
    }
  }
  const cameraDirection = () => camera.position.clone().sub(controls.target).normalize().toArray();

  // ------------------------------------------------------------ poses
  // Overview pose; the slider value is remembered while a demonstration is open.
  function pose(value) {
    amount = value;
    for (const p of parts) {
      p.object.position.copy(p.position).addScaledVector(p.offset, value);
      p.object.quaternion.copy(p.quaternion);
    }
    boxesDirty = true;
  }
  function restPose() {
    for (const p of parts) {
      p.object.position.copy(p.position);
      p.object.quaternion.copy(p.quaternion);
    }
    boxesDirty = true;
  }
  const jointMatrix = (joint, angle) => {
    const pivot = new THREE.Vector3(...joint.pivot);
    return new THREE.Matrix4()
      .makeTranslation(pivot.x, pivot.y, pivot.z)
      .multiply(new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(...joint.axis).normalize(), angle))
      .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
  };
  // Translations follow the exported tracks; during the motion window every joint
  // rotates its downstream CAD bodies about the recorded pivot and axis, applied in
  // chain order (the same product of rotations the Blender scene was baked with).
  function animate(t) {
    time = Math.min(Math.max(t, 0), demo.duration);
    const m = demo.motion,
      inMotion = time >= m.start && time <= m.end,
      angles = inMotion ? sample(m.keys, time - m.start) : null,
      joints = inMotion ? m.joints.map((j, i) => [j, jointMatrix(j, angles[i])]) : [];
    for (const id of demo.parts) {
      const p = byId.get(id),
        offset = sample(demo.tracks[id], time) || [0, 0, 0];
      p.object.position.copy(p.position).add(new THREE.Vector3(...offset));
      p.object.quaternion.copy(p.quaternion);
      if (inMotion) {
        const world = new THREE.Matrix4();
        for (const [j, matrix] of joints) if (j.parts.includes(id)) world.multiply(matrix);
        world.multiply(new THREE.Matrix4().compose(p.object.position, p.object.quaternion, p.object.scale));
        world.decompose(p.object.position, p.object.quaternion, new THREE.Vector3());
      }
    }
    boxesDirty = true;
  }

  // ------------------------------------------------------------ highlight
  function tint(list) {
    const set = new Set(list);
    for (const p of parts) {
      if (set.has(p.id)) {
        p.material.emissive?.copy(HIGHLIGHT);
        p.material.emissiveIntensity = 0.16;
      } else {
        p.material.emissive?.copy(p.emissive);
        p.material.emissiveIntensity = 1;
      }
    }
  }

  // ------------------------------------------------------------ overview labels
  const overlay = document.createElement("div");
  overlay.className = "part-labels";
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  overlay.append(svg);
  stage.append(overlay);
  const labels = layout.overview.labels
    .map((l) => {
      const part = byId.get(l.id);
      if (!part) return null;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "part-label";
      button.textContent = text(l);
      button.setAttribute("aria-label", (zh ? "放大查看：" : "Inspect: ") + button.textContent);
      button.addEventListener("click", () => select(part, true));
      button.addEventListener("pointerenter", () => !current && tint([part.id]));
      button.addEventListener("pointerleave", () => !current && tint([]));
      button.addEventListener("focus", () => !current && tint([part.id]));
      button.addEventListener("blur", () => !current && tint([]));
      overlay.append(button);
      const line = document.createElementNS(svg.namespaceURI, "path"),
        dot = document.createElementNS(svg.namespaceURI, "circle");
      dot.setAttribute("r", "2.2");
      svg.append(line, dot);
      return { part, button, line, dot };
    })
    .filter(Boolean);
  const v = new THREE.Vector3();
  function placeLabels() {
    const w = stage.clientWidth,
      h = stage.clientHeight,
      show = mode === "overview" && !current && amount > 0.4;
    overlay.hidden = !show;
    if (!show) return;
    if (boxesDirty) {
      for (const p of parts) p.box.setFromObject(p.object);
      boxesDirty = false;
    }
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    let minX = w,
      maxX = 0;
    for (const p of parts)
      for (const x of [p.box.min.x, p.box.max.x])
        for (const y of [p.box.min.y, p.box.max.y])
          for (const z of [p.box.min.z, p.box.max.z]) {
            v.set(x, y, z).project(camera);
            const sx = ((v.x + 1) * w) / 2;
            minX = Math.min(minX, sx);
            maxX = Math.max(maxX, sx);
          }
    const groups = [[], []];
    for (const l of labels) {
      l.part.box.getCenter(v).project(camera);
      const ok = v.z > -1 && v.z < 1;
      l.button.hidden = !ok;
      l.line.style.display = l.dot.style.display = ok ? "" : "none";
      if (!ok) continue;
      l.ax = ((v.x + 1) * w) / 2;
      l.ay = ((1 - v.y) * h) / 2;
      // Each callout goes to the side its part is on, so leader lines never cross the robot.
      groups[l.ax < (minX + maxX) / 2 ? 0 : 1].push(l);
    }
    const gap = 26;
    groups.forEach((list, side) => {
      // Text columns sit just outside the robot; rows keep their anchor height
      // where possible and only slide apart to avoid overlapping.
      list.sort((a, b) => a.ay - b.ay);
      let y = 14;
      for (const l of list) {
        l.y = Math.max(l.ay, y);
        y = l.y + gap;
      }
      let limit = h - 14;
      for (let i = list.length - 1; i >= 0; i--) {
        list[i].y = Math.min(list[i].y, limit);
        limit = list[i].y - gap;
      }
      for (const l of list) {
        const bw = l.button.offsetWidth,
          bh = l.button.offsetHeight,
          x = side ? Math.min(maxX + 36, w - bw - 4) : Math.max(minX - 36 - bw, 4),
          edge = side ? x - 4 : x + bw + 4,
          elbow = edge + (side ? -14 : 14);
        l.button.style.left = Math.round(x) + "px";
        l.button.style.top = Math.round(l.y - bh / 2) + "px";
        l.line.setAttribute("d", `M ${l.ax.toFixed(1)} ${l.ay.toFixed(1)} L ${elbow.toFixed(1)} ${l.y.toFixed(1)} L ${edge.toFixed(1)} ${l.y.toFixed(1)}`);
        l.dot.setAttribute("cx", l.ax.toFixed(1));
        l.dot.setAttribute("cy", l.ay.toFixed(1));
      }
    });
  }

  // ------------------------------------------------------------ info panel
  function describe(title, body) {
    info.replaceChildren();
    const strong = document.createElement("strong"),
      p = document.createElement("p");
    strong.textContent = title;
    p.textContent = body;
    info.append(strong, p);
  }
  const overviewStatus = () =>
    zh ? `${parts.length} 个独立部件 · 拖动旋转` : `${parts.length} parts · Drag to orbit`;
  function select(part, zoom = false) {
    current = part;
    for (const p of parts) p.object.visible = !part || p === part;
    tint([]);
    if (part) {
      status.textContent = zh ? "部件细节 · 拖动旋转" : "Part detail · Drag to orbit";
      describe(part.object.userData["label_" + lang] || part.object.name, part.object.userData["description_" + lang] || "");
      if (zoom) frameCurrent();
    } else {
      status.textContent = overviewStatus();
      describe(
        zh ? "点击部件或名称，即可放大查看" : "Click a part or its label to inspect it",
        zh
          ? "左键拖动旋转，按住滚轮拖动平移，滚轮缩放；滑块在装配与展开之间切换。上方可切换到颈部、髋关节和踝部的组装演示。"
          : "Left-drag to orbit, middle-drag to pan, scroll to zoom; the slider moves between assembled and exploded. Use the buttons above for the neck, hip and ankle assembly demonstrations.",
      );
    }
    root.dataset.selectedPart = part?.id || "";
  }

  // ------------------------------------------------------------ modes
  const modeButtons = [
    ["overview", zh ? "整体结构" : "Overview"],
    ...demos.map((d) => [d.id, text(d)]),
  ].map(([id, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "anatomy-mode";
    b.dataset.mode = id;
    b.textContent = label;
    b.addEventListener("click", () => setMode(id));
    modes.append(b);
    return b;
  });
  const steps = (d) => [...d.steps, { start: d.motion.start, end: d.motion.end, zh: d.motion.zh, en: d.motion.en, parts: [] }];
  function renderSteps() {
    stepsList.replaceChildren();
    steps(demo).forEach((s, i) => {
      const li = document.createElement("li"),
        b = document.createElement("button");
      b.type = "button";
      b.textContent = text(s);
      b.addEventListener("click", () => {
        animate(i === 0 ? 0 : s.start);
        play(true);
      });
      li.append(b);
      stepsList.append(li);
    });
  }
  function setMode(id) {
    mode = id;
    current = null;
    root.dataset.mode = id;
    modeButtons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.mode === id)));
    explodeControl.hidden = id !== "overview";
    assembly.hidden = panel.hidden = id === "overview";
    info.hidden = id !== "overview";
    stepIndex = -1;
    if (id === "overview") {
      demo = null;
      play(false);
      for (const p of parts) p.object.visible = true;
      explodeInput.value = Math.round(amount * 100);
      pose(amount);
      select(null);
      frameCurrent(true);
      return;
    }
    demo = demos.find((d) => d.id === id);
    const visible = new Set(demo.parts);
    for (const p of parts) p.object.visible = visible.has(p.id);
    restPose();
    tint([]);
    status.textContent = zh
      ? `${text(demo)} · ${demo.parts.length} 个零件 · 拖动旋转`
      : `${text(demo)} · ${demo.parts.length} parts · Drag to orbit`;
    root.dataset.selectedPart = "";
    renderSteps();
    animate(0);
    frameCurrent(true);
    play(true);
  }
  function play(on) {
    if (on && demo && time >= demo.duration) animate(0);
    playing = Boolean(on && demo);
    last = performance.now();
    playButton.textContent = playing ? playButton.dataset.pause : playButton.dataset.play;
    playButton.setAttribute("aria-pressed", String(playing));
    root.dataset.playing = String(playing);
  }
  function syncDemoUI() {
    if (!demo) return;
    scrub.value = Math.round((time / demo.duration) * 1000);
    clock.textContent = `${time.toFixed(1)} / ${demo.duration.toFixed(1)} s`;
    const list = steps(demo);
    let index = 0;
    list.forEach((s, i) => {
      if (time >= s.start) index = i;
    });
    const active = list[index],
      moving = index === list.length - 1 ? (time <= demo.motion.end ? demo.motion.joints.flatMap((j) => j.parts) : []) : time <= active.end ? active.parts : [];
    tint(moving);
    if (index !== stepIndex) {
      stepIndex = index;
      [...stepsList.children].forEach((li, i) => {
        li.classList.toggle("is-done", i < index);
        if (i === index) li.setAttribute("aria-current", "step");
        else li.removeAttribute("aria-current");
      });
      root.dataset.step = String(index);
    }
  }
  explodeInput.addEventListener("input", (e) => {
    if (current) select(null);
    pose(Number(e.target.value) / 100);
    // Keep the robot a similar size while it opens.
    frameCurrent();
  });
  playButton.addEventListener("click", () => play(!playing));
  replayButton.addEventListener("click", () => {
    animate(0);
    play(true);
  });
  scrub.addEventListener("input", () => {
    play(false);
    animate((Number(scrub.value) / 1000) * demo.duration);
  });
  function reset() {
    amount = 1;
    setMode("overview");
  }
  root.querySelector(".view-reset").addEventListener("click", reset);

  // ------------------------------------------------------------ picking
  const raycaster = new THREE.Raycaster(),
    pointer = new THREE.Vector2();
  let down;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    down = e.button === 0 ? [e.clientX, e.clientY] : null;
  });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, (-(e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(
      parts.filter((p) => p.object.visible).map((p) => p.object),
      false,
    )[0];
    if (!hit) return;
    const part = parts.find((p) => p.object === hit.object);
    if (mode === "overview") select(part, true);
    else status.textContent = part.object.userData["label_" + lang] || part.id;
  });

  // ------------------------------------------------------------ loop
  stage.append(renderer.domElement);
  const resize = () => {
    renderer.setSize(stage.clientWidth, stage.clientHeight);
    camera.aspect = stage.clientWidth / Math.max(1, stage.clientHeight);
    camera.updateProjectionMatrix();
  };
  resize();
  applyTheme();
  new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  reset();
  new ResizeObserver(() => {
    resize();
    frameCurrent();
  }).observe(stage);
  let visible = true;
  new IntersectionObserver((es) => {
    visible = es[0].isIntersecting;
  }).observe(stage);
  renderer.setAnimationLoop(() => {
    if (!visible || document.hidden) {
      last = performance.now();
      return;
    }
    if (playing) {
      const now = performance.now();
      animate(time + Math.min(0.1, (now - last) / 1000));
      last = now;
      if (time >= demo.duration) play(false);
    }
    if (demo) syncDemoUI();
    controls.update();
    rig.quaternion.copy(camera.quaternion);
    renderer.render(scene, camera);
    placeLabels();
  });
  root.dataset.loaded = "true";
  root.dataset.componentCount = parts.length;
  // Hooks for automated checks: seek a demonstration and read node transforms.
  root.anatomy = {
    setMode,
    seek: (t) => {
      play(false);
      animate(t);
    },
    play,
    state: () => ({ mode, time, playing, amount, step: stepIndex, selected: current?.id || null }),
    // Visible model extent on screen as a fraction of the stage (width, height).
    extent: () => {
      model.updateMatrixWorld(true);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of parts) {
        if (!p.object.visible) continue;
        const b = new THREE.Box3().setFromObject(p.object);
        for (const x of [b.min.x, b.max.x])
          for (const y of [b.min.y, b.max.y])
            for (const z of [b.min.z, b.max.z]) {
              v.set(x, y, z).project(camera);
              x0 = Math.min(x0, v.x); x1 = Math.max(x1, v.x); y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y);
            }
      }
      return { width: (x1 - x0) / 2, height: (y1 - y0) / 2 };
    },
    nodes: () => Object.fromEntries(parts.map((p) => [p.id, { visible: p.object.visible, position: p.object.position.toArray(), quaternion: p.object.quaternion.toArray() }])),
  };
}
