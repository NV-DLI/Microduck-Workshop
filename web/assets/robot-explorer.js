import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

// "Inside the little duck": an illustrated, cel-shaded explorer of the Microduck CAD.
// Three views -- the whole duck, an exploded view in functional groups, and how the
// servos drive the joints (9 motions: head, neck and the left leg; the right leg mirrors it). Groups, offsets, palette, joint and servo data come from
// robot/explorer-layout.json; the rest pose is the unchanged CAD STAND pose in robot/microduck.glb.
//
// Look: a toon material with three flat tones per part, then a screen-space pass that inks
// part boundaries, silhouettes and sharp creases from a part-ID/normal/depth buffer (no mesh
// tessellation edges), with a pale halo so the figure reads on dark pages too. In the joint
// view the parts that do not move are drawn only as light contour lines from a separate
// ID/depth buffer, depth-tested against the solid parts. The selected joint shows a thin,
// slightly transparent white shaft centred in its servo housing (depth-tested, so it disappears
// inside the motor) and, just beyond the shaft end, a white paper-ribbon arrow curved on a cylinder
// about the same axis; drag it to turn the joint.
//
// Joint angles: q is the signed hinge coordinate of the course MJCF (radians, right-hand rule
// about the joint axis), shown in degrees. The glb rest pose is the STAND key, so a part below
// joint j turns by q_j - stand_j about the joint's rest pivot and axis; several joints compose
// root first (product of exponentials).

const UI = {
  zh: {
    chapters: [
      ["整只小鸭", "先认识六个部分：头、脖子、身体、髋、膝和脚。点一个名称，看哪些舵机让它动起来；点零件可以放大看它是什么，拖动可以转着看。"],
      ["拆解图", "同一组零件一起散开。标签和“电机怎样驱动关节”里的部位一一对应，共 15 个舵机：引线从每个舵机外壳出发，汇合后指向部位名称。点一个标签，看这些舵机装在哪里。"],
      ["电机怎样驱动关节", ""],
    ],
    body: "身体",
    eye: "眼睛 = 摄像头",
    tof: "ToF 测距",
    board: "主控板",
    battery: "电池",
    fixed: "仿真模型中固定",
    list: "、",
    servos: (n, perSide) => (perSide ? `每侧 ${n} 个舵机` : `${n} 个舵机`),
    side: (side) => (side === "left" ? "左侧" : "右侧"),
    play: "演示",
    pause: "暂停",
    zero: "回到站姿",
    q: "q",
    stand: "站姿",
    delta: "相对站姿",
    range: "范围",
    to: "至",
    limit: "已到限位",
    handle: (name) => `转动${name}：方向键每次 1°，Page Up/Down 每次 10°`,
    back: "返回",
    part: "你点的是：",
  },
  en: {
    chapters: [
      ["The whole duck", "Meet the six parts: head, neck, body, hip, knee and foot. Click a name to see which servos move it; click a part to look closer, and drag to turn the duck around."],
      ["Exploded view", "Parts that belong together move apart as a group. The labels match the parts in How motors drive joints, 15 servos in all: a leader starts at each servo housing and they merge before the name. Click a label to see where those servos sit."],
      ["How motors drive joints", ""],
    ],
    body: "Body",
    eye: "Eye = camera",
    tof: "ToF range sensor",
    board: "Controller board",
    battery: "Battery",
    fixed: "fixed in the simulation model",
    list: ", ",
    servos: (n, perSide) => `${n} servo${n === 1 ? "" : "s"}${perSide ? " per side" : ""}`,
    side: (side) => `${side} side`,
    play: "Demo",
    pause: "Pause",
    zero: "Back to stand",
    q: "q",
    stand: "stand",
    delta: "from stand",
    range: "range",
    to: "to",
    limit: "at the limit",
    handle: (name) => `Turn ${name}: arrow keys 1°, Page Up/Down 10°`,
    back: "Back",
    part: "You clicked: ",
  },
};

const THEMES = {
  light: { ghost: "#EDEFF3", ghostInk: "#C4CAD4", ghostLine: "#A3ACBB", halo: "#FFFFFF", haloAlpha: 0.9, shaftEdge: "#5E6572" },
  dark: { ghost: "#3B4250", ghostInk: "#5B6373", ghostLine: "#7C8698", halo: "#F4ECDF", haloAlpha: 0.92, shaftEdge: "#14171D" },
};
// Joint shaft and paper-ribbon arrow (metres): radius ~0.3 mm, clearance and gap beyond the shaft end.
const SHAFT_R = 0.0003,
  SHAFT_MARGIN = 0.008, // shaft end beyond the last part inside the ribbon's cylinder
  SHAFT_EXPOSED = 0.01, // at least this much shaft shows beyond each face of the housing
  SHAFT_EXTRA = 0.01, // and the shaft runs this much further at both ends
  ARC_GAP = 0.007,
  RIBBON_RADIUS_SCALE = 1.45, // the ribbon's radius, relative to the base size below
  RIBBON_WIDTH_SCALE = 1.75, // and its width along the axis
  RIBBON_T = 0.0018; // band thickness (radial)

const rgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return new THREE.Vector3(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};
const clamp01 = (u) => Math.min(1, Math.max(0, u));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const easeInOut = (u) => ((u = clamp01(u)) < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2);
const DEG = 180 / Math.PI;
// Piecewise keys [t, value] with the same easing per segment.
function sampleKeys(keys, t) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [ta, a] = keys[i - 1],
      [tb, b] = keys[i];
    if (t <= tb) return a + (b - a) * easeInOut((t - ta) / (tb - ta || 1));
  }
  return keys[keys.length - 1][1];
}

// ------------------------------------------------------------------ shaders
const TOON_VERT = /* glsl */ `
varying vec3 vN;
void main() {
  vN = normalize(normalMatrix * normal);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
// Three flat tones from a half-Lambert term; teaching tints keep the same banding.
const TOON_FRAG = /* glsl */ `
uniform vec3 uLight, uMid, uShadow, uTint, uGhostColor, uLightDir;
uniform float uTintMix, uGhost, uPulse;
uniform vec2 uBands;
varying vec3 vN;
void main() {
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  float l = 0.5 * dot(n, normalize(uLightDir)) + 0.5;
  float band = l > uBands.y ? 1.0 : (l > uBands.x ? 0.86 : 0.7);
  vec3 c = l > uBands.y ? uLight : (l > uBands.x ? uMid : uShadow);
  c = mix(c, uTint * band, uTintMix);
  c = mix(c, vec3(1.0), uPulse);
  c = mix(c, uGhostColor * (0.94 + 0.06 * band), uGhost);
  gl_FragColor = vec4(c, 1.0);
}`;
const ID_FRAG = /* glsl */ `
uniform float uId;
varying vec3 vN;
void main() {
  vec3 n = normalize(vN);
  if (!gl_FrontFacing) n = -n;
  gl_FragColor = vec4(n * 0.5 + 0.5, uId / 255.0);
}`;
// The ribbon arrow: a curved band with a visible thickness. Each vertex carries its surface
// (0 outer, 1 inner, 2 cut edge), so the outer face stays near-white, the inner face soft grey and
// the edges medium grey from any view; smooth radial normals add a gentle diffuse curvature shade.
const RIBBON_VERT = /* glsl */ `
attribute float aSurface;
varying vec3 vN;
varying float vSurface;
void main() {
  vN = normalize(normalMatrix * normal);
  vSurface = aSurface;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const RIBBON_FRAG = /* glsl */ `
uniform vec3 uLightDir;
varying vec3 vN;
varying float vSurface;
void main() {
  float l = 0.5 * dot(normalize(vN), normalize(uLightDir)) + 0.5;
  float base = vSurface < 0.5 ? 0.98 : (vSurface < 1.5 ? 0.8 : 0.62);
  gl_FragColor = vec4(vec3(base * mix(0.8, 1.0, l)), 1.0);
}`;
// The shaft: near-white with a soft two-tone shade, its casing a neutral edge colour.
const SHAFT_FRAG = /* glsl */ `
uniform vec3 uColor, uLightDir;
uniform float uShade;
varying vec3 vN;
void main() {
  float l = 0.5 * dot(normalize(vN), normalize(uLightDir)) + 0.5;
  gl_FragColor = vec4(uColor * mix(1.0 - uShade, 1.0, step(0.45, l)), 1.0);
}`;
const POST_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
// Ink: part-ID changes and depth jumps inside the figure, a heavier silhouette, a halo
// outside it. Ghosted parts (ID >= 128) get pale lines. In contour mode a second ID/depth
// buffer holds the parts that stay still; only their edges are drawn, and only where they are
// nearer than the solid parts. Output is premultiplied alpha.
const POST_FRAG = /* glsl */ `
uniform sampler2D tColor, tId, tDepth, tGhostId, tGhostDepth, tMoveColor, tMoveId, tMoveDepth, tShaft, tShaftId, tShaftDepth;
uniform vec2 uTexel;
uniform float uInner, uOuter, uHaloW, uHaloA, uDepthK, uContour, uContourA, uMoveA, uShaftA;
uniform vec3 uInk, uGhostInk, uHalo, uContourInk, uShaftInk;
varying vec2 vUv;
float pid(vec2 uv) { return floor(texture2D(tId, uv).a * 255.0 + 0.5); }
void main() {
  vec4 col = texture2D(tColor, vUv);
  vec4 s0 = texture2D(tId, vUv);
  float id0 = floor(s0.a * 255.0 + 0.5);
  float d0 = texture2D(tDepth, vUv).r;
  bool obj0 = id0 > 0.5;
  bool g0 = id0 >= 127.5;
  float inner = 0.0, ghostLine = 0.0, outer = 0.0, ghostOuter = 0.0, halo = 0.0;
  for (int k = 0; k < 8; k++) {
    float a = float(k) * 0.7853982;
    vec2 dir = vec2(cos(a), sin(a)) * uTexel;
    vec2 uv1 = vUv + dir * uInner;
    vec4 s1 = texture2D(tId, uv1);
    float id1 = floor(s1.a * 255.0 + 0.5);
    if (obj0 && id1 > 0.5) {
      float e = 0.0;
      if (abs(id1 - id0) > 0.5) e = 1.0;
      else if (abs(texture2D(tDepth, uv1).r - d0) * uDepthK > 1.0) e = 1.0;
      else if (dot(s0.rgb * 2.0 - 1.0, s1.rgb * 2.0 - 1.0) < 0.5) e = 0.6;
      if (g0 && id1 >= 127.5) ghostLine += e; else inner += e;
    }
    float id2 = pid(vUv + dir * uOuter);
    if (obj0 != (id2 > 0.5)) {
      if ((obj0 && g0) || (!obj0 && id2 >= 127.5)) ghostOuter += 1.0; else outer += 1.0;
    }
    if (!obj0 && pid(vUv + dir * (uOuter + uHaloW)) > 0.5) halo += 1.0;
  }
  float li = clamp(inner / 2.0, 0.0, 1.0), lg = clamp(ghostLine / 2.0, 0.0, 1.0);
  float lo = clamp(outer / 1.5, 0.0, 1.0), lgo = clamp(ghostOuter / 1.5, 0.0, 1.0);
  vec4 outC;
  if (obj0) {
    vec3 c = col.a > 0.0 ? col.rgb / col.a : col.rgb;
    c = mix(c, uGhostInk, max(lg * 0.8, lgo));
    c = mix(c, uInk, max(li, lo));
    outC = vec4(c, 1.0);
  } else {
    float h = clamp(halo / 1.5, 0.0, 1.0) * uHaloA;
    vec4 under = vec4(uHalo * h, h);
    vec4 base = col + under * (1.0 - col.a);
    float ink = max(lo, lgo);
    vec3 inkC = lo >= lgo ? uInk : uGhostInk;
    outC = vec4(inkC * ink, ink) + base * (1.0 - ink);
  }
  if (uContour > 0.5) {
    // Moving parts other than the servo and bracket: a fill with ink lines, in front of the solid
    // parts only where they are nearer.
    vec4 m0 = texture2D(tMoveId, vUv);
    float mid0 = floor(m0.a * 255.0 + 0.5);
    float md0 = texture2D(tMoveDepth, vUv).r;
    float mnear = mid0 > 0.5 ? md0 : 1.0, me = 0.0;
    for (int k = 0; k < 8; k++) {
      float a = float(k) * 0.7853982;
      vec2 uv1 = vUv + vec2(cos(a), sin(a)) * uTexel * uInner;
      vec4 m1 = texture2D(tMoveId, uv1);
      float mid1 = floor(m1.a * 255.0 + 0.5);
      if (mid0 < 0.5 && mid1 < 0.5) continue;
      float md1 = texture2D(tMoveDepth, uv1).r;
      if (abs(mid1 - mid0) > 0.5) { me += 1.0; if (mid1 > 0.5) mnear = min(mnear, md1); }
      else if (abs(md1 - md0) * uDepthK > 1.0) me += 1.0;
      else if (dot(m0.rgb * 2.0 - 1.0, m1.rgb * 2.0 - 1.0) < 0.5) me += 0.6;
    }
    // The shaft (premultiplied, anti-aliased) shows where nothing solid is nearer. Behind a moving
    // part it shows through that part like a servo does; otherwise it is drawn over it.
    // Mask IDs: 254 shaft, 255 ribbon. Anti-aliased fringe pixels outside the mask take the depth of
    // a masked neighbour; ribbon pixels next to anything else get a thin neutral outline.
    vec4 sc = texture2D(tShaft, vUv);
    float sid = floor(texture2D(tShaftId, vUv).a * 255.0 + 0.5);
    float sd = sid > 0.5 ? texture2D(tShaftDepth, vUv).r : 1.0, rEdge = 0.0;
    for (int k = 0; k < 4; k++) {
      vec2 uv1 = vUv + vec2(float(k == 0) - float(k == 1), float(k == 2) - float(k == 3)) * uTexel * uInner;
      float nid = floor(texture2D(tShaftId, uv1).a * 255.0 + 0.5);
      if (sid < 0.5 && nid > 0.5) sd = min(sd, texture2D(tShaftDepth, uv1).r);
      if (sid > 254.5 && nid < 254.5) rEdge = 1.0;
    }
    bool shaftOn = sc.a > 0.003 && sd < d0;
    bool shaftBehind = shaftOn && mid0 > 0.5 && md0 < sd;
    vec4 shaftC = shaftOn ? sc * uShaftA : vec4(0.0);
    if (shaftOn && rEdge > 0.5) shaftC = vec4(uShaftInk * 0.85, 0.85) + shaftC * 0.15;
    if (shaftBehind) outC = shaftC + outC * (1.0 - shaftC.a);
    if (mid0 > 0.5 && md0 < d0) {
      // See-through where a servo, bracket or the shaft lies behind; nearly opaque elsewhere, so
      // the part keeps its colours on light and dark pages.
      float a = (obj0 || shaftBehind) ? uMoveA : 0.9;
      vec4 mc = texture2D(tMoveColor, vUv);
      outC = vec4(mc.rgb, 1.0) * a + outC * (1.0 - a);
    }
    float ma = clamp(me / 2.0, 0.0, 1.0) * (mnear <= d0 + 1e-6 ? 0.9 : 0.0);
    outC = vec4(uInk * ma, ma) + outC * (1.0 - ma);
    d0 = min(d0, mid0 > 0.5 ? md0 : 1.0);
    if (shaftOn && !shaftBehind) outC = shaftC + outC * (1.0 - shaftC.a);
    d0 = min(d0, shaftOn ? sd : 1.0);
    vec4 c0 = texture2D(tGhostId, vUv);
    float gid0 = floor(c0.a * 255.0 + 0.5);
    float gd0 = texture2D(tGhostDepth, vUv).r;
    float near = gid0 > 0.5 ? gd0 : 1.0, e = 0.0;
    for (int k = 0; k < 8; k++) {
      float a = float(k) * 0.7853982;
      vec2 uv1 = vUv + vec2(cos(a), sin(a)) * uTexel * uInner;
      vec4 c1 = texture2D(tGhostId, uv1);
      float gid1 = floor(c1.a * 255.0 + 0.5);
      if (gid0 < 0.5 && gid1 < 0.5) continue;
      float gd1 = texture2D(tGhostDepth, uv1).r;
      if (abs(gid1 - gid0) > 0.5) { e += 1.0; if (gid1 > 0.5) near = min(near, gd1); }
      else if (abs(gd1 - gd0) * uDepthK > 1.0) e += 1.0;
      else if (dot(c0.rgb * 2.0 - 1.0, c1.rgb * 2.0 - 1.0) < 0.5) e += 0.5;
    }
    // Hidden behind a solid part: no line. In front of it or over empty space: a light line.
    float vis = near <= d0 + 1e-6 ? 1.0 : 0.0;
    float ga = clamp(e / 2.0, 0.0, 1.0) * uContourA * vis;
    outC = vec4(uContourInk * ga, ga) + outC * (1.0 - ga);
  }
  gl_FragColor = outC;
}`;

// Mounts the explorer. If anything fails part-way, everything set up so far is undone (canvas,
// WebGL context, listeners, observers, controls shown) so the page can offer a clean retry.
export async function mountExplorer(root, lang, onProgress) {
  const undo = [];
  try {
    return await mount(root, lang, onProgress, undo);
  } catch (error) {
    for (const fn of undo.reverse())
      try {
        fn();
      } catch {}
    throw error;
  }
}

async function mount(root, lang, onProgress, undo) {
  const T = UI[lang === "zh" ? "zh" : "en"],
    zh = lang === "zh",
    tx = (o, key) => o[`${key}_${zh ? "zh" : "en"}`],
    $ = (s) => root.querySelector(s),
    stage = $(".dx-stage"),
    calloutLayer = $(".dx-callouts"),
    leaders = $(".dx-leaders"),
    gizmo = $(".dx-gizmo"),
    handle = $(".dx-handle"),
    chapterButtons = [...root.querySelectorAll(".dx-chapter")],
    resetButton = $(".dx-reset"),
    backButton = $(".dx-back"),
    card = $(".dx-card"),
    title = $(".dx-title"),
    fact = $(".dx-fact"),
    partLine = $(".dx-part"),
    drivePanel = $(".dx-drive"),
    regionButtons = $(".dx-regions"),
    jointButtons = $(".dx-joints"),
    fixedNote = $(".dx-fixed"),
    readQ = $(".dx-q"),
    readDetail = $(".dx-q-detail"),
    readId = $(".dx-q-id"),
    playButton = $(".dx-play"),
    zeroButton = $(".dx-zero");
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");

  // Listeners on elements that outlive a failed attempt go through this signal.
  const listeners = new AbortController(),
    signal = listeners.signal;
  let dead = false;
  undo.push(() => {
    dead = true;
    listeners.abort();
    for (const sel of [".dx-chapters", ".dx-panel", ".dx-handle"]) $(sel).hidden = true;
    for (const el of [calloutLayer, leaders, gizmo]) el.replaceChildren();
    gizmo.setAttribute("hidden", "");
    root.querySelector(".dx-motor-tip")?.remove();
    root.querySelector(".dx-side-note")?.remove();
    delete root.explorerState;
  });
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, premultipliedAlpha: true });
  undo.push(() => {
    renderer.domElement.remove();
    renderer.dispose();
    renderer.forceContextLoss();
  });
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = false;
  // At least 1.5x pixels (capped at 2x): the browser downsamples the canvas, which smooths the ink.
  const dpr = () => Math.min(Math.max(devicePixelRatio || 1, 1.5), 2);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 3),
    controls = new OrbitControls(camera, renderer.domElement);
  undo.push(() => controls.dispose());
  controls.enableZoom = false; // page scrolling stays with the page; closeups are framed for the learner
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.rotateSpeed = 0.8;
  controls.minPolarAngle = 0.15;
  controls.maxPolarAngle = Math.PI - 0.15;

  // The page names the versions it was built with: the layout and parts list share this module's
  // version, the large model has its own, so no earlier copy of any of them is reused by mistake.
  const url = (p, v = root.dataset.dxVersion) => {
      const u = new URL(p, import.meta.url);
      if (v) u.searchParams.set("v", v);
      return u.href;
    },
    json = (p) =>
      fetch(url(p)).then((r) => {
        if (!r.ok) throw Error(`Missing ${p}`);
        return r.json();
      });
  const [{ scene: model }, meta, layout] = await Promise.all([
    new GLTFLoader().loadAsync(url("./robot/microduck.glb", root.dataset.dxModelVersion), (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    }),
    json("./robot/components.json"),
    json("./robot/explorer-layout.json"),
  ]);
  stage.prepend(renderer.domElement);
  renderer.domElement.classList.add("dx-canvas");
  renderer.domElement.setAttribute("aria-hidden", "true");

  const scene = new THREE.Scene();
  scene.add(model);
  model.updateMatrixWorld(true);
  const pal = layout.palette,
    labels = new Map(meta.parts.map((p) => [p.id, p])),
    lightDir = { value: new THREE.Vector3(...pal.light_view).normalize() },
    ghostColor = { value: rgb(THEMES.light.ghost) },
    bands = { value: new THREE.Vector2(...pal.bands_half_lambert) };
  const toonMaterial = ([light, mid, shadow]) =>
    new THREE.ShaderMaterial({
      vertexShader: TOON_VERT,
      fragmentShader: TOON_FRAG,
      side: THREE.DoubleSide,
      uniforms: {
        uLight: { value: rgb(light) },
        uMid: { value: rgb(mid) },
        uShadow: { value: rgb(shadow) },
        uTint: { value: new THREE.Vector3() },
        uTintMix: { value: 0 },
        uGhost: { value: 0 },
        uPulse: { value: 0 },
        uGhostColor: ghostColor,
        uLightDir: lightDir,
        uBands: bands,
      },
    });
  const idMaterial = (id) =>
    new THREE.ShaderMaterial({ vertexShader: TOON_VERT, fragmentShader: ID_FRAG, side: THREE.DoubleSide, uniforms: { uId: { value: id } } });

  // ------------------------------------------------------------ parts and groups
  const parts = [],
    byId = new Map(),
    groups = new Map(layout.explode.groups.map((g) => [g.id, { ...g, members: [] }]));
  const meshes = [];
  model.traverse((o) => o.isMesh && meshes.push(o));
  meshes.forEach((o, index) => {
    const id = o.userData.id;
    o.geometry.computeBoundingBox();
    const part = {
      id,
      index: index + 1,
      object: o,
      toon: toonMaterial(pal.roles[layout.roles[id]]),
      idMaterial: idMaterial(index + 1),
      group: null,
      position: o.position.clone(),
      quaternion: o.quaternion.clone(),
      offset: new THREE.Vector3(...(layout.explode.parts[id] || [0, 0, 0])),
      layer: "solid", // joint view: solid | moving (see-through) | still (contour lines only)
      deepest: null, // joint view: the deepest joint this part moves with
    };
    o.material = part.toon;
    parts.push(part);
    byId.set(id, part);
  });
  for (const g of groups.values()) {
    for (const id of g.parts) {
      const p = byId.get(id);
      p.group = g;
      g.members.push(p);
    }
    g.memberSet = new Set(g.members);
  }

  // ------------------------------------------------------------ joints (source-derived)
  const drive = layout.drive;
  const joints = drive.joints.map((j) => ({
    ...j,
    pivotRest: new THREE.Vector3(...j.pivot),
    axisRest: new THREE.Vector3(...j.axis).normalize(),
    movingSet: new Set(j.moving),
    bracketSet: new Set(j.bracket),
    world: new THREE.Matrix4(), // ancestors and own rotation, applied to rest coordinates
    parentWorld: new THREE.Matrix4(),
  }));
  const jointById = new Map(joints.map((j) => [j.id, j]));
  // The joint view offers 9 motions (head, neck, left leg); their servo housings are click targets.
  const enabled = drive.enabled.map((id) => jointById.get(id)),
    motorJoint = new Map(enabled.map((j) => [j.servo, j])),
    regionOf = new Map(drive.regions.flatMap((r) => r.joints.map((id) => [id, r])));
  // Functional parts (drive.functional, derived from drive.regions): the same head, neck, hip, knee
  // and foot as the joint view, with every servo that drives them (per side for the legs) and the
  // fixed beak servo. The whole-duck and exploded labels use them.
  const regionById = new Map(drive.regions.map((r) => [r.id, r])),
    functional = new Map(drive.functional.map((f) => [f.id, f])),
    owned = (f) => (f.per_side ? f[nearSide] : f.both),
    partInfo = (id) => (id === "beak" ? regionById.get("head").fixed : regionById.get(id)),
    motionList = (f) =>
      owned(f)
        .joints.map((id, i) => {
          const name = tx(jointById.get(id), "label").replace(/\s*[（(].*?[)）]/g, "");
          return i && !zh ? name[0].toLowerCase() + name.slice(1) : name; // "Hip yaw, hip roll, hip pitch"
        })
        .join(T.list);
  // Clicking a part's label shows the explode groups that hold its servos and brackets, with the
  // servos in the actuator colour.
  function partFocus(id) {
    const f = functional.get(id),
      o = owned(f),
      info = partInfo(id),
      members = parts.filter((p) => o.focus_groups.includes(p.group.id));
    return {
      id: `part:${id}${f.per_side ? `:${nearSide}` : ""}`,
      label_zh: `${info.label_zh}${f.per_side ? `（${UI.zh.side(nearSide)}）` : ""} · ${UI.zh.servos(o.servos.length, false)}`,
      label_en: `${info.label_en}${f.per_side ? ` (${UI.en.side(nearSide)})` : ""} · ${UI.en.servos(o.servos.length, false)}`,
      fact_zh: f.fixed ? info.note_zh : info.fact_zh,
      fact_en: f.fixed ? info.note_en : info.fact_en,
      members,
      memberSet: new Set(members),
      servos: new Set(o.servos),
    };
  }
  joints.sort((a, b) => a.ancestors.length - b.ancestors.length);
  for (const p of parts)
    for (const j of joints) if (j.movingSet.has(p.id) && (!p.deepest || j.ancestors.length > p.deepest.ancestors.length)) p.deepest = j;
  const q = new Map(); // current q per joint (rad)
  const resetAngles = () => joints.forEach((j) => q.set(j.id, j.stand_rad));
  resetAngles();
  const ACCENT = { actuator: rgb(pal.accents.actuator), link: rgb(pal.accents.link) };

  // ------------------------------------------------------------ NPR render targets
  const post = new THREE.ShaderMaterial({
    vertexShader: POST_VERT,
    fragmentShader: POST_FRAG,
    depthTest: false,
    depthWrite: false,
    transparent: false,
    uniforms: {
      tColor: { value: null },
      tId: { value: null },
      tDepth: { value: null },
      tGhostId: { value: null },
      tGhostDepth: { value: null },
      tMoveColor: { value: null },
      tMoveId: { value: null },
      tMoveDepth: { value: null },
      tShaft: { value: null },
      tShaftId: { value: null },
      tShaftDepth: { value: null },
      uMoveA: { value: 0.42 },
      uShaftA: { value: 0.8 },
      uShaftInk: { value: rgb(THEMES.light.shaftEdge) },
      uTexel: { value: new THREE.Vector2() },
      uInner: { value: 1 },
      uOuter: { value: 1.6 },
      uHaloW: { value: 2 },
      uHaloA: { value: 0.9 },
      uDepthK: { value: 1 },
      uContour: { value: 0 },
      uContourA: { value: 0.85 },
      uInk: { value: rgb(pal.outline.ink) },
      uGhostInk: { value: rgb(THEMES.light.ghostInk) },
      uContourInk: { value: rgb(THEMES.light.ghostLine) },
      uHalo: { value: rgb(THEMES.light.halo) },
    },
  });
  const postScene = new THREE.Scene(),
    postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  postScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), post));
  let colorTarget = null,
    idTarget = null,
    ghostTarget = null,
    moveColor = null,
    moveId = null,
    shaftColor = null,
    shaftId = null;
  function resizeTargets(w, h) {
    for (const t of [colorTarget, idTarget, ghostTarget, moveColor, moveId, shaftColor, shaftId]) {
      t?.depthTexture?.dispose();
      t?.dispose();
    }
    colorTarget = new THREE.WebGLRenderTarget(w, h, { samples: 4 });
    moveColor = new THREE.WebGLRenderTarget(w, h, { samples: 4 });
    shaftColor = new THREE.WebGLRenderTarget(w, h, { samples: 4 });
    const idT = () => {
      const t = new THREE.WebGLRenderTarget(w, h, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
      t.depthTexture = new THREE.DepthTexture(w, h);
      return t;
    };
    idTarget = idT();
    ghostTarget = idT();
    moveId = idT();
    shaftId = idT();
    post.uniforms.tShaft.value = shaftColor.texture;
    post.uniforms.tShaftId.value = shaftId.texture;
    post.uniforms.tShaftDepth.value = shaftId.depthTexture;
    post.uniforms.tMoveColor.value = moveColor.texture;
    post.uniforms.tMoveId.value = moveId.texture;
    post.uniforms.tMoveDepth.value = moveId.depthTexture;
    post.uniforms.tColor.value = colorTarget.texture;
    post.uniforms.tId.value = idTarget.texture;
    post.uniforms.tDepth.value = idTarget.depthTexture;
    post.uniforms.tGhostId.value = ghostTarget.texture;
    post.uniforms.tGhostDepth.value = ghostTarget.depthTexture;
    post.uniforms.uTexel.value.set(1 / w, 1 / h);
    const r = dpr();
    post.uniforms.uInner.value = 0.8 * r;
    post.uniforms.uOuter.value = 1.35 * r;
    post.uniforms.uHaloW.value = 2.2 * r;
  }

  // ------------------------------------------------------------ shaft (depth-tested)
  // A thin white rod on the joint axis, drawn in its own pass: an edge-coloured casing one device
  // pixel wider, the white core on top, and a slightly wider mask whose depth hides the length
  // inside the motor or any other solid part. The post pass blends it at uShaftA.
  const shaftMaterial = (color, shade) =>
    new THREE.ShaderMaterial({
      vertexShader: TOON_VERT,
      fragmentShader: SHAFT_FRAG,
      depthTest: false,
      depthWrite: false,
      uniforms: { uColor: { value: rgb(color) }, uLightDir: lightDir, uShade: { value: shade } },
    });
  const rodGeometry = new THREE.CylinderGeometry(1, 1, 1, 24),
    shaftCasing = new THREE.Mesh(rodGeometry, shaftMaterial(THEMES.light.shaftEdge, 0)),
    shaftCore = new THREE.Mesh(rodGeometry, shaftMaterial("#FFFFFF", 0.1)),
    shaftMask = new THREE.Mesh(rodGeometry, idMaterial(254)),
    ribbon = new THREE.Mesh(
      new THREE.BufferGeometry()
        .setAttribute("position", new THREE.Float32BufferAttribute(new Array(9).fill(0), 3))
        .setAttribute("normal", new THREE.Float32BufferAttribute(new Array(9).fill(0), 3))
        .setAttribute("aSurface", new THREE.Float32BufferAttribute([0, 0, 0], 1)),
      new THREE.ShaderMaterial({
        vertexShader: RIBBON_VERT,
        fragmentShader: RIBBON_FRAG,
        side: THREE.DoubleSide,
        uniforms: { uLightDir: lightDir },
      }),
    ),
    ribbonMask = new THREE.Mesh(ribbon.geometry, idMaterial(255)),
    shaftScene = new THREE.Scene();
  shaftCasing.renderOrder = 0;
  shaftCore.renderOrder = 1;
  ribbon.renderOrder = 2;
  for (const m of [ribbon, ribbonMask]) m.matrixAutoUpdate = false;
  shaftScene.add(shaftCasing, shaftCore, ribbon);
  const maskScene = new THREE.Scene();
  maskScene.add(shaftMask, ribbonMask);

  // ------------------------------------------------------------ state
  let chapter = 0,
    focus = null, // focused group (whole/apart)
    clickedPart = null,
    hovered = null,
    hoverMotor = null, // joint view: the joint whose servo housing is under the pointer
    frameNo = 0, // frames drawn (diagnostics cache key)
    region = null,
    joint = null,
    playing = false,
    playStart = 0,
    playKeys = null,
    explodeT = 0, // 0 assembled ... 1 exploded (timeline with stagger)
    explodeFrom = 0,
    explodeTo = 0,
    explodeStart = 0,
    visible = true,
    needsRender = true,
    camTween = null,
    nearSide = "right";
  const DUR = layout.explode.duration_s;
  const groupProgress = (g, t) => easeInOut((t - g.delay) / layout.explode.group_share);
  const rotation = (pivot, axis, a) =>
    new THREE.Matrix4()
      .makeTranslation(pivot.x, pivot.y, pivot.z)
      .multiply(new THREE.Matrix4().makeRotationAxis(axis, a))
      .multiply(new THREE.Matrix4().makeTranslation(-pivot.x, -pivot.y, -pivot.z));
  const angles = () => (chapter === 2 ? q : null);
  function updateJointMatrices(qs) {
    for (const j of joints) {
      const parent = j.ancestors.length ? jointById.get(j.ancestors[j.ancestors.length - 1]) : null;
      j.parentWorld.copy(parent ? parent.world : new THREE.Matrix4());
      const d = qs ? qs.get(j.id) - j.stand_rad : 0;
      j.world.copy(j.parentWorld);
      if (d) j.world.multiply(rotation(j.pivotRest, j.axisRest, d));
      j.atRest = !d && (!parent || parent.atRest); // nothing above or at this joint has turned
    }
  }
  // Pose for an explosion time and the joint angles (joint view only; elsewhere the CAD rest pose).
  const scratch = new THREE.Matrix4(),
    unitScale = new THREE.Vector3();
  function applyPose(t = explodeT, qs = angles()) {
    updateJointMatrices(qs);
    for (const p of parts) {
      const o = p.object;
      o.position.copy(p.position).addScaledVector(p.offset, groupProgress(p.group, t));
      o.quaternion.copy(p.quaternion);
      if (qs && p.deepest && !p.deepest.atRest) {
        scratch.copy(p.deepest.world).multiply(new THREE.Matrix4().compose(o.position, o.quaternion, unitScale.set(1, 1, 1)));
        scratch.decompose(o.position, o.quaternion, unitScale);
      }
    }
    model.updateMatrixWorld(true);
    placeShaft();
  }
  // Current pivot and axis of a joint (its rest values carried by the joints above it).
  const jointPivot = (j) => j.pivotRest.clone().applyMatrix4(j.parentWorld);
  const jointAxis = (j) => j.axisRest.clone().transformDirection(j.parentWorld);

  // ------------------------------------------------------------ framing (orthographic)
  const corners = (p, out) => {
    const b = p.object.geometry.boundingBox;
    for (let i = 0; i < 8; i++)
      out.push(new THREE.Vector3(i & 1 ? b.max.x : b.min.x, i & 2 ? b.max.y : b.min.y, i & 4 ? b.max.z : b.min.z).applyMatrix4(p.object.matrixWorld));
  };
  // Camera that shows `points` from `direction`; widthShare leaves room for callouts.
  function framing(points, direction, widthShare, heightShare) {
    const d = new THREE.Vector3(...direction).normalize(),
      right = new THREE.Vector3(0, 1, 0).cross(d).normalize(),
      up = new THREE.Vector3().crossVectors(d, right);
    let x0 = Infinity,
      x1 = -Infinity,
      y0 = Infinity,
      y1 = -Infinity;
    const c = new THREE.Vector3();
    for (const v of points) {
      x0 = Math.min(x0, v.dot(right));
      x1 = Math.max(x1, v.dot(right));
      y0 = Math.min(y0, v.dot(up));
      y1 = Math.max(y1, v.dot(up));
      c.add(v);
    }
    c.divideScalar(points.length);
    const depth = c.dot(d),
      target = right
        .clone()
        .multiplyScalar((x0 + x1) / 2)
        .addScaledVector(up, (y0 + y1) / 2)
        .addScaledVector(d, depth);
    const aspect = stage.clientWidth / Math.max(1, stage.clientHeight),
      half = Math.max((y1 - y0) / 2 / heightShare, (x1 - x0) / 2 / (aspect * widthShare));
    return { target, direction: d, half };
  }
  function pointsFor(selector, t, poses, extra = null) {
    const pts = [];
    for (const qs of poses) {
      applyPose(t, qs);
      for (const p of parts) if (selector(p)) corners(p, pts);
      extra?.(pts);
    }
    applyPose();
    return pts;
  }
  const roomy = () => stage.clientWidth >= 640;
  function goalView() {
    if (chapter === 2 && joint) {
      const sel = (p) => joint.movingSet.has(p.id) || p.id === joint.servo;
      // Fit everything the learner can reach by dragging this joint through its whole range.
      const [lo, hi] = joint.range_rad,
        poses = [0, 0.25, 0.5, 0.75, 1].map((u) => new Map(q).set(joint.id, lo + (hi - lo) * u));
      poses.push(new Map(q));
      // ... together with the shaft and the semicircle, which turn with it.
      const pts = pointsFor(sel, 0, poses, (out) => out.push(...ribbonOutline(), ...shaftNow().ends));
      return framing(pts, jointView(joint), 0.86, 0.84);
    }
    const t = chapter === 1 ? 1 : 0;
    if (focus) {
      // Large but not microscopic: small groups keep some of their neighbours in view.
      const view = framing(pointsFor((p) => focus.memberSet.has(p), t, [null]), currentDirection(), 0.62, 0.7);
      view.half = Math.max(view.half, 0.065);
      return view;
    }
    return framing(pointsFor(() => true, t, [null]), layout.view, roomy() ? 0.6 : 0.88, 0.94);
  }
  // Look at the joint's rotation plane at about 57 degrees from its axis, from the shaft end that
  // holds the semicircle and from the front-above, so the semicircle opens up beside the motor
  // instead of collapsing into a line or lying over it.
  function jointView(j) {
    const a = jointAxis(j),
      side = j.side === "left" ? -1 : j.side === "right" ? 1 : 1,
      base = new THREE.Vector3(...layout.view).normalize(),
      sideView = new THREE.Vector3(base.x, base.y, Math.abs(base.z) * side).normalize(),
      s = gizmoFor(j).side,
      along = a.clone().multiplyScalar(s),
      perp = sideView.clone().addScaledVector(along, -sideView.dot(along));
    if (perp.lengthSq() < 1e-6) perp.set(1, 0.4, 0);
    perp.normalize();
    const d = along.multiplyScalar(Math.cos(1.0)).addScaledVector(perp, Math.sin(1.0)).normalize();
    if (d.y < 0.15) d.y = 0.15;
    return d.normalize().toArray();
  }
  const currentDirection = () => camera.position.clone().sub(controls.target).normalize().toArray();
  let half = 0.15;
  function setCamera(target, direction, h) {
    half = h;
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(new THREE.Vector3(...(direction.toArray ? direction.toArray() : direction)).normalize(), 1.2);
    updateFrustum();
    controls.update();
  }
  function updateFrustum() {
    const aspect = stage.clientWidth / Math.max(1, stage.clientHeight);
    camera.left = -half * aspect;
    camera.right = half * aspect;
    camera.top = half;
    camera.bottom = -half;
    camera.near = 0.6;
    camera.far = 1.8;
    camera.updateProjectionMatrix();
    post.uniforms.uDepthK.value = (camera.far - camera.near) / 0.006;
  }
  // Drop any orbit inertia left from the learner's last drag (an update without damping clears it).
  function stopOrbitInertia() {
    controls.enableDamping = false;
    controls.update();
    controls.enableDamping = true;
  }
  function moveCamera(instant = false) {
    stopOrbitInertia();
    const goal = goalView();
    if (goal.direction && goal.direction.toArray === undefined) goal.direction = new THREE.Vector3(...goal.direction);
    if (instant || reduceMotion.matches) {
      camTween = null;
      setCamera(goal.target, goal.direction, goal.half);
    } else {
      camTween = {
        start: performance.now(),
        from: { target: controls.target.clone(), direction: new THREE.Vector3(...currentDirection()), half },
        to: goal,
      };
    }
    needsRender = true;
  }

  // ------------------------------------------------------------ appearance
  const contourMode = () => chapter === 2 && Boolean(joint);
  function appearance() {
    const dark = document.documentElement.dataset.theme === "dark",
      th = THEMES[dark ? "dark" : "light"];
    ghostColor.value.copy(rgb(th.ghost));
    post.uniforms.uGhostInk.value.copy(rgb(th.ghostInk));
    post.uniforms.uContourInk.value.copy(rgb(th.ghostLine));
    post.uniforms.uHalo.value.copy(rgb(th.halo));
    post.uniforms.uHaloA.value = th.haloAlpha;
    post.uniforms.uContour.value = contourMode() ? 1 : 0;
    post.uniforms.uMoveA.value = dark ? 0.5 : 0.42;
    shaftCasing.material.uniforms.uColor.value.copy(rgb(th.shaftEdge));
    post.uniforms.uShaftInk.value.copy(rgb(th.shaftEdge));
    for (const p of parts) {
      const u = p.toon.uniforms;
      let ghost = 0,
        tint = null,
        mix = 0,
        pulse = 0;
      p.layer = "solid";
      if (contourMode()) {
        if (p.id === joint.servo) [tint, mix] = [ACCENT.actuator, 1];
        else if (hoverMotor && p.id === hoverMotor.servo) [tint, mix, pulse] = [ACCENT.actuator, 0.5, 0.3]; // pale coral: click to drive it
        else if (motorJoint.has(p.id) && joint.movingSet.has(p.id)); // another motor carried along: solid, seen through the shells
        else if (joint.bracketSet.has(p.id)) [tint, mix] = [ACCENT.link, 1];
        else p.layer = joint.movingSet.has(p.id) ? "moving" : "still";
      } else if (chapter !== 2 && focus && !focus.memberSet.has(p)) ghost = 0.82;
      else if (chapter !== 2 && focus?.servos?.has(p.id)) [tint, mix] = [ACCENT.actuator, 1];
      if (hovered && p.group === hovered && chapter !== 2) pulse = Math.max(pulse, focus?.memberSet.has(p) ? 0.04 : 0.12);
      if (clickedPart === p && focus) pulse = Math.max(pulse, 0.1);
      if (tint) u.uTint.value.copy(tint);
      u.uTintMix.value = mix;
      u.uGhost.value = ghost;
      u.uPulse.value = pulse;
      p.idMaterial.uniforms.uId.value = p.index + (ghost > 0.5 ? 128 : 0);
    }
  }
  const isPale = (p) => p.layer !== "solid" || p.toon.uniforms.uGhost.value > 0.5;

  // ------------------------------------------------------------ callouts
  const callouts = [];
  let figureSpan = null;
  function setCallouts(list) {
    calloutLayer.replaceChildren();
    callouts.length = 0;
    for (const item of list) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dx-callout" + (item.sub ? " has-sub" : "");
      b.innerHTML = `<span class="dx-callout-dot" aria-hidden="true"></span><span class="dx-callout-body"><span class="dx-callout-text"></span></span>`;
      b.querySelector(".dx-callout-text").textContent = item.text;
      if (item.sub) {
        const s = document.createElement("span");
        s.className = "dx-callout-sub";
        s.textContent = item.sub;
        b.querySelector(".dx-callout-body").append(s);
      }
      if (item.servo) b.classList.add("is-count");
      if (item.part) b.addEventListener("click", () => focusGroup(partFocus(item.part)));
      else if (item.group) b.addEventListener("click", () => focusGroup(groups.get(sided(item.group))));
      calloutLayer.append(b);
      callouts.push({ ...item, el: b });
    }
    needsRender = true;
  }
  // "{side}" in a group key is the leg nearer to the camera.
  const sided = (key) => key.replace("{side}", nearSide);
  const groupCentre = (key) => centreOf((p) => p.group.id === sided(key));
  // A sensor site, or a teaching marker, moves with the explode group of the part it sits on.
  const sitePoint = (s) => {
    const p = byId.get(s.near_part);
    return new THREE.Vector3(...s.pos).addScaledVector(p.offset, groupProgress(p.group, explodeT));
  };
  const partCentre = (id) => centreOf((p) => p.id === id);
  // A joint's pivot where its servo's explode group has moved it to.
  const jointPoint = (id) => {
    const j = jointById.get(id),
      servo = byId.get(j.servo);
    return j.pivotRest.clone().addScaledVector(servo.offset, groupProgress(servo.group, explodeT));
  };
  function chapterCallouts() {
    if (focus || chapter === 2) return setCallouts([]);
    const label = (id) => tx(partInfo(id), "label"),
      body = { group: "shells", text: T.body, anchor: () => groupCentre("shells") };
    if (chapter === 0)
      // Six parts; the knee label points at the knee joint itself.
      return setCallouts([
        { part: "head", text: label("head"), anchor: () => groupCentre("head") },
        { part: "neck", text: label("neck"), anchor: () => groupCentre("neck") },
        body,
        { part: "hip", text: label("hip"), anchor: () => groupCentre("{side}-hip") },
        { part: "knee", text: label("knee"), anchor: () => jointPoint(functional.get("knee")[nearSide].joints[0]) },
        { part: "foot", text: label("foot"), anchor: () => groupCentre("{side}-foot") },
      ]);
    const counts = drive.functional.map((f) => ({
      part: f.id,
      text: `${label(f.id)} · ${T.servos(owned(f).servos.length, f.per_side)}`,
      sub: f.fixed ? T.fixed : motionList(f),
      servo: true,
      anchors: () => owned(f).servos.map(partCentre),
    }));
    setCallouts([
      { group: "face", text: T.eye, anchor: () => sitePoint(drive.sensors.head_camera) },
      // The ToF sensor sits in a window of the face board beside the eye, so its marker follows that board.
      { group: "face", text: T.tof, glyph: true, anchor: () => sitePoint(drive.annotations.tof) },
      { group: "boards", text: T.board, anchor: () => groupCentre("boards") },
      { group: "battery", text: T.battery, anchor: () => groupCentre("battery") },
      body,
      ...counts,
    ]);
  }
  function centreOf(selector) {
    const box = new THREE.Box3(),
      b = new THREE.Box3();
    for (const p of parts) if (selector(p)) box.union(b.copy(p.object.geometry.boundingBox).applyMatrix4(p.object.matrixWorld));
    return box.getCenter(new THREE.Vector3());
  }
  const project = (v) => {
    const s = v.clone().project(camera);
    return [(s.x * 0.5 + 0.5) * stage.clientWidth, (-s.y * 0.5 + 0.5) * stage.clientHeight];
  };
  function updateNearSide() {
    const depth = (side) => centreOf((p) => p.group.id === `${side}-thigh`).project(camera).z;
    nearSide = depth("left") < depth("right") ? "left" : "right";
  }
  // Labels sit in side columns outside the figure. Single leaders run from a dot on the part;
  // servo-count leaders start at every servo housing and merge at one junction before the label.
  function placeCallouts() {
    if (!callouts.length) {
      leaders.replaceChildren();
      return;
    }
    const W = stage.clientWidth,
      H = stage.clientHeight;
    let fx0 = Infinity,
      fx1 = -Infinity;
    const pts = [];
    for (const p of parts) if (!isPale(p)) corners(p, pts);
    for (const v of pts) {
      const [x] = project(v);
      fx0 = Math.min(fx0, x);
      fx1 = Math.max(fx1, x);
    }
    figureSpan = [fx0, fx1];
    const mid = (fx0 + fx1) / 2,
      items = callouts.map((c) => {
        const anchors = (c.anchors ? c.anchors() : [c.anchor()]).map(project),
          ax = anchors.reduce((s, a) => s + a[0], 0) / anchors.length,
          ay = anchors.reduce((s, a) => s + a[1], 0) / anchors.length;
        return { c, anchors, ax, ay, w: c.el.offsetWidth, h: c.el.offsetHeight, left: ax < mid };
      });
    // Keep both columns in use when one side would be crowded.
    for (const side of [true, false]) {
      const col = items.filter((i) => i.left === side);
      const other = items.filter((i) => i.left !== side);
      while (col.length > Math.ceil(items.length / 2) && other.length < col.length - 1) {
        const move = col.reduce((best, i) => (Math.abs(i.ax - mid) < Math.abs(best.ax - mid) ? i : best));
        move.left = !side;
        col.splice(col.indexOf(move), 1);
        other.push(move);
      }
    }
    const pad = 10,
      gap = 6;
    let svg = "";
    const f = (v) => v.toFixed(1);
    for (const side of [true, false]) {
      const col = items.filter((i) => i.left === side).sort((a, b) => a.ay - b.ay);
      let y = pad;
      for (const i of col) {
        i.y = Math.max(y, Math.min(i.ay - i.h / 2, H - pad - i.h));
        y = i.y + i.h + gap;
      }
      // Pull up from the bottom if the column overflows.
      let bottom = H - pad;
      for (const i of [...col].reverse()) {
        if (i.y + i.h > bottom) i.y = bottom - i.h;
        bottom = i.y - gap;
      }
      for (const i of col) {
        const reach = side ? Math.min(...i.anchors.map((a) => a[0])) : Math.max(...i.anchors.map((a) => a[0]));
        const x = side ? Math.max(pad, Math.min(fx0 - 22 - i.w, reach - 40 - i.w)) : Math.min(W - pad - i.w, Math.max(fx1 + 22, reach + 40));
        i.c.el.style.transform = `translate(${Math.round(x)}px, ${Math.round(i.y)}px)`;
        i.c.el.classList.toggle("is-right", !side);
        const ly = i.y + i.h / 2,
          lx = side ? x + i.w : x,
          elbow = side ? lx + 10 : lx - 10,
          cls = `dx-leader${i.c.servo ? " is-servo" : ""}`,
          label = i.c.text.replace(/"/g, "&quot;");
        if (i.anchors.length > 1) {
          // Junction just outside the outermost housing, level between the housings and the label.
          const jx = reach + (elbow - reach) * 0.45,
            jy = i.ay * 0.6 + ly * 0.4;
          svg += `<path class="${cls}" fill="none" d="M${f(lx)} ${f(ly)}H${f(elbow)}L${f(jx)} ${f(jy)}"/>`;
          for (const [axp, ayp] of i.anchors) svg += `<path class="${cls} is-branch" fill="none" d="M${f(jx)} ${f(jy)}L${f(axp)} ${f(ayp)}"/>`;
          svg += `<circle class="dx-junction" cx="${f(jx)}" cy="${f(jy)}" r="2.6"/>`;
          for (const [axp, ayp] of i.anchors) svg += `<circle class="dx-anchor is-servo" data-label="${label}" cx="${f(axp)}" cy="${f(ayp)}" r="3.6"/>`;
        } else {
          svg += `<path class="${cls}" fill="none" d="M${f(lx)} ${f(ly)}H${f(elbow)}L${f(i.ax)} ${f(i.ay)}"/>`;
          svg += i.c.glyph
            ? `<g class="dx-site" data-label="${label}"><circle cx="${f(i.ax)}" cy="${f(i.ay)}" r="5.5"/><path d="M${f(i.ax - 8)} ${f(i.ay)}H${f(i.ax + 8)}M${f(i.ax)} ${f(i.ay - 8)}V${f(i.ay + 8)}"/></g>`
            : `<circle class="dx-anchor${i.c.servo ? " is-servo" : ""}" data-label="${label}" cx="${f(i.ax)}" cy="${f(i.ay)}" r="4.5"/>`;
        }
      }
    }
    leaders.setAttribute("viewBox", `0 0 ${W} ${H}`);
    leaders.innerHTML = svg;
  }

  // ------------------------------------------------------------ joint gizmo (joint view)
  // Shaft: on the hinge axis, centred on the plane that halves the servo housing across the axis.
  // The housing's vertices are projected on the axis in the rest pose and the middle of their
  // min/max is the centre; it is stored in the housing's own frame, so it moves with the housing
  // (also for servos mounted in the moving link). The shaft is symmetric about that centre and
  // reaches just past the servo, bracket and moving parts near the axis on the chosen side.
  // Semicircle: 180 degrees in the rotation plane, centred on the axis a little beyond that shaft
  // end, bulging toward a world direction at STAND (drive.gizmo), and turning with the child link.
  const gizmoCache = new Map();
  let jg = null; // the selected joint's gizmo geometry
  function gizmoFor(j) {
    if (gizmoCache.has(j.id)) return gizmoCache.get(j.id);
    applyPose(0, null);
    const a = j.axisRest,
      servo = byId.get(j.servo),
      v = new THREE.Vector3(),
      each = (part, fn) => {
        const pos = part.object.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) fn(v.fromBufferAttribute(pos, i).applyMatrix4(part.object.matrixWorld));
      };
    let t0 = Infinity,
      t1 = -Infinity;
    each(servo, (w) => {
      const t = w.sub(j.pivotRest).dot(a);
      t0 = Math.min(t0, t);
      t1 = Math.max(t1, t);
    });
    const centre = j.pivotRest.clone().addScaledVector(a, (t0 + t1) / 2),
      cfg = drive.gizmo[j.id],
      side = Math.sign(a.dot(new THREE.Vector3(...cfg.end))) || 1;
    // Base radius: around the servo and bracket (their farthest point from the axis) with some room,
    // 30-42 mm, or as set for the joint (head yaw circles the whole head). The ribbon is drawn
    // RIBBON_RADIUS_SCALE times larger in radius and RIBBON_WIDTH_SCALE times wider; the shaft keeps
    // its base-size length.
    let reach = 0;
    for (const part of parts)
      if (part.id === j.servo || j.bracketSet.has(part.id))
        each(part, (w) => {
          const r = w.sub(j.pivotRest);
          reach = Math.max(reach, r.addScaledVector(a, -r.dot(a)).length());
        });
    const baseRadius = cfg.radius ?? clamp(1.2 * reach, 0.03, 0.042),
      radius = RIBBON_RADIUS_SCALE * baseRadius,
      width = RIBBON_WIDTH_SCALE * clamp(0.15 * baseRadius, 0.0045, 0.0075); // ribbon width along the axis
    let ext = (t1 - t0) / 2;
    for (const part of parts)
      if (part.id === j.servo || j.bracketSet.has(part.id) || j.movingSet.has(part.id))
        each(part, (w) => {
          const r = w.sub(centre),
            t = r.dot(a);
          if (t * side > 0 && r.addScaledVector(a, -t).length() <= baseRadius + 0.004) ext = Math.max(ext, t * side);
        });
    const halfLength = Math.min(Math.max(ext + SHAFT_MARGIN, (t1 - t0) / 2 + SHAFT_EXPOSED) + SHAFT_EXTRA, 0.085),
      u = new THREE.Vector3(...cfg.toward);
    u.addScaledVector(a, -u.dot(a)).normalize();
    const g = {
      axial: [t0, t1], // servo housing along the axis, from the rest pivot (m)
      centreRest: centre,
      centreLocal: centre.clone().applyMatrix4(servo.object.matrixWorld.clone().invert()),
      halfLength,
      side,
      baseRadius,
      radius,
      u,
      w: new THREE.Vector3().crossVectors(a, u), // u -> w is the positive (right-hand) direction
      arcCentreRest: centre.clone().addScaledVector(a, side * (halfLength + ARC_GAP)),
      width,
      headWidth: 3 * width, // arrowhead base, across the ribbon
      headLength: 2.3 * width, // arrowhead length along the circle
    };
    g.profile = ribbonProfile(g);
    g.geometry = ribbonGeometry(g, a);
    gizmoCache.set(j.id, g);
    applyPose();
    return g;
  }
  // The ribbon's outline in (angle, axial half-width): an arrowhead at both ends of a 180-degree
  // band, tips on the diameter. Repeated angles at the arrowhead bases make the shoulder steps.
  function ribbonProfile(g) {
    const head = g.headLength / g.radius,
      a0 = -Math.PI / 2,
      a1 = Math.PI / 2,
      out = [],
      n = 36;
    for (let k = 0; k <= 4; k++) out.push([a0 + (head * k) / 4, (g.headWidth / 2) * (k / 4)]);
    out.push([a0 + head, g.width / 2]);
    for (let k = 1; k < n; k++) out.push([a0 + head + ((a1 - a0 - 2 * head) * k) / n, g.width / 2]);
    out.push([a1 - head, g.width / 2]);
    for (let k = 4; k >= 0; k--) out.push([a1 - (head * k) / 4, (g.headWidth / 2) * (k / 4)]);
    return out;
  }
  // A paper band on the cylinder of radius g.radius about the hinge axis (coaxial with the shaft):
  // outer and inner faces RIBBON_T apart and the cut edges between them, in rest coordinates. Every
  // face is wound counter-clockwise seen from outside the solid, and its normals point outwards:
  // radially out on the outer face, towards the axis on the inner face (smooth along the curve).
  function ribbonGeometry(g, a) {
    const pos = [],
      nor = [],
      surf = [],
      R = (t) => g.u.clone().multiplyScalar(Math.cos(t)).addScaledVector(g.w, Math.sin(t)),
      P = (t, z, r) => g.arcCentreRest.clone().addScaledVector(R(t), r).addScaledVector(a, z),
      // v0..v3 counter-clockwise from outside; n gives the normal at each corner
      quad = (v, n, s) => {
        for (const i of [0, 1, 2, 0, 2, 3]) {
          pos.push(v[i].x, v[i].y, v[i].z);
          nor.push(n[i].x, n[i].y, n[i].z);
          surf.push(s);
        }
      },
      ro = g.radius + RIBBON_T / 2,
      ri = g.radius - RIBBON_T / 2,
      prof = g.profile;
    for (let i = 0; i + 1 < prof.length; i++) {
      const [t0, h0] = prof[i],
        [t1, h1] = prof[i + 1],
        n0 = R(t0),
        n1 = R(t1),
        m0 = n0.clone().negate(),
        m1 = n1.clone().negate();
      if (t1 > t0) {
        // the repeated angles at the arrowhead shoulders carry only the cut edge
        quad([P(t0, h0, ro), P(t0, -h0, ro), P(t1, -h1, ro), P(t1, h1, ro)], [n0, n0, n1, n1], 0);
        quad([P(t0, h0, ri), P(t1, h1, ri), P(t1, -h1, ri), P(t0, -h0, ri)], [m0, m1, m1, m0], 1);
      }
      for (const s of [1, -1]) {
        const v = [P(t0, s * h0, ro), P(t1, s * h1, ro), P(t1, s * h1, ri), P(t0, s * h0, ri)];
        // cut edge: perpendicular to the radial direction and the edge; the s = -1 side mirrors the
        // s = +1 side, so its winding and normal are both reversed
        const n = new THREE.Vector3().crossVectors(v[1].clone().sub(v[0]), v[3].clone().sub(v[0])).normalize();
        if (s < 0) n.negate();
        quad(s > 0 ? v : [v[0], v[3], v[2], v[1]], [n, n, n, n], 2);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute("aSurface", new THREE.Float32BufferAttribute(surf, 1));
    geo.computeBoundingSphere();
    return geo;
  }
  // Ribbon outline points at the current pose (mid-thickness), for the hit area and framing.
  function ribbonOutline() {
    if (!joint || !jg) return [];
    const a = jg.arcCentreRest,
      P = (t, z) => a.clone().addScaledVector(jg.u, Math.cos(t) * jg.radius).addScaledVector(jg.w, Math.sin(t) * jg.radius).addScaledVector(joint.axisRest, z).applyMatrix4(joint.world);
    return [...jg.profile.map(([t, h]) => P(t, h)), ...[...jg.profile].reverse().map(([t, h]) => P(t, -h))];
  }
  // Shaft at the current pose: the housing's centre carried by the housing, on the current axis.
  function shaftNow() {
    const p = jointPivot(joint),
      a = jointAxis(joint),
      c = jg.centreLocal.clone().applyMatrix4(byId.get(joint.servo).object.matrixWorld),
      centre = p.addScaledVector(a, c.sub(p).dot(a));
    return { centre, axis: a, ends: [-1, 1].map((k) => centre.clone().addScaledVector(a, k * jg.halfLength)) };
  }
  const arcCentre = () => jg.arcCentreRest.clone().applyMatrix4(joint.world);
  // Semicircle points at the current pose, from -90 to +90 degrees (positive q direction).
  function arcPoints(n = 48) {
    if (!joint || !jg) return [];
    const out = [];
    for (let i = 0; i <= n; i++) {
      const t = -Math.PI / 2 + (Math.PI * i) / n;
      out.push(
        jg.arcCentreRest
          .clone()
          .addScaledVector(jg.u, Math.cos(t) * jg.radius)
          .addScaledVector(jg.w, Math.sin(t) * jg.radius)
          .applyMatrix4(joint.world),
      );
    }
    return out;
  }
  const yAxis = new THREE.Vector3(0, 1, 0);
  function placeShaft() {
    const on = contourMode() && Boolean(jg);
    shaftCore.visible = shaftCasing.visible = shaftMask.visible = ribbon.visible = ribbonMask.visible = on;
    if (!on) return;
    ribbon.geometry = ribbonMask.geometry = jg.geometry;
    for (const m of [ribbon, ribbonMask]) {
      m.matrix.copy(joint.world); // the ribbon turns with the child link
      m.updateMatrixWorld(true);
    }
    const { centre, axis } = shaftNow(),
      px = (2 * half) / Math.max(1, stage.clientHeight * dpr()); // one device pixel in metres
    for (const [m, r] of [
      [shaftCore, SHAFT_R],
      [shaftCasing, SHAFT_R + 1.1 * px],
      [shaftMask, SHAFT_R + 2.2 * px],
    ]) {
      m.position.copy(centre);
      m.quaternion.setFromUnitVectors(yAxis, axis);
      m.scale.set(r, 2 * jg.halfLength, r);
      m.updateMatrixWorld(true);
    }
  }
  function drawGizmo() {
    // SVG elements have no `hidden` property; toggle the attribute itself.
    gizmo.toggleAttribute("hidden", !contourMode());
    if (!contourMode()) {
      handle.hidden = true;
      return;
    }
    // The ribbon itself is drawn in 3D; the SVG holds only a transparent hit area projected from
    // its exact outline (arrowheads included), widened a little for thin, oblique views.
    const W = stage.clientWidth,
      H = stage.clientHeight,
      pts = arcPoints(48).map(project),
      d = ribbonOutline()
        .map(project)
        .map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`)
        .join("");
    gizmo.setAttribute("viewBox", `0 0 ${W} ${H}`);
    if (!gizmo.firstChild) {
      // Built once so a pointer capture on the hit path survives redraws during a drag. The inline
      // style keeps it invisible and grabbable across its whole area whatever stylesheet is loaded
      // (an unstyled closed SVG path would be filled black).
      gizmo.innerHTML = `<path class="dx-arc-hit" fill="none" stroke="none" stroke-width="12" stroke-linejoin="round" style="opacity:0;pointer-events:all"/>`;
      gizmo.firstChild.addEventListener("pointerdown", startDrag);
    }
    gizmo.firstChild.setAttribute("d", `${d}Z`);
    const [hx, hy] = pts[24];
    handle.hidden = false;
    handle.style.transform = `translate(${Math.round(hx)}px, ${Math.round(hy)}px)`;
    handle.classList.toggle("is-dragging", Boolean(drag));
  }

  // ------------------------------------------------------------ direct drag on the arc
  let drag = null;
  const ray = new THREE.Raycaster(),
    ndc = new THREE.Vector2();
  function planeAngle(e) {
    // Angle of the pointer about the joint axis, measured in the rotation plane.
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(drag.axis, drag.pivot), new THREE.Vector3());
    if (!hit) return null;
    const v = hit.sub(drag.pivot);
    return Math.atan2(v.dot(drag.w), v.dot(drag.u));
  }
  function startDrag(e) {
    if (!joint || e.button > 0) return;
    e.preventDefault();
    e.stopPropagation();
    stopPlay();
    const axis = jointAxis(joint),
      pivot = arcCentre(), // the drag plane is the semicircle's own plane
      view = camera.getWorldDirection(new THREE.Vector3()),
      u = new THREE.Vector3(0, 1, 0).cross(axis);
    if (u.lengthSq() < 1e-6) u.set(1, 0, 0).cross(axis);
    u.normalize();
    drag = {
      id: e.pointerId,
      el: e.currentTarget,
      axis,
      pivot,
      u,
      w: new THREE.Vector3().crossVectors(axis, u),
      q0: q.get(joint.id),
      x0: e.clientX,
      y0: e.clientY,
      // Nearly edge-on planes make the pointer angle unstable: use the arc's screen tangent instead.
      planar: Math.abs(view.dot(axis)) > 0.26,
      turned: 0,
    };
    if (drag.planar) drag.last = planeAngle(e);
    if (drag.last == null) drag.planar = false;
    if (!drag.planar) {
      // Screen direction of positive rotation at the grabbed point, and pixels per radian.
      const pts = arcPoints(48),
        k = nearestArcIndex(pts.map(project), e),
        rel = pts[k].clone().sub(pivot),
        tangent = new THREE.Vector3().crossVectors(axis, rel),
        [x0, y0] = project(pts[k]),
        [x1, y1] = project(pts[k].clone().add(tangent.multiplyScalar(0.05)));
      let tx = (x1 - x0) / 0.05,
        ty = (y1 - y0) / 0.05,
        len = Math.hypot(tx, ty);
      if (len < 12) {
        // Tangent points at the viewer here: fall back to the arc's widest screen tangent.
        const [px, py] = project(pivot);
        tx = -(y0 - py) || 1;
        ty = x0 - px;
        len = Math.hypot(tx, ty);
      }
      drag.tangent = [tx / len, ty / len, Math.max(len, 12)];
    }
    drag.el.setPointerCapture?.(e.pointerId);
    stopOrbitInertia();
    controls.enabled = false;
    root.classList.add("is-dragging");
    handle.focus({ preventScroll: true });
    window.addEventListener("pointermove", moveDrag);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);
    wake();
  }
  function nearestArcIndex(screen, e) {
    const r = stage.getBoundingClientRect(),
      x = e.clientX - r.left,
      y = e.clientY - r.top;
    let best = 0;
    screen.forEach(([sx, sy], i) => {
      if (Math.hypot(sx - x, sy - y) < Math.hypot(screen[best][0] - x, screen[best][1] - y)) best = i;
    });
    return best;
  }
  function moveDrag(e) {
    if (!drag || e.pointerId !== drag.id) return;
    let turned;
    if (drag.planar) {
      const a = planeAngle(e);
      if (a == null) return;
      let d = a - drag.last;
      d -= Math.round(d / (2 * Math.PI)) * 2 * Math.PI;
      drag.last = a;
      drag.turned += d;
      turned = drag.turned;
    } else {
      const [tx, ty, pxPerRad] = drag.tangent;
      turned = ((e.clientX - drag.x0) * tx + (e.clientY - drag.y0) * ty) / pxPerRad;
    }
    setAngle(drag.q0 + turned);
  }
  function endDrag(e) {
    if (!drag || (e && e.pointerId !== drag.id)) return;
    if (drag.el.hasPointerCapture?.(drag.id)) drag.el.releasePointerCapture(drag.id);
    drag = null;
    controls.enabled = true;
    root.classList.remove("is-dragging");
    window.removeEventListener("pointermove", moveDrag);
    window.removeEventListener("pointerup", endDrag);
    window.removeEventListener("pointercancel", endDrag);
    needsRender = true;
    wake();
  }
  handle.addEventListener("pointerdown", startDrag, { signal });
  handle.addEventListener(
    "keydown",
    (e) => {
      if (!joint) return;
      const step = { ArrowRight: 1, ArrowUp: 1, "+": 1, "=": 1, ArrowLeft: -1, ArrowDown: -1, "-": -1, PageUp: 10, PageDown: -10 }[e.key];
      let next = null;
      if (step) next = q.get(joint.id) + ((e.shiftKey && e.key.startsWith("Arrow") ? 5 : 1) * step) / DEG;
      else if (e.key === "Home") next = joint.range_rad[0];
      else if (e.key === "End") next = joint.range_rad[1];
      if (next === null) return;
      e.preventDefault();
      stopPlay();
      setAngle(next);
    },
    { signal },
  );

  // ------------------------------------------------------------ joint state and readout
  const fmt = (rad) => {
    const v = Math.round(rad * DEG * 10) / 10 || 0;
    return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}°`;
  };
  function setAngle(value) {
    const [lo, hi] = joint.range_rad;
    q.set(joint.id, clamp(value, lo, hi));
    applyPose();
    updateReadout();
    needsRender = true;
    wake();
  }
  const jointName = (j) => (j.side ? `${zh ? (j.side === "left" ? "左" : "右") : j.side === "left" ? "left " : "right "}` : "") + tx(j, "label");
  function updateReadout() {
    if (!joint) return;
    const v = q.get(joint.id),
      [lo, hi] = joint.range_rad,
      atLimit = v <= lo + 1e-9 || v >= hi - 1e-9;
    readQ.innerHTML = `<var>${T.q}</var> = <output>${fmt(v)}</output>${atLimit ? `<span class="dx-limit">${T.limit}</span>` : ""}`;
    readDetail.textContent = `${T.stand} ${fmt(joint.stand_rad)} · ${T.delta} ${fmt(v - joint.stand_rad)} · ${T.range} ${fmt(lo)} ${T.to} ${fmt(hi)}`;
    readId.textContent = joint.id;
    handle.setAttribute("aria-valuemin", (lo * DEG).toFixed(1));
    handle.setAttribute("aria-valuemax", (hi * DEG).toFixed(1));
    handle.setAttribute("aria-valuenow", (v * DEG).toFixed(1));
    handle.setAttribute("aria-valuetext", `q = ${fmt(v)}`);
    handle.setAttribute("aria-label", T.handle(jointName(joint)));
  }

  // ------------------------------------------------------------ text panel
  function showText() {
    backButton.hidden = !focus;
    root.dataset.chapter = String(chapter);
    card.hidden = chapter === 2;
    drivePanel.hidden = chapter !== 2;
    const [name, intro] = T.chapters[chapter];
    if (focus) {
      title.textContent = tx(focus, "label");
      fact.textContent = tx(focus, "fact");
      const label = clickedPart && labels.get(clickedPart.id);
      partLine.textContent = label ? T.part + tx(label, "label") : "";
      partLine.hidden = !label;
    } else {
      title.textContent = name;
      fact.textContent = intro;
      partLine.hidden = true;
    }
    chapterButtons.forEach((b, i) => b.setAttribute("aria-pressed", String(i === chapter)));
  }

  // ------------------------------------------------------------ actions
  function animateExplode(to) {
    explodeFrom = explodeT;
    explodeTo = to;
    explodeStart = performance.now();
    if (reduceMotion.matches) {
      explodeT = to;
      applyPose();
    }
    needsRender = true;
  }
  function setChapter(i) {
    stopPlay();
    endDrag();
    setHoverMotor(null);
    chapter = i;
    focus = null;
    clickedPart = null;
    resetAngles();
    if (i === 2) {
      explodeT = explodeTo = 0; // joints are shown on the assembled duck
      selectRegion(region || drive.regions[0], true);
    } else animateExplode(i === 1 ? 1 : 0);
    applyPose();
    chapterCallouts();
    showText();
    moveCamera();
  }
  function focusGroup(g, part = null) {
    if (!g || chapter === 2) return;
    focus = g;
    clickedPart = part;
    chapterCallouts();
    showText();
    moveCamera();
  }
  function unfocus() {
    if (!focus) return false;
    focus = null;
    clickedPart = null;
    chapterCallouts();
    showText();
    moveCamera();
    return true;
  }
  function selectRegion(r, fromChapter = false, target = null) {
    region = r;
    [...regionButtons.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.region === r.id)));
    jointButtons.replaceChildren(
      ...r.joints.map((id) => {
        const j = jointById.get(id),
          b = document.createElement("button");
        b.type = "button";
        b.className = "dx-joint";
        b.dataset.joint = id;
        b.textContent = tx(j, "label");
        b.addEventListener("click", () => selectJoint(j));
        return b;
      }),
    );
    fixedNote.hidden = !r.fixed;
    if (r.fixed) {
      fixedNote.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = tx(r.fixed, "label");
      fixedNote.append(strong, ` ${tx(r.fixed, "note")}`);
    }
    selectJoint(target || jointById.get(r.joints.includes(joint?.id) ? joint.id : r.joints[0]), fromChapter);
  }
  function selectJoint(j, fromChapter = false) {
    stopPlay();
    endDrag();
    joint = j;
    [...jointButtons.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.joint === j.id)));
    jg = gizmoFor(j);
    updateReadout();
    if (!fromChapter) moveCamera();
    needsRender = true;
  }
  function startPlay() {
    if (!joint) return;
    const base = q.get(joint.id),
      [lo, hi] = joint.range_rad,
      amp = joint.amplitude_deg / DEG;
    playKeys = [
      [0, base],
      [0.9, clamp(base + amp, lo, hi)],
      [2.0, clamp(base - amp, lo, hi)],
      [2.9, base],
    ];
    playing = true;
    playStart = performance.now();
    playButton.textContent = T.pause;
    playButton.setAttribute("aria-pressed", "true");
    needsRender = true;
  }
  function stopPlay() {
    if (!playing) return;
    playing = false;
    playButton.textContent = T.play;
    playButton.setAttribute("aria-pressed", "false");
    needsRender = true;
  }

  // ------------------------------------------------------------ build controls
  regionButtons.replaceChildren(
    ...drive.regions.map((r) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "dx-region";
      b.dataset.region = r.id;
      b.textContent = tx(r, "label");
      b.addEventListener("click", () => selectRegion(r));
      return b;
    }),
  );
  const sideNote = document.createElement("p");
  sideNote.className = "dx-side-note";
  sideNote.textContent = drive.side_note[zh ? "zh" : "en"];
  jointButtons.closest(".dx-field").after(sideNote);
  const motorTip = document.createElement("p");
  motorTip.className = "dx-motor-tip";
  motorTip.setAttribute("aria-hidden", "true");
  motorTip.hidden = true;
  stage.append(motorTip);
  playButton.textContent = T.play;
  zeroButton.textContent = T.zero;
  chapterButtons.forEach((b, i) => b.addEventListener("click", () => setChapter(i), { signal }));
  resetButton.addEventListener(
    "click",
    () => {
      region = null;
      joint = null;
      setChapter(0);
    },
    { signal },
  );
  backButton.addEventListener("click", unfocus, { signal });
  playButton.addEventListener("click", () => (playing ? stopPlay() : startPlay()), { signal });
  zeroButton.addEventListener(
    "click",
    () => {
      stopPlay();
      resetAngles();
      applyPose();
      updateReadout();
      needsRender = true;
    },
    { signal },
  );
  root.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Escape" && unfocus()) {
        event.preventDefault();
        chapterButtons[chapter].focus();
      }
    },
    { signal },
  );

  // ------------------------------------------------------------ picking
  const raycaster = new THREE.Raycaster(),
    pointer = new THREE.Vector2();
  const setPointer = (x, y) => {
    pointer.set((x / stage.clientWidth) * 2 - 1, -(y / stage.clientHeight) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
  };
  const stageXY = (event) => {
    const r = renderer.domElement.getBoundingClientRect();
    return [event.clientX - r.left, event.clientY - r.top];
  };
  function pick(event) {
    setPointer(...stageXY(event));
    const candidates = parts.filter((p) => !isPale(p)).map((p) => p.object);
    const hit = raycaster.intersectObjects(candidates, false)[0];
    return hit ? byId.get(hit.object.userData.id) : null;
  }
  // Joint view: the servo housing under the pointer, if it drives one of the 9 motions. Solid parts
  // hide what is behind them. Outlined parts do not, and moving parts turn see-through wherever a
  // housing lies behind them, so every housing you can see stays clickable.
  function pickMotorAt(x, y) {
    setPointer(x, y);
    const candidates = parts.filter((p) => motorJoint.has(p.id) || p.layer === "solid").map((p) => p.object);
    const hit = raycaster.intersectObjects(candidates, false)[0];
    return (hit && motorJoint.get(hit.object.userData.id)) || null;
  }
  function selectMotor(j) {
    stopPlay();
    if (j !== joint) selectRegion(regionOf.get(j.id), false, j);
  }
  function setHoverMotor(j, xy = null) {
    renderer.domElement.style.cursor = j ? "pointer" : "";
    const show = j && j !== joint;
    motorTip.hidden = !show;
    if (show) {
      motorTip.textContent = `${tx(regionOf.get(j.id), "label")} · ${tx(j, "label")}`;
      motorTip.style.transform = `translate(${Math.round(xy[0] + 14)}px, ${Math.round(xy[1] + 12)}px)`;
    }
    if (j !== hoverMotor) {
      hoverMotor = j;
      needsRender = true;
    }
  }
  let down = null,
    hoverPending = false;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    down = [e.clientX, e.clientY];
    motorTip.hidden = true;
    // Keep keyboard focus inside the figure so Escape works after clicking the model.
    if (!root.contains(document.activeElement)) stage.focus({ preventScroll: true });
  });
  stage.tabIndex = -1;
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || e.button > 0) return;
    if (chapter === 2) {
      const j = contourMode() ? pickMotorAt(...stageXY(e)) : null;
      if (j) {
        selectMotor(j);
        setHoverMotor(j, stageXY(e));
      }
      return;
    }
    const p = pick(e);
    if (p) focusGroup(p.group, p);
    else if (focus) unfocus();
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (e.buttons || hoverPending || drag) return;
    hoverPending = true;
    requestAnimationFrame(() => {
      hoverPending = false;
      if (chapter === 2) {
        setHoverMotor(contourMode() ? pickMotorAt(...stageXY(e)) : null, stageXY(e));
        return;
      }
      const p = pick(e),
        g = p ? p.group : null;
      renderer.domElement.style.cursor = p ? "pointer" : "";
      if (g !== hovered) {
        hovered = g;
        needsRender = true;
      }
    });
  });
  renderer.domElement.addEventListener("pointerleave", () => {
    if (hovered) needsRender = true;
    hovered = null;
    setHoverMotor(null);
  });
  controls.addEventListener("change", () => (needsRender = true));

  // ------------------------------------------------------------ loop
  function resize() {
    const w = stage.clientWidth,
      h = stage.clientHeight,
      r = dpr();
    renderer.setPixelRatio(r);
    renderer.setSize(w, h, false);
    resizeTargets(Math.round(w * r), Math.round(h * r));
    moveCamera(true);
  }
  function draw(target, material, show) {
    renderer.setRenderTarget(target);
    renderer.clear();
    for (const p of parts) {
      p.object.material = material(p);
      p.object.visible = show(p);
    }
    renderer.render(scene, camera);
  }
  const toonOf = (p) => p.toon,
    idOf = (p) => p.idMaterial;
  function render() {
    const solid = (p) => p.layer === "solid";
    draw(colorTarget, toonOf, solid);
    draw(idTarget, idOf, solid);
    if (contourMode()) {
      draw(moveColor, toonOf, (p) => p.layer === "moving");
      draw(moveId, idOf, (p) => p.layer === "moving");
      draw(ghostTarget, idOf, (p) => p.layer === "still");
      placeShaft();
      renderer.setRenderTarget(shaftColor);
      renderer.clear();
      renderer.render(shaftScene, camera);
      renderer.setRenderTarget(shaftId);
      renderer.clear();
      renderer.render(maskScene, camera);
    }
    frameNo++;
    for (const p of parts) {
      p.object.material = p.toon;
      p.object.visible = true;
    }
    renderer.setRenderTarget(null);
    renderer.clear();
    renderer.render(postScene, postCamera);
  }
  let pausedAt = null;
  function frame(now) {
    if (dead) return (running = false);
    if (!visible) {
      // Off screen or hidden tab: stop drawing and remember where every animation was.
      pausedAt ??= now;
      return (running = false);
    }
    if (pausedAt !== null) {
      const away = now - pausedAt;
      playStart += away;
      explodeStart += away;
      if (camTween) camTween.start += away;
      pausedAt = null;
    }
    let busy = false;
    if (explodeT !== explodeTo) {
      const u = clamp01((now - explodeStart) / (DUR * 1000));
      explodeT = explodeFrom + (explodeTo - explodeFrom) * u;
      if (u >= 1) explodeT = explodeTo;
      applyPose();
      busy = true;
    }
    if (camTween) {
      const u = easeInOut((now - camTween.start) / 900),
        { from, to } = camTween;
      const dir = from.direction.clone().lerp(to.direction, u).normalize();
      setCamera(from.target.clone().lerp(to.target, u), dir, from.half + (to.half - from.half) * u);
      if (u >= 1) camTween = null;
      busy = true;
    }
    if (playing && joint) {
      const phase = (now - playStart) / 1000,
        end = playKeys[playKeys.length - 1][0];
      q.set(joint.id, sampleKeys(playKeys, Math.min(phase, end)));
      applyPose();
      updateReadout();
      if (phase >= end) stopPlay();
      busy = true;
    }
    if (controls.update()) busy = true;
    if (busy || needsRender) {
      if (chapter !== 2) updateNearSide();
      appearance();
      render();
      placeCallouts();
      drawGizmo();
      needsRender = false;
    }
    requestAnimationFrame(frame);
  }
  let running = false;
  const wake = () => {
    needsRender = true;
    if (!running && visible) {
      running = true;
      requestAnimationFrame(frame);
    }
  };
  // Pause the loop while the figure is off screen or the tab is hidden.
  let onScreen = true;
  const updateVisible = () => {
    visible = onScreen && !document.hidden;
    wake();
  };
  const observers = [
    new IntersectionObserver(([entry]) => {
      onScreen = entry.isIntersecting;
      updateVisible();
    }),
    new MutationObserver(wake),
    new ResizeObserver(() => {
      resize();
      wake();
    }),
  ];
  undo.push(() => observers.forEach((o) => o.disconnect()));
  observers[0].observe(stage);
  observers[1].observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  observers[2].observe(stage);
  document.addEventListener("visibilitychange", updateVisible, { signal });
  const kick = () => wake();
  for (const el of [stage, $(".dx-panel"), $(".dx-chapters")])
    for (const type of ["pointerdown", "pointermove", "click", "keydown", "input"]) el.addEventListener(type, kick, { signal });

  $(".dx-chapters").hidden = false;
  $(".dx-panel").hidden = false;
  resize();
  // Compile the shaft materials up front so the first joint view starts without a hitch.
  shaftCore.visible = shaftCasing.visible = shaftMask.visible = true;
  renderer.compile(shaftScene, camera);
  renderer.compile(maskScene, camera);
  placeShaft();
  setChapter(0);
  moveCamera(true);
  wake();
  // Read-only geometry diagnostics for automated checks (nothing here changes the figure).
  function gizmoDiagnostics() {
    if (!contourMode() || !jg) return null;
    const sh = shaftNow(),
      a = sh.axis,
      servo = byId.get(joint.servo).object,
      pos = servo.geometry.attributes.position,
      v = new THREE.Vector3();
    let t0 = Infinity,
      t1 = -Infinity;
    for (let i = 0; i < pos.count; i++) {
      const t = v.fromBufferAttribute(pos, i).applyMatrix4(servo.matrixWorld).sub(sh.centre).dot(a);
      t0 = Math.min(t0, t);
      t1 = Math.max(t1, t);
    }
    const pts = arcPoints(12),
      m = new THREE.Matrix3().setFromMatrix4(joint.world);
    return {
      shaft: {
        center: sh.centre.toArray(),
        ends: sh.ends.map((e) => e.toArray()),
        axis: a.toArray(),
        radius: SHAFT_R,
        opacity: post.uniforms.uShaftA.value,
        halfLength: jg.halfLength,
        housingAxial: [t0, t1], // current housing extent along the axis, measured from the shaft centre
        restAxial: jg.axial, // rest housing extent along the axis, measured from the rest pivot
        screen: sh.ends.map(project),
      },
      arcGeometry: {
        center: arcCentre().toArray(),
        axis: a.toArray(),
        radius: jg.radius,
        baseRadius: jg.baseRadius,
        span: Math.PI, // a semicircle; its ends are not joint limits
        ribbon: {
          width: jg.width,
          headWidth: jg.headWidth,
          headLength: jg.headLength,
          thickness: RIBBON_T,
          opacity: post.uniforms.uShaftA.value,
          triangles: jg.geometry.attributes.position.count / 3,
          // triangles per surface (outer, inner, cut edge) whose winding agrees with their normals
          winding: (() => {
            const pos = jg.geometry.attributes.position,
              nor = jg.geometry.attributes.normal,
              surf = jg.geometry.attributes.aSurface,
              v = [0, 1, 2].map(() => new THREE.Vector3()),
              n = new THREE.Vector3(),
              out = [0, 1, 2].map(() => ({ agree: 0, disagree: 0, degenerate: 0 }));
            for (let i = 0; i < pos.count; i += 3) {
              v.forEach((p, k) => p.fromBufferAttribute(pos, i + k));
              const c = new THREE.Vector3().crossVectors(v[1].sub(v[0]), v[2].sub(v[0])),
                o = out[surf.getX(i)];
              n.set(0, 0, 0);
              for (let k = 0; k < 3; k++) n.add(new THREE.Vector3().fromBufferAttribute(nor, i + k));
              if (c.length() < 1e-10) o.degenerate++;
              else if (c.dot(n) > 0) o.agree++;
              else o.disagree++;
            }
            return { outer: out[0], inner: out[1], edge: out[2] };
          })(),
          outline: ribbonOutline().map((v) => v.toArray()),
          // ~120 vertices sampled from the actual position buffer, carried by the current pose
          vertices: (() => {
            const pos = jg.geometry.attributes.position,
              step = Math.max(1, Math.floor(pos.count / 120)),
              out = [];
            for (let i = 0; i < pos.count; i += step) out.push(new THREE.Vector3().fromBufferAttribute(pos, i).applyMatrix4(joint.world).toArray());
            return out;
          })(),
          hitScreen: ribbonOutline().map(project),
        },
        side: jg.side,
        toward: jg.u.clone().applyMatrix3(m).normalize().toArray(),
        points: pts.map((p) => p.toArray()),
        screen: pts.map(project),
      },
    };
  }
  // Each of the 9 servo housings: centre, and a screen point where a click selects it (null if the
  // housing is hidden from this view). Cached until the next frame is drawn.
  let motorCache = null;
  function motorDiagnostics() {
    if (motorCache?.frame === frameNo) return motorCache.list;
    const W = stage.clientWidth,
      H = stage.clientHeight,
      hit = gizmo.firstChild && contourMode() ? gizmo.firstChild : null, // the ribbon's drag area
      arcScreen = contourMode() ? ribbonOutline().map(project) : [],
      onRibbon = (x, y) => {
        if (!hit) return false;
        const pt = gizmo.createSVGPoint();
        pt.x = x;
        pt.y = y;
        return hit.isPointInFill(pt) || hit.isPointInStroke(pt);
      },
      v = new THREE.Vector3();
    const ok = ([x, y], j) =>
      x > 6 && y > 6 && x < W - 6 && y < H - 6 && !onRibbon(x, y) && arcScreen.every(([ax, ay]) => Math.hypot(ax - x, ay - y) > 10) && pickMotorAt(x, y) === j;
    const list = enabled.map((j) => {
      const part = byId.get(j.servo),
        centre = centreOf((p) => p === part),
        [cx, cy] = project(centre),
        pos = part.object.geometry.attributes.position,
        step = Math.max(1, Math.floor(pos.count / 80)),
        cands = [[cx, cy]];
      for (let i = 0; i < pos.count; i += step) {
        const [x, y] = project(v.fromBufferAttribute(pos, i).applyMatrix4(part.object.matrixWorld));
        cands.push([cx + (x - cx) * 0.7, cy + (y - cy) * 0.7]);
      }
      const solid = (c) => ok(c, j) && [[3, 0], [-3, 0], [0, 3], [0, -3]].every(([dx, dy]) => ok([c[0] + dx, c[1] + dy], j));
      return { joint: j.id, servo: j.servo, center: centre.toArray(), screen: [cx, cy], point: cands.find(solid) || cands.find((c) => ok(c, j)) || null };
    });
    motorCache = { frame: frameNo, list };
    return list;
  }
  // Test hook (read-only state) for automated page checks.
  root.explorerState = (options = {}) => ({
    chapter,
    focus: focus?.id || null,
    region: region?.id || null,
    joint: joint?.id || null,
    q: joint ? q.get(joint.id) : null,
    qAll: Object.fromEntries(q),
    explode: explodeT,
    playing,
    dragging: Boolean(drag),
    orbitEnabled: controls.enabled,
    nearSide,
    figure: figureSpan,
    visible,
    running,
    parts: parts.length,
    meshes: new Set(parts.map((p) => p.object.geometry.uuid)).size,
    layers: Object.fromEntries(parts.map((p) => [p.id, p.layer])),
    pose: parts.map((p) => [p.id, p.object.position.toArray(), p.object.quaternion.toArray()]),
    rest: parts.map((p) => [p.id, p.position.toArray(), p.quaternion.toArray()]),
    pivot: joint ? jointPivot(joint).toArray() : null,
    axis: joint ? jointAxis(joint).toArray() : null,
    arc: joint ? arcPoints(8).map(project) : null,
    enabled: drive.enabled,
    hoverMotor: hoverMotor?.id || null,
    // Labels on the stage: functional part labels carry the joints and servos they count.
    callouts: callouts.map((c) => {
      const o = c.part ? owned(functional.get(c.part)) : null,
        anchors = c.anchors ? c.anchors() : [c.anchor()];
      return {
        text: c.text,
        sub: c.sub || null,
        part: c.part || null,
        group: c.group || null,
        side: c.part && functional.get(c.part).per_side ? nearSide : null,
        joints: o ? o.joints : null,
        servos: o ? o.servos : null,
        anchors: anchors.map((v) => v.toArray()),
        anchorsScreen: anchors.map(project),
        box: c.el.getBoundingClientRect().toJSON(),
      };
    }),
    ...(() => {
      const g = gizmoDiagnostics();
      if (!g) return { shaft: null, arcGeometry: null, gizmo: null };
      for (const v of [g.shaft, g.arcGeometry]) v.centre = v.center;
      return { ...g, gizmo: { shaft: g.shaft, arc: g.arcGeometry } };
    })(),
    motors: chapter === 2 && options.motors !== false ? motorDiagnostics() : null,
    camera: { position: camera.position.toArray(), target: controls.target.toArray() },
  });
}
