// Checks web/assets/robot/microduck-meshopt.glb against the unchanged source GLB.
//
//   node artwork/meshopt/check.mjs [--source GLB] [--derivative GLB] [--report JSON] [--position-tolerance M]
//
// Uses only the decoder and Three.js files the course ships (web/assets/vendor), not the
// encoder, so it checks what the browser will see:
// - every glTF field other than buffers, buffer views and position bounds is identical;
// - each decoded buffer has the declared size, and position bounds match the decoded data;
// - per mesh, the vertex and triangle counts match, every decoded vertex lies within the stated
//   tolerance of a source vertex with the same normal, the triangles (corners and winding) are
//   exactly the source triangles, and no triangle whose source cross product is non-zero decodes
//   to a zero cross product or to one pointing the other way (flipped);
// - the explorer imports a decoder whose name carries its content hash and which includes the
//   full MIT license;
// - Three's GLTFLoader builds the same 70 named parts with the same extras, hierarchy and
//   rest transforms, and bounding boxes within the position tolerance.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { registerHooks } from "node:module";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { isDeepStrictEqual as same } from "node:util";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const VENDOR = path.join(ROOT, "web/assets/vendor");
const { values: args } = parseArgs({
  options: { source: { type: "string" }, derivative: { type: "string" }, report: { type: "string" }, "position-tolerance": { type: "string" } },
});
const sourcePath = path.resolve(args.source ?? path.join(ROOT, "web/assets/robot/microduck.glb"));
const derivedPath = path.resolve(args.derivative ?? path.join(ROOT, "web/assets/robot/microduck-meshopt.glb"));
const POSITION_TOL = Number(args["position-tolerance"] ?? 1e-6); // metres
const NORMAL_TOL_DEG = 0.25;

registerHooks({
  resolve(specifier, context, next) {
    if (specifier === "three") return { url: pathToFileURL(path.join(VENDOR, "three.module.js")).href, shortCircuit: true };
    if (specifier.startsWith("three/addons/"))
      return { url: pathToFileURL(path.join(VENDOR, "addons", specifier.slice(13))).href, shortCircuit: true };
    return next(specifier, context);
  },
});
const THREE = await import("three");
const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
// The decoder the explorer actually imports.
const explorer = fs.readFileSync(path.join(ROOT, "web/assets/robot-explorer.js"), "utf8");
const decoderName = explorer.match(/from "\.\/vendor\/(meshopt_decoder-[^"]+\.js)"/)?.[1];
if (!decoderName) throw Error("robot-explorer.js does not import a vendored meshopt decoder");
const decoderPath = path.join(VENDOR, decoderName), decoderBytes = fs.readFileSync(decoderPath);
const { MeshoptDecoder } = await import(pathToFileURL(decoderPath));
await MeshoptDecoder.ready;

const failures = [];
const check = (ok, message) => ok || failures.push(message);
const sha = (b) => crypto.createHash("sha256").update(b).digest("hex");
check(decoderName.endsWith(`-${sha(decoderBytes).slice(0, 12)}.js`), "decoder name does not match its content hash");
check(decoderBytes.includes("Permission is hereby granted, free of charge") && decoderBytes.includes("Copyright (c) 2016-2026 Arseny Kapoulkine"), "decoder lacks the MIT license text");
check(explorer.includes("setMeshoptDecoder(MeshoptDecoder)") && explorer.includes('"./robot/microduck-meshopt.glb"'), "explorer does not load the compressed model");

function readGlb(bytes) {
  const jsonLength = bytes.readUInt32LE(12);
  const json = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString("utf8"));
  const at = 20 + jsonLength;
  return { json, bin: bytes.subarray(at + 8, at + 8 + bytes.readUInt32LE(at)) };
}
const sourceBytes = fs.readFileSync(sourcePath), derivedBytes = fs.readFileSync(derivedPath);
const S = readGlb(sourceBytes), D = readGlb(derivedBytes);

// ---------------------------------------------------------------- glTF fields
for (const key of new Set([...Object.keys(S.json), ...Object.keys(D.json)])) {
  if (["buffers", "bufferViews", "accessors", "extensionsUsed", "extensionsRequired"].includes(key)) continue;
  check(same(S.json[key], D.json[key]), `glTF ${key} differs`);
}
const plus = (list) => [...(list ?? []), "KHR_meshopt_compression"];
check(same(D.json.extensionsUsed, plus(S.json.extensionsUsed)), "extensionsUsed");
check(same(D.json.extensionsRequired, plus(S.json.extensionsRequired)), "extensionsRequired");
check(S.json.accessors.length === D.json.accessors.length && S.json.bufferViews.length === D.json.bufferViews.length, "accessor/view count");
S.json.accessors.forEach((a, i) => {
  const { min, max, ...rest } = a, d = D.json.accessors[i], { min: dmin, max: dmax, ...drest } = d;
  check(same(rest, drest), `accessor ${i} fields differ`);
  check(!min === !dmin && !max === !dmax, `accessor ${i} bounds presence`);
});

// ---------------------------------------------------------------- decoded buffers
const decoded = D.json.bufferViews.map((v, i) => {
  const s = S.json.bufferViews[i], ext = v.extensions?.KHR_meshopt_compression;
  check(v.byteLength === s.byteLength && v.target === s.target && v.byteStride === s.byteStride, `view ${i} header`);
  check(ext && D.json.buffers[v.buffer].extensions?.KHR_meshopt_compression?.fallback === true, `view ${i} not compressed`);
  check(ext.count * ext.byteStride === v.byteLength, `view ${i} decoded size`);
  const out = new Uint8Array(ext.count * ext.byteStride);
  MeshoptDecoder.decodeGltfBuffer(out, ext.count, ext.byteStride, D.bin.subarray(ext.byteOffset, ext.byteOffset + ext.byteLength), ext.mode, ext.filter);
  return out;
});
const original = (i) => S.bin.subarray(S.json.bufferViews[i].byteOffset, S.json.bufferViews[i].byteOffset + S.json.bufferViews[i].byteLength);
const typed = (json, data, accessor) => {
  const a = json.accessors[accessor], b = Uint8Array.from(data);
  return a.componentType === 5126 ? new Float32Array(b.buffer) : a.componentType === 5125 ? new Uint32Array(b.buffer) : new Uint16Array(b.buffer);
};
const identicalViews = decoded.filter((d, i) => Buffer.compare(Buffer.from(d), original(i)) === 0).length;

// ---------------------------------------------------------------- geometry per mesh
let maxPos = 0, maxNormal = 0, vertices = 0, triangles = 0, sourceDegenerate = 0, derivedDegenerate = 0, flippedTriangles = 0, ambiguousVertices = 0, resolvedTriangles = 0;
// Cross product of the edges of triangle (i, j, k), in double precision from float32 coordinates.
const crossOf = (pos, [i, j, k]) => {
  const u = [pos[3 * j] - pos[3 * i], pos[3 * j + 1] - pos[3 * i + 1], pos[3 * j + 2] - pos[3 * i + 2]];
  const v = [pos[3 * k] - pos[3 * i], pos[3 * k + 1] - pos[3 * i + 1], pos[3 * k + 2] - pos[3 * i + 2]];
  return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
};
// Exact zero cross product, in double precision from the float32 coordinates.
const flat = (pos, ids, t) => {
  const [a, b, c] = [ids[3 * t] * 3, ids[3 * t + 1] * 3, ids[3 * t + 2] * 3];
  const u = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]];
  const v = [pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]];
  return u[1] * v[2] - u[2] * v[1] === 0 && u[2] * v[0] - u[0] * v[2] === 0 && u[0] * v[1] - u[1] * v[0] === 0;
};
for (const mesh of S.json.meshes) {
  const p = mesh.primitives[0], at = (k) => S.json.accessors[k].bufferView;
  const sp = typed(S.json, original(at(p.attributes.POSITION)), p.attributes.POSITION), sn = typed(S.json, original(at(p.attributes.NORMAL)), p.attributes.NORMAL);
  const si = typed(S.json, original(at(p.indices)), p.indices);
  const dp = typed(D.json, decoded[at(p.attributes.POSITION)], p.attributes.POSITION), dn = typed(D.json, decoded[at(p.attributes.NORMAL)], p.attributes.NORMAL);
  const di = typed(D.json, decoded[at(p.indices)], p.indices);
  const n = sp.length / 3;
  check(dp.length === sp.length && dn.length === sn.length && di.length === si.length, `${mesh.name}: counts`);
  check(di.every((x) => x < n), `${mesh.name}: index out of range`);
  const da = D.json.accessors[p.attributes.POSITION];
  for (let c = 0; c < 3; c++) {
    let lo = Infinity, hi = -Infinity;
    for (let v = c; v < dp.length; v += 3) (lo = Math.min(lo, dp[v])), (hi = Math.max(hi, dp[v]));
    check(Math.fround(da.min[c]) === lo && Math.fround(da.max[c]) === hi, `${mesh.name}: bounds do not match decoded data`);
  }
  // Source vertices with identical values are one class; triangles are compared as class triples.
  const key = (a, b, v) => `${a[3 * v]},${a[3 * v + 1]},${a[3 * v + 2]}|${b[3 * v]},${b[3 * v + 1]},${b[3 * v + 2]}`;
  const classOf = new Map(), cls = new Int32Array(n), grid = new Map(), cell = (x) => Math.floor(x / POSITION_TOL);
  for (let v = 0; v < n; v++) {
    const k = key(sp, sn, v);
    if (!classOf.has(k)) {
      classOf.set(k, v);
      const g = `${cell(sp[3 * v])},${cell(sp[3 * v + 1])},${cell(sp[3 * v + 2])}`;
      grid.has(g) ? grid.get(g).push(v) : grid.set(g, [v]);
    }
    cls[v] = classOf.get(k);
  }
  // Candidate source classes within tolerance for every decoded vertex, closest first. A few CAD
  // vertices are distinct but closer together than the tolerance; their triangles decide.
  const candidates = [];
  for (let v = 0; v < n; v++) {
    const list = [];
    const [x, y, z] = [cell(dp[3 * v]), cell(dp[3 * v + 1]), cell(dp[3 * v + 2])];
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++)
      for (const s of grid.get(`${x + a},${y + b},${z + c}`) ?? []) {
        const e = Math.max(...[0, 1, 2].map((k) => Math.abs(dp[3 * v + k] - sp[3 * s + k])));
        const [a0, a1, a2] = [dn[3 * v], dn[3 * v + 1], dn[3 * v + 2]], [b0, b1, b2] = [sn[3 * s], sn[3 * s + 1], sn[3 * s + 2]];
        const cross = Math.hypot(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0);
        const angle = (Math.atan2(cross, a0 * b0 + a1 * b1 + a2 * b2) * 180) / Math.PI;
        if (e <= POSITION_TOL && angle <= NORMAL_TOL_DEG) list.push({ s, e, angle, score: e / POSITION_TOL + angle / NORMAL_TOL_DEG });
      }
    check(list.length > 0, `${mesh.name}: vertex ${v} has no source vertex within tolerance`);
    candidates.push(list.sort((p, q) => p.score - q.score));
    if (list.length > 1) ambiguousVertices++;
  }
  const tri = (c) => {
    const r = c.indexOf(Math.min(...c));
    return `${c[r]},${c[(r + 1) % 3]},${c[(r + 2) % 3]}`; // rotation only: winding is kept
  };
  const count = new Map(), flatCount = new Map(), m = di.length / 3;
  for (let t = 0; t < m; t++) {
    const k = tri([cls[si[3 * t]], cls[si[3 * t + 1]], cls[si[3 * t + 2]]]);
    count.set(k, (count.get(k) ?? 0) + 1);
    if (flat(sp, si, t)) flatCount.set(k, (flatCount.get(k) ?? 0) + 1), sourceDegenerate++;
  }
  // Each decoded triangle takes one source triangle: the closest candidates first, then any
  // other combination of candidates. Errors are measured for the vertices actually used.
  let added = 0, flippedHere = 0;
  const consume = (t, picks) => {
    const k = tri(picks.map((q) => q.s));
    if (!(count.get(k) > 0)) return false;
    count.set(k, count.get(k) - 1);
    for (const q of picks) (maxPos = Math.max(maxPos, q.e)), (maxNormal = Math.max(maxNormal, q.angle));
    // Orientation: source cross product from the matched source vertices, decoded from the decoded ones.
    const sc = crossOf(sp, picks.map((q) => q.s)), dc = crossOf(dp, [di[3 * t], di[3 * t + 1], di[3 * t + 2]]);
    if (sc[0] * dc[0] + sc[1] * dc[1] + sc[2] * dc[2] < 0) flippedTriangles++, flippedHere++;
    if (flat(dp, di, t)) {
      derivedDegenerate++;
      if (flatCount.get(k) > 0) flatCount.set(k, flatCount.get(k) - 1);
      else added++;
    }
    return true;
  };
  const pending = [];
  for (let t = 0; t < m; t++) {
    const first = [0, 1, 2].map((k) => candidates[di[3 * t + k]]?.[0]);
    if (!first.every(Boolean) || !consume(t, first)) pending.push(t);
  }
  for (const t of pending) {
    const [A, B, C] = [0, 1, 2].map((k) => candidates[di[3 * t + k]] ?? []);
    let best = null; // the available combination with the smallest worst-vertex score
    for (const a of A) for (const b of B) for (const c of C)
      if (count.get(tri([a.s, b.s, c.s])) > 0 && (!best || Math.max(a.score, b.score, c.score) < Math.max(...best.map((q) => q.score)))) best = [a, b, c];
    if (best && consume(t, best)) resolvedTriangles++;
    else check(false, `${mesh.name}: decoded triangle ${t} matches no source triangle`);
  }
  check([...count.values()].every((x) => x === 0), `${mesh.name}: triangles differ from the source`);
  check(added === 0, `${mesh.name}: ${added} triangles decode with a zero cross product`);
  check(flippedHere === 0, `${mesh.name}: ${flippedHere} triangles decode flipped`);
  vertices += n, triangles += m;
}

// ---------------------------------------------------------------- Three.js GLTFLoader
const load = (bytes, decoder) =>
  new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    if (decoder) loader.setMeshoptDecoder(decoder);
    loader.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "", resolve, reject);
  });
const describe = (scene) => {
  const parts = [];
  scene.updateMatrixWorld(true);
  scene.traverse((o) => {
    if (!o.isMesh) return;
    o.geometry.computeBoundingBox();
    parts.push({
      name: o.name, parent: o.parent.name, userData: o.userData,
      position: o.position.toArray(), quaternion: o.quaternion.toArray(), scale: o.scale.toArray(),
      material: o.material.name, vertices: o.geometry.attributes.position.count, indices: o.geometry.index.count,
      normalType: o.geometry.attributes.normal.array.constructor.name, positionType: o.geometry.attributes.position.array.constructor.name,
      box: [...o.geometry.boundingBox.min.toArray(), ...o.geometry.boundingBox.max.toArray()], geometry: o.geometry,
    });
  });
  return parts;
};
const [a, b] = [describe((await load(sourceBytes)).scene), describe((await load(derivedBytes, MeshoptDecoder)).scene)];
let maxBox = 0;
check(a.length === 70 && b.length === 70 && new Set(b.map((p) => p.geometry)).size === 38, "Three: expected 70 parts sharing 38 geometries");
a.forEach((p, i) => {
  const q = b[i];
  for (const k of ["name", "parent", "userData", "position", "quaternion", "scale", "material", "vertices", "indices", "normalType", "positionType"])
    check(same(p[k], q[k]), `Three: ${p.name} ${k} differs`);
  maxBox = Math.max(maxBox, ...p.box.map((x, k) => Math.abs(x - q.box[k])));
});
check(maxBox <= POSITION_TOL, "Three: bounding boxes outside tolerance");
const ids = b.map((p) => p.userData.id);
check(new Set(ids).size === 70 && ids.every(Boolean), "Three: extras.id not unique");

const gz = (bytes) => import("node:zlib").then((z) => z.gzipSync(bytes, { level: 9 }).length);
const report = {
  passed: failures.length === 0,
  failures,
  source: { path: path.relative(ROOT, sourcePath), bytes: sourceBytes.length, gzip9: await gz(sourceBytes), sha256: sha(sourceBytes) },
  derivative: { path: path.relative(ROOT, derivedPath), bytes: derivedBytes.length, gzip9: await gz(derivedBytes), sha256: sha(derivedBytes) },
  decoder: { path: path.relative(ROOT, decoderPath), bytes: fs.statSync(decoderPath).size, gzip9: await gz(fs.readFileSync(decoderPath)), sha256: sha(fs.readFileSync(decoderPath)) },
  meshes: S.json.meshes.length, nodes: S.json.nodes.length, materials: S.json.materials.length, accessors: S.json.accessors.length,
  vertices, triangles, zero_area_triangles: { source: sourceDegenerate, derivative: derivedDegenerate }, flipped_triangles: flippedTriangles,
  vertices_with_several_candidates: ambiguousVertices, triangles_resolved_among_candidates: resolvedTriangles, buffer_views_byte_identical: identicalViews, buffer_views: decoded.length,
  max_position_error_m: maxPos, max_normal_error_deg: maxNormal, max_bounding_box_error_m: maxBox,
  tolerance: { position_m: POSITION_TOL, normal_deg: NORMAL_TOL_DEG },
};
if (args.report) fs.writeFileSync(args.report, JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify(report, null, 2));
process.exit(report.passed ? 0 : 1);
