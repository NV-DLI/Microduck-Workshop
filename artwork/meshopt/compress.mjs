// Writes a smaller web copy of the Microduck GLB using KHR_meshopt_compression, plus the
// course's copy of the matching browser decoder.
//
//   node artwork/meshopt/compress.mjs [--source GLB] [--output GLB] [--meshoptimizer DIR] [--lossless]
//
// Nodes, transforms, extras, materials and the number of vertices and triangles stay exactly as in
// the source; no quantised integer types or compensating node transforms are introduced, so the
// viewer's rest poses, pivots and bounding boxes use the source coordinates. Only vertex data is
// compressed:
// - each mesh's vertices and triangles are put in an order that compresses well (every vertex keeps
//   its values; every triangle keeps its corners and winding);
// - positions stay float32 rounded to 18 mantissa bits per axis of each mesh (EXPONENTIAL filter,
//   under 0.5 micrometres here). A mesh where that rounding would make any triangle's cross product
//   zero, or turn it to face the other way, tries 20 bits, then keeps its exact positions;
// - normals stay float32 with 10 mantissa bits per vector (under 0.1 degree);
// - indices and the filtered floats are then encoded losslessly.
// --lossless keeps the order and every value: the decoded buffers equal the source byte for byte.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const HERE = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.resolve(HERE, "../..");
const { values: args } = parseArgs({
  options: { source: { type: "string" }, output: { type: "string" }, meshoptimizer: { type: "string" }, lossless: { type: "boolean" } },
});
const SETTINGS = args.lossless
  ? { reorder: false, indexMode: "INDICES", positionBits: [], normalBits: 0 }
  : { reorder: true, indexMode: "TRIANGLES", positionBits: [18, 20], positionMode: "SharedComponent", normalBits: 10, normalMode: "SharedVector" };
const source = path.resolve(args.source ?? path.join(ROOT, "web/assets/robot/microduck.glb"));
const output = path.resolve(args.output ?? path.join(ROOT, "web/assets/robot/microduck-meshopt.glb"));
const lib = path.resolve(args.meshoptimizer ?? path.join(HERE, "node_modules/meshoptimizer"));
const { MeshoptEncoder: E } = await import(pathToFileURL(path.join(lib, "meshopt_encoder.js")));
await E.ready;

function readGlb(bytes) {
  if (bytes.readUInt32LE(0) !== 0x46546c67) throw Error("not a GLB");
  const jsonLength = bytes.readUInt32LE(12);
  const json = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString("utf8"));
  const binStart = 20 + jsonLength;
  const bin = bytes.subarray(binStart + 8, binStart + 8 + bytes.readUInt32LE(binStart));
  return { json, bin };
}

function writeGlb(json, bin) {
  // JSON.stringify writes -0 as 0; the source's -0.0 values are kept as written.
  const text = JSON.stringify(json, (k, v) => (Object.is(v, -0) ? "\u0000-0" : v)).replaceAll('"\\u0000-0"', "-0.0");
  const pad = (b, fill) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4, fill)]);
  const j = pad(Buffer.from(text), 0x20), b = pad(bin, 0);
  const header = Buffer.alloc(12), jh = Buffer.alloc(8), bh = Buffer.alloc(8);
  header.writeUInt32LE(0x46546c67, 0), header.writeUInt32LE(2, 4), header.writeUInt32LE(12 + 8 + j.length + 8 + b.length, 8);
  jh.writeUInt32LE(j.length, 0), jh.writeUInt32LE(0x4e4f534a, 4);
  bh.writeUInt32LE(b.length, 0), bh.writeUInt32LE(0x004e4942, 4);
  return Buffer.concat([header, jh, j, bh, b]);
}

// Value the browser decodes from one EXPONENTIAL word: 24-bit mantissa times 2^(top byte).
const expValue = (word) => Math.fround(((word << 8) >> 8) * 2 ** (word >> 24));

// Cross product of triangle t's edges, in double precision from the float32 coordinates.
function cross(pos, index, t) {
  const [a, b, c] = [index[t] * 3, index[t + 1] * 3, index[t + 2] * 3];
  const u = [pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]];
  const v = [pos[c] - pos[a], pos[c + 1] - pos[a + 1], pos[c + 2] - pos[a + 2]];
  return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
}

// Triangles the rounded positions damage: a non-zero source cross product that becomes exactly
// zero (zero area) or points against the source one (flipped).
function damaged(pos, rounded, index) {
  let zero = 0, flipped = 0;
  for (let t = 0; t < index.length / 3; t++) {
    const s = cross(pos, index, t), r = cross(rounded, index, t);
    if (s.every((x) => x === 0)) continue;
    if (r.every((x) => x === 0)) zero++;
    else if (s[0] * r[0] + s[1] * r[1] + s[2] * r[2] < 0) flipped++;
  }
  return { zero, flipped };
}

const { json: src, bin } = readGlb(fs.readFileSync(source));
const view = (accessor) => {
  const a = src.accessors[accessor], v = src.bufferViews[a.bufferView];
  if (v.byteStride || a.byteOffset || a.sparse || v.buffer !== 0) throw Error(`accessor ${accessor}: unexpected layout`);
  return { a, bytes: Uint8Array.from(bin.subarray(v.byteOffset, v.byteOffset + v.byteLength)) };
};

// Every primitive is an indexed triangle list with float POSITION and NORMAL, each accessor in
// its own buffer view; anything else is refused rather than silently changed.
const out = structuredClone(src);
const chunks = [], perMesh = [];
let packed = 0, fallback = 0;
const place = (accessor, data, count, stride, mode, filter) => {
  const bv = out.bufferViews[out.accessors[accessor].bufferView];
  const encoded = E.encodeGltfBuffer(data, count, stride, mode, 1);
  const ext = { buffer: 0, byteOffset: packed, byteLength: encoded.length, byteStride: stride, count, mode };
  if (filter) ext.filter = filter;
  Object.assign(bv, { buffer: 1, byteOffset: fallback, extensions: { KHR_meshopt_compression: ext } });
  chunks.push(encoded, new Uint8Array((4 - (encoded.length % 4)) % 4));
  packed += Math.ceil(encoded.length / 4) * 4;
  fallback += Math.ceil(bv.byteLength / 4) * 4;
  return encoded.length;
};
const seen = new Set();
for (const mesh of src.meshes)
  for (const p of mesh.primitives) {
    const names = Object.keys(p.attributes).sort().join();
    if ((p.mode ?? 4) !== 4 || names !== "NORMAL,POSITION" || p.indices === undefined || p.targets) throw Error(`mesh ${mesh.name}: unsupported primitive`);
    for (const k of [p.indices, p.attributes.POSITION, p.attributes.NORMAL]) {
      if (seen.has(src.accessors[k].bufferView)) throw Error("shared buffer view");
      seen.add(src.accessors[k].bufferView);
    }
    const I = view(p.indices), P = view(p.attributes.POSITION), N = view(p.attributes.NORMAL);
    const wide = I.a.componentType === 5125, n = P.a.count;
    if (P.a.componentType !== 5126 || N.a.componentType !== 5126 || N.a.count !== n) throw Error("expected float32 vertices");
    let index = Uint32Array.from(wide ? new Uint32Array(I.bytes.buffer) : new Uint16Array(I.bytes.buffer));
    let pos = new Float32Array(P.bytes.buffer), nor = new Float32Array(N.bytes.buffer);
    if (SETTINGS.reorder) {
      const [remap, unique] = E.reorderMesh(index, true, true); // rewrites index in place
      if (unique !== n) throw Error(`mesh ${mesh.name}: unreferenced vertices`);
      const p2 = new Float32Array(n * 3), n2 = new Float32Array(n * 3);
      for (let v = 0; v < n; v++) {
        p2.set(pos.subarray(v * 3, v * 3 + 3), remap[v] * 3);
        n2.set(nor.subarray(v * 3, v * 3 + 3), remap[v] * 3);
      }
      pos = p2, nor = n2;
    }
    const record = { mesh: mesh.name, vertices: n, triangles: index.length / 3 };
    const indexBytes = new Uint8Array((wide ? index : Uint16Array.from(index)).buffer);
    record.index_bytes = place(p.indices, indexBytes, index.length, wide ? 4 : 2, SETTINGS.indexMode);

    // Positions: the first precision that damages no triangle, otherwise exact floats.
    let positions = null;
    for (const bits of SETTINGS.positionBits) {
      const words = E.encodeFilterExp(pos, n, 12, bits, SETTINGS.positionMode), q = new Int32Array(words.buffer);
      const decodedPos = Float32Array.from(q, expValue);
      const { zero, flipped } = damaged(pos, decodedPos, index);
      record[`damaged_at_${bits}_bits`] = { zero_area: zero, flipped };
      if (zero + flipped === 0) {
        positions = { bits, words, decodedPos };
        break;
      }
    }
    if (positions) {
      record.position_bits = positions.bits;
      record.position_bytes = place(p.attributes.POSITION, positions.words, n, 12, "ATTRIBUTES", "EXPONENTIAL");
      const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
      positions.decodedPos.forEach((x, i) => ((min[i % 3] = Math.min(min[i % 3], x)), (max[i % 3] = Math.max(max[i % 3], x))));
      Object.assign(out.accessors[p.attributes.POSITION], { min, max });
    } else {
      record.position_bits = "exact";
      record.position_bytes = place(p.attributes.POSITION, new Uint8Array(pos.buffer), n, 12, "ATTRIBUTES");
    }
    record.normal_bytes = SETTINGS.normalBits
      ? place(p.attributes.NORMAL, E.encodeFilterExp(nor, n, 12, SETTINGS.normalBits, SETTINGS.normalMode), n, 12, "ATTRIBUTES", "EXPONENTIAL")
      : place(p.attributes.NORMAL, new Uint8Array(nor.buffer), n, 12, "ATTRIBUTES");
    perMesh.push(record);
  }
if (seen.size !== src.bufferViews.length) throw Error("buffer views outside the meshes");

out.buffers = [{ byteLength: packed }, { byteLength: fallback, extensions: { KHR_meshopt_compression: { fallback: true } } }];
out.extensionsUsed = [...(src.extensionsUsed ?? []), "KHR_meshopt_compression"];
out.extensionsRequired = [...(src.extensionsRequired ?? []), "KHR_meshopt_compression"];
const glb = writeGlb(out, Buffer.concat(chunks));
fs.writeFileSync(output, glb);

// The browser decoder: the pinned upstream module with its MIT license appended, so the notice
// travels with every copy. The content hash in the name keeps caches from mixing versions.
const pkg = JSON.parse(fs.readFileSync(path.join(lib, "package.json")));
const license = fs.readFileSync(path.join(lib, "LICENSE.md"), "utf8").trim();
const decoderText = `${fs.readFileSync(path.join(lib, "meshopt_decoder.mjs"), "utf8").trimEnd()}\n\n/*\nmeshoptimizer ${pkg.version} (https://github.com/zeux/meshoptimizer), js/meshopt_decoder.mjs\n\n${license}\n*/\n`;
const decoderName = `meshopt_decoder-${pkg.version}-${crypto.createHash("sha256").update(decoderText).digest("hex").slice(0, 12)}.js`;
fs.writeFileSync(path.join(ROOT, "web/assets/vendor", decoderName), decoderText);

const sha = (b) => crypto.createHash("sha256").update(b).digest("hex");
const provenance = {
  description: "Meshopt-compressed web copy of microduck.glb for the 3D explorer; geometry adaptations CC BY-NC-SA 4.0 (Pollen Robotics).",
  source: { file: path.relative(ROOT, source), bytes: fs.statSync(source).size, sha256: sha(fs.readFileSync(source)) },
  output: { file: path.relative(ROOT, output), bytes: glb.length, sha256: sha(glb) },
  decoder: { file: `web/assets/vendor/${decoderName}`, bytes: Buffer.byteLength(decoderText), sha256: sha(decoderText) },
  tool: { meshoptimizer: pkg.version, node: process.version, command: `node artwork/meshopt/compress.mjs${args.lossless ? " --lossless" : ""}` },
  settings: SETTINGS,
  meshes: perMesh,
};
fs.writeFileSync(output.replace(/\.glb$/, ".provenance.json"), JSON.stringify(provenance, null, 2) + "\n");
const exact = perMesh.filter((m) => m.position_bits === "exact").length;
console.log(`${path.relative(ROOT, output)}: ${glb.length} bytes (source ${provenance.source.bytes}); positions exact in ${exact} of ${perMesh.length} meshes; decoder ${decoderName}`);
