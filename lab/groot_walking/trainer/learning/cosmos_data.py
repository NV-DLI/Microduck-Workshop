"""Cosmos 3 controlled appearance transfer with explicit label review.

This freezes Cosmos and trains the downstream action policy on accepted output.
A generated clip is not a physics rollout and cannot demonstrate task success.
"""

import argparse
import hashlib
import json
from pathlib import Path

MODEL_REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
MODEL_REPO = "nvidia/Cosmos3-Nano"


def training_source_binding(episode, start, frames, dataset):
    """Only augment frames already admitted to the declared training split."""
    episode, dataset = Path(episode).resolve(), Path(dataset).resolve()
    split_path, provenance_path = dataset / "split.json", dataset / "provenance.json"
    if json.loads(split_path.read_text()).get("split") != "train":
        raise ValueError("Appearance augmentation requires a training dataset")
    source_sha = hashlib.sha256(
        (episode / "learning/samples.jsonl").read_bytes()
    ).hexdigest()
    records = json.loads(provenance_path.read_text())
    admitted = any(
        Path(row["source"]).resolve() == episode
        and row.get("split") == "train"
        and row.get("samples_sha256") == source_sha
        and row["start_frame"] <= start
        and start + frames - 1 <= row["end_frame"]
        for row in records
    )
    if not admitted:
        raise ValueError("Clip frames are not admitted in this training dataset")
    return dict(
        dataset=str(dataset),
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        source_samples_sha256=source_sha,
    )


def validate_training_source(meta):
    binding = meta.get("training_source")
    if not isinstance(binding, dict) or not binding.get("dataset"):
        raise ValueError("Clip is missing its training-source binding")
    actual = training_source_binding(
        meta["episode"], meta["start_frame"], meta["frames"], binding["dataset"]
    )
    if actual != binding:
        raise ValueError("Training-source binding changed; prepare and review again")


def load_transfer_components(pipe, dtype):
    """Pin the actual delayed component loads, not only the pipeline index."""
    names = ["text_tokenizer", "vae", "transformer", "scheduler"]
    for name in names:
        if pipe.get_component_spec(name).pretrained_model_name_or_path != MODEL_REPO:
            raise ValueError("Unexpected transfer component repository: " + name)
    pipe.load_components(names=names, torch_dtype=dtype, revision=MODEL_REVISION)
    sources = {}
    for name in names:
        component = getattr(pipe, name, None)
        spec = pipe.get_component_spec(name)
        if component is None or spec.revision != MODEL_REVISION:
            raise ValueError("Pinned transfer component did not load: " + name)
        sources[name] = dict(
            repository=spec.pretrained_model_name_or_path,
            revision=spec.revision,
            subfolder=spec.subfolder,
            load_id=getattr(component, "_diffusers_load_id", None),
        )
    return sources


def validate_review(clip):
    """Bind accepted images and frame-to-label alignment to the reviewed output."""
    clip = Path(clip)
    review = json.loads((clip / "review.json").read_text())
    meta_path = clip / "generation.json"
    meta = json.loads(meta_path.read_text())
    if review.get("accepted") is not True or not all(
        isinstance(review.get(k), str) and review[k].strip()
        for k in ("reviewer", "reason")
    ):
        raise ValueError("Generated clip has not been accepted after review")
    if (
        review.get("generation_sha256")
        != hashlib.sha256(meta_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Generation metadata changed after review; inspect again")
    if sorted(review.get("inspected_frames", [])) != list(range(meta["frames"])):
        raise ValueError("Review the complete clip before reusing actions")
    hashes = [
        hashlib.sha256((clip / "generated" / f"{i:04d}.png").read_bytes()).hexdigest()
        for i in range(meta["frames"])
    ]
    if review.get("generated_frame_sha256") != hashes:
        raise ValueError("Generated frames changed after review")
    source = Path(meta["episode"]) / "learning/samples.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != meta["source_samples_sha256"]:
        raise ValueError("Source labels changed after generation")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line]
    selected = rows[meta["start_frame"] : meta["start_frame"] + meta["frames"]]
    if len(selected) != meta["frames"] or selected != meta["samples"]:
        raise ValueError("Generated frame-to-label alignment differs from source")
    validate_training_source(meta)
    return meta


def prepare(episode, output, start, frames, training_dataset):
    import cv2
    import numpy as np
    from PIL import Image

    if output.exists():
        raise FileExistsError(output)
    if start < 0 or frames < 17 or frames % 4 != 1:
        raise ValueError(
            "Use at least 17 consecutive frames (4n+1), with a non-negative start"
        )
    rows = [
        json.loads(x)
        for x in (episode / "learning/samples.jsonl").read_text().splitlines()
    ]
    selected = rows[start : start + frames]
    if len(selected) != frames:
        raise ValueError("Clip extends past episode")
    if any(
        abs(selected[i + 1]["timestamp_s"] - selected[i]["timestamp_s"] - 0.08) > 1e-6
        for i in range(frames - 1)
    ):
        raise ValueError("Non-contiguous clip")
    binding = training_source_binding(episode, start, frames, training_dataset)
    output.mkdir(parents=True)
    (output / "control").mkdir()
    (output / "reference").mkdir()
    for i, row in enumerate(selected):
        image = (
            Image.open(episode / "learning" / row["image"])
            .convert("RGB")
            .resize((512, 288))
        )
        image.save(output / "reference" / f"{i:04d}.png")
        edges = cv2.Canny(np.asarray(image), 100, 200)
        Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2)).save(
            output / "control" / f"{i:04d}.png"
        )
    metadata = dict(
        episode=str(episode.resolve()),
        start_frame=start,
        frames=frames,
        fps=12.5,
        source_samples_sha256=hashlib.sha256(
            (episode / "learning/samples.jsonl").read_bytes()
        ).hexdigest(),
        samples=selected,
        control="canny_100_200",
        width=512,
        height=288,
        review_status="pending",
        label_policy="Reuse actions only if task geometry and temporal alignment remain intact",
        training_source=binding,
    )
    (output / "clip.json").write_text(json.dumps(metadata, indent=2))


def generate(clip, prompt, seed, steps):
    import time
    import torch
    from PIL import Image
    from diffusers import Cosmos3OmniModularPipeline
    import diffusers

    cfg = json.loads((clip / "clip.json").read_text())
    validate_training_source(cfg)
    dest = clip / "generated"
    if dest.exists():
        raise FileExistsError(dest)
    # Do not silently drop the official safety checker when its gated model is
    # unavailable. Hugging Face access must be configured before this step.
    pipe = Cosmos3OmniModularPipeline.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=torch.bfloat16
    )
    component_sources = load_transfer_components(pipe, torch.bfloat16)
    pipe.enable_safety_checker()
    pipe.to("cuda")
    control = [
        Image.open(clip / "control" / f"{i:04d}.png").convert("RGB")
        for i in range(cfg["frames"])
    ]
    started = time.monotonic()
    videos = pipe(
        prompt=prompt,
        negative_prompt="Extra balls, extra goalposts, displaced objects, warped geometry, motion blur, flicker.",
        control_videos={"edge": control},
        control_guidance=1.5,
        num_frames=cfg["frames"],
        num_video_frames_per_chunk=cfg["frames"],
        num_conditional_frames=1,
        width=cfg["width"],
        height=cfg["height"],
        fps=cfg["fps"],
        num_inference_steps=steps,
        guidance_scale=3.0,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        output="videos",
    )
    # The modular pipeline exposes a video as a list of PIL frames (possibly
    # batch-wrapped). Reject unfamiliar output rather than drop/repeat frames.
    if len(videos) == 1 and isinstance(videos[0], list):
        videos = videos[0]
    if len(videos) != cfg["frames"] or not all(
        isinstance(v, Image.Image) for v in videos
    ):
        raise ValueError("Unexpected generated video representation")
    dest.mkdir()
    for i, frame in enumerate(videos):
        frame.save(dest / f"{i:04d}.png")
    from diffusers.utils import export_to_video

    export_to_video(
        videos, str(clip / "generated.mp4"), fps=cfg["fps"], macro_block_size=1
    )
    cfg.update(
        prompt=prompt,
        seed=seed,
        steps=steps,
        elapsed_s=time.monotonic() - started,
        model=MODEL_REPO,
        model_revision=MODEL_REVISION,
        component_sources=component_sources,
        diffusers_version=diffusers.__version__,
        review_status="pending",
    )
    (clip / "generation.json").write_text(json.dumps(cfg, indent=2))
    review_sheet(clip)


def review_sheet(clip):
    from html import escape
    from PIL import Image, ImageDraw

    cfg = json.loads((clip / "clip.json").read_text())
    indices = sorted(set(round(i * (cfg["frames"] - 1) / 7) for i in range(8)))
    sheet = Image.new("RGB", (1024, len(indices) * 310), "white")
    draw = ImageDraw.Draw(sheet)
    for row, idx in enumerate(indices):
        for col, folder in enumerate(("reference", "generated")):
            image = Image.open(clip / folder / f"{idx:04d}.png").convert("RGB")
            sheet.paste(image, (col * 512, row * 310 + 22))
            draw.text(
                (col * 512 + 5, row * 310 + 3),
                f"{folder} frame {idx} @ {idx/12.5:.2f}s",
                fill="black",
            )
    sheet.save(clip / "review.png")
    # The overview is only a navigation aid. Full-resolution pairs expose every
    # frame so reviewers can find brief geometry changes between sampled frames.
    figures = []
    for idx in range(cfg["frames"]):
        figures.append(
            f'<figure id="frame-{idx}"><figcaption>帧 / Frame {idx} · '
            f'{idx/cfg["fps"]:.2f} s</figcaption><div class="pair">'
            f'<img loading="lazy" src="reference/{idx:04d}.png" alt="原始 / Source {idx}">'
            f'<img loading="lazy" src="generated/{idx:04d}.png" alt="生成 / Generated {idx}">'
            "</div></figure>"
        )
    (clip / "review.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>逐帧检查 / Frame review</title><style>"
        "body{font:16px/1.6 system-ui;max-width:1100px;margin:24px auto;padding:0 16px}"
        ".pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}"
        "figure{margin:24px 0}figcaption{font-weight:bold}"
        "@media(max-width:600px){.pair{grid-template-columns:1fr}}"
        "</style><h1>逐帧检查 / Frame review</h1>"
        "<p>左侧（手机上方）为原始图，右侧（手机下方）为生成图。检查球和球门的位置、"
        "边界、遮挡以及帧间连续性；任何改变任务几何的片段都不应沿用原动作标签。</p>"
        "<p>Source on the left (top on mobile), generated image on the right (bottom). "
        "Check object geometry, occlusion and temporal continuity before reusing action labels.</p>"
        f'<p>{escape(str(clip.name))} · {cfg["frames"]} frames</p>'
        + "".join(figures)
        + "</html>",
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    prep = s.add_parser("prepare")
    prep.add_argument("--episode", type=Path, required=True)
    prep.add_argument("--training-dataset", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--start", type=int, default=0)
    prep.add_argument("--frames", type=int, default=49)
    gen = s.add_parser("generate")
    gen.add_argument("--clip", type=Path, required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--seed", type=int, default=2026)
    gen.add_argument("--steps", type=int, default=50)
    review = s.add_parser("review")
    review.add_argument("--clip", type=Path, required=True)
    review.add_argument("--decision", choices=["accept", "reject"], required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reason", required=True)
    review.add_argument("--inspected-all-frames", action="store_true")
    manifest = s.add_parser(
        "manifest", help="Export the accepted clip's training-data manifest"
    )
    manifest.add_argument("--clip", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.episode, a.output, a.start, a.frames, a.training_dataset)
    elif a.command == "generate":
        generate(a.clip, a.prompt, a.seed, a.steps)
    elif a.command == "manifest":
        meta = validate_review(a.clip)
        episode = Path(meta["episode"])
        report = json.loads((episode / "report.json").read_text())
        with a.output.open("x") as f:
            json.dump(
                [
                    dict(
                        path=str(episode.resolve()),
                        seed=report["seed"],
                        split="train",
                        augmentation_clip=str(a.clip.resolve()),
                    )
                ],
                f,
                indent=2,
            )
    else:
        if a.decision == "accept" and not a.inspected_all_frames:
            p.error("Inspect the whole clip before accepting action-label reuse")
        if not a.reviewer.strip() or not a.reason.strip():
            p.error("Provide a reviewer and a reason")
        meta = json.loads((a.clip / "generation.json").read_text())
        hashes = [
            hashlib.sha256(
                (a.clip / "generated" / f"{i:04d}.png").read_bytes()
            ).hexdigest()
            for i in range(meta["frames"])
        ]
        record = dict(
            accepted=a.decision == "accept",
            reviewer=a.reviewer,
            reason=a.reason,
            inspected_frames=(
                list(range(meta["frames"])) if a.inspected_all_frames else []
            ),
            generated_frame_sha256=hashes,
            generation_sha256=hashlib.sha256(
                (a.clip / "generation.json").read_bytes()
            ).hexdigest(),
        )
        with (a.clip / "review.json").open("x") as f:
            json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
