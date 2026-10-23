"""Small explicit practice runs; course weights are never overwritten."""

from pathlib import Path
import argparse
import json
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("command", choices=["detect", "train-detector"])
p.add_argument("--output", type=Path, required=True)
p.add_argument("--model", type=Path)
p.add_argument("--source", type=Path)
p.add_argument("--data", type=Path)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seed", type=int, default=303)
a = p.parse_args()
if a.output.exists():
    p.error("Choose a new output directory")

# Inference uses the supplied checkpoint; practice training starts a separate model.
if a.command == "detect":
    if not a.model or not a.source:
        p.error("model and source are required")
    paths = (
        sorted(
            x
            for x in a.source.rglob("*")
            if x.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if a.source.is_dir()
        else [a.source]
    )
    if not paths:
        p.error("No input images")

    # Save annotated still images so detections can be compared side by side.
    results = YOLO(str(a.model)).predict(
        source=[str(x) for x in paths],
        imgsz=640,
        save=True,
        project=str(a.output.parent),
        name=a.output.name,
        exist_ok=False,
        device="cpu",
    )
else:
    if not a.data or a.epochs < 1 or a.batch < 1:
        p.error("data and positive training arguments required")
    # Resolve the dataset relative to its file, regardless of Ultralytics settings.
    import yaml
    import tempfile

    d = yaml.safe_load(a.data.read_text())
    d["path"] = str(a.data.parent.resolve())
    with tempfile.TemporaryDirectory(prefix="microduck-data-") as temp:
        config = Path(temp) / "data.yaml"
        config.write_text(yaml.safe_dump(d))
        YOLO("yolo26n.yaml").train(
            data=str(config),
            epochs=a.epochs,
            batch=a.batch,
            seed=a.seed,
            imgsz=640,
            pretrained=False,
            project=str(a.output.parent),
            name=a.output.name,
            exist_ok=False,
            device=0,
            workers=2,
            plots=True,
        )
print("Outputs:", a.output)
