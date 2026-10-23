"""Reproduce the preserved short-training figure; requires Matplotlib and Noto CJK. No training.
python3 artwork/render_training_curves.py --output /path/to/training.svg
"""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R = Path(__file__).resolve().parents[1]
import argparse

p = argparse.ArgumentParser()
p.add_argument(
    "--data", type=Path, default=R / "validation/training-sample-curves-v2.json"
)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
data = json.loads(args.data.read_text())
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update(
    {
        "svg.fonttype": "path",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#a5b29e",
        "axes.labelcolor": "#35452e",
        "xtick.color": "#586650",
        "ytick.color": "#586650",
        "figure.facecolor": "#f6f8f5",
        "axes.facecolor": "#f6f8f5",
    }
)
fig, axes = plt.subplots(2, 1, figsize=(8, 7.4), layout="constrained")
for ax, key, title in zip(
    axes,
    ["Train/mean_reward", "Train/mean_episode_length"],
    ["平均回合回报", "平均回合长度 / 控制步"],
):
    rows = data[key]
    xs = [r["step"] + 1 for r in rows]
    ys = [r["value"] for r in rows]
    ax.plot(xs, ys, color="#608d30", linewidth=2.2, marker="o", markersize=4)
    ax.set_title(title, fontproperties=font, fontsize=17, loc="left", pad=14)
    ax.grid(axis="y", alpha=0.18)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("训练迭代次数", fontproperties=font, fontsize=13)
    ax.tick_params(labelsize=12)
fig.suptitle(
    "64 个环境 × 20 轮：一次短训练记录",
    fontproperties=font,
    fontsize=20,
    x=0.04,
    ha="left",
)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=300, bbox_inches="tight")
plt.close(fig)
print(
    "Saved authentic training curves",
    [(k, len(data[k])) for k in ["Train/mean_reward", "Train/mean_episode_length"]],
)
