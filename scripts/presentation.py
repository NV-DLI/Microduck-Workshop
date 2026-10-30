"""Presentation components kept separate from the lesson author's content.

All plots use recorded data. Media substitutions preserve source timing and claims.
"""
from copy import deepcopy
from html import escape as e
import json
from pathlib import Path
from presentation_components import control_timing, code_walkthrough

ROOT = Path(__file__).resolve().parents[1]


def prepare_sections(page, sections, lang):
    sections = deepcopy(sections)
    for section in sections:
        blocks = section["blocks"]
        if page == "07-groot" and section["id"] == "actions":
            blocks.append({"type": "control-timing"})
        if page == "07-groot" and section["id"] == "lab-environment":
            blocks.insert(1, {"type": "model-card", "model": "groot"})
        if page == "09-worlds" and section["id"] == "lab-cosmos-source":
            blocks.insert(1, {"type": "model-card", "model": "cosmos"})
        if page == "03-learning" and section["id"] == "read-curves":
            if lang == "en":
                blocks.append({"type": "training-chart"})
            blocks.append({"type": "tensorboard-screen"})
        if page == "04-terrain" and section["id"] == "evaluate":
            for i, b in enumerate(blocks):
                if b["type"] == "table":
                    blocks.insert(i, {"type": "benchmark-chart", "table": b})
                    break
        if page == "05-skills" and section["id"] == "community":
            indices = [i for i,b in enumerate(blocks) if b["type"] == "media"]
            if indices:
                blocks.insert(indices[-1]+1, {"type":"html",lang:"</div>"})
                blocks.insert(indices[0], {"type":"html",lang:'<div class="media-row">'})
    return sections


def training_chart(lang):
    zh = lang == "zh"
    data = json.loads((ROOT / "content/presentation/walk-training-scalars.json").read_text())
    metrics = [("Train/mean_reward", "平均回合回报" if zh else "Mean episode reward"),
               ("Train/mean_episode_length", "平均回合长度 / 控制步" if zh else "Mean episode length / control steps")]
    title = "64 个环境 · 20 轮短训练" if zh else "64 environments · 20 training iterations"
    panels = []
    for key, label in metrics:
        rows = data[key]
        values = [r["value"] for r in rows]
        low, high = min(0, min(values)), max(values)*1.1
        x = lambda step: 44 + step/19*300
        y = lambda value: 190-(value-low)/(high-low)*150
        grid = "".join(f'<path class="chart-grid" d="M44 {y(low+(high-low)*i/4):.2f}H344"/><text x="36" y="{y(low+(high-low)*i/4)+4:.2f}" text-anchor="end">{low+(high-low)*i/4:.1f}</text>' for i in range(5))
        axis = "".join(f'<text x="{x(i-1):.2f}" y="213" text-anchor="middle">{i}</text>' for i in (1,5,10,15,20))
        line = " ".join(f'{x(r["step"]):.2f},{y(r["value"]):.2f}' for r in rows)
        points = "".join(f'<circle class="chart-dot" cx="{x(r["step"]):.2f}" cy="{y(r["value"]):.2f}" r="3.5" tabindex="0" data-value="{e(label)} · {r["step"]+1}: {r["value"]:.4f}" aria-label="{e(label)}, {r["step"]+1}, {r["value"]:.4f}"><title>{r["step"]+1}: {r["value"]:.4f}</title></circle>' for r in rows)
        panels.append(f'<div class="chart-panel"><h4>{label}</h4><svg viewBox="0 0 370 242" role="group" aria-label="{label}">{grid}{axis}<polyline class="chart-line" points="{line}"/>{points}<text x="194" y="238" text-anchor="middle">{"训练迭代次数" if zh else "Training iteration"}</text></svg></div>')
    table = '<table><thead><tr><th>'+('迭代' if zh else 'Iteration')+'</th>'+''.join('<th>'+label+'</th>' for _,label in metrics)+'</tr></thead><tbody>'
    for i in range(20):
        table += f'<tr><td>{i+1}</td>'+''.join(f'<td>{data[key][i]["value"]:.6f}</td>' for key,_ in metrics)+'</tr>'
    table += '</tbody></table>'
    cap = ('DGX Spark 上的训练阶段约 13.1 秒，含保存与导出，不含环境初始化，也不是收敛耗时。第 1–20 轮对应 TensorBoard Step 0–19。原始日志，未平滑。回报上升不代表已经学会行走；该短训练模型在独立测试中很快跌倒。' if zh else 'The training stage took about 13.1 s on DGX Spark, including saving and export, excluding environment initialization; this is not time to convergence. Iterations 1–20 correspond to TensorBoard Steps 0–19. Raw values, without smoothing. Rising reward does not establish learned walking; this short-run model fell quickly in an independent test.')
    return f'<figure class="course-chart" data-chart><div class="chart-header"><h3>{title}</h3><span>{"训练阶段约 13.1 秒" if zh else "Training stage ≈ 13.1 s"}</span></div><div class="chart-grid-pair">{"".join(panels)}</div><p class="chart-readout" aria-live="polite">{"悬停、点选或用 Tab 选择数据点，查看原始数值。" if zh else "Hover, tap or Tab to a point to inspect its logged value."}</p><details class="chart-data"><summary>{"查看全部数据" if zh else "View all values"}</summary><div class="table-scroll">{table}</div></details><figcaption>{cap}</figcaption></figure>'


def render_presentation(b, lang):
    zh = lang == "zh"
    if b["type"] == "control-timing":
        return control_timing(lang)
    if b["type"] == "code":
        walkthrough = code_walkthrough(b, lang)
        if walkthrough:
            return walkthrough
    if b["type"] == "image" and b.get("file") == "media/indoor-mats.webp":
        labels = ["条纹足球场", "简洁足球场", "大理石", "棕色木地板"] if zh else ["Striped football pitch", "Plain football pitch", "Marble", "Brown wood floor"]
        cards = ''.join(f'<div class="scene-card"><h4>{letter} · {e(label)}</h4><a href="../assets/media/indoor-mat-{letter.lower()}.png"><img loading="lazy" src="../assets/media/indoor-mat-{letter.lower()}.png" alt="{e(label)}"></a></div>' for letter,label in zip('ABCD',labels))
        return f'<figure class="media"><div class="scene-card-grid">{cards}</div><figcaption>{b.get(lang, "")}</figcaption></figure>'
    media_config = ROOT / "content/presentation/media.json"
    if b["type"] == "media" and media_config.exists():
        config = json.loads(media_config.read_text()).get(b.get("file"))
        if config:
            prefix = "../assets/media/" + config["name"]
            stage = ' data-stages="../assets/media/' + config["stages"] + '"' if config.get("stages") else ""
            labels = ('<div class="comparison-labels"><span>' + ('上：官方策略' if zh else 'Top: official policy') + '</span><span>' + ('下：坡地策略' if zh else 'Bottom: terrain policy') + '</span></div>') if config.get("comparison") else ""
            return f'<figure class="media">{labels}<video controls playsinline preload="metadata" poster="{prefix}.jpg" aria-label="{e(b["alt"][lang])}"{stage}><source src="{prefix}.mp4" type="video/mp4"></video><figcaption>{b.get(lang, "")}</figcaption></figure>'
    if b["type"] == "training-chart" or (b["type"] == "image" and b.get("file") == "media/walk-training-curves-hd.svg"):
        return training_chart(lang)
    if b["type"] == "tensorboard-screen":
        path = "../assets/media/tensorboard-training-en.png"
        caption = ('TensorBoard 英文界面：对照 Step、Value 与训练曲线，再用独立运行验证行为。' if zh else 'TensorBoard: compare Step, Value and the training curves, then verify behavior in a separate rollout.')
        return f'<figure class="media"><a class="still-link" href="{path}"><img loading="lazy" src="{path}" alt="TensorBoard Scalars: Train/mean_reward and Train/mean_episode_length"></a><figcaption>{caption}</figcaption></figure>'
    if b["type"] == "model-card":
        groot = b["model"] == "groot"
        name = "GR00T-N1.7-3B" if groot else "Cosmos3-Nano"
        # Use the IDs from the installed learning configuration; no download is started here.
        if not groot:
            name = "Cosmos3-Nano"
        url = "https://huggingface.co/nvidia/" + name
        title = ('视觉语言动作基础模型' if zh else 'Vision-language-action base model') if groot else ('视频生成模型' if zh else 'Video generation model')
        facts = [("用途" if zh else "Purpose", "动作预测" if zh and groot else "视频外观迁移" if zh else "Action prediction" if groot else "Visual transfer"),
                 ("下载方式" if zh else "Download", "由训练命令下载" if zh and groot else "由生成命令下载" if zh else "Training command" if groot else "Generation command"),
                 ("访问" if zh else "Access", "Hugging Face · 本机登录" if zh else "Hugging Face · local login")]
        note = ('处理器资源来自 <a href="https://huggingface.co/nvidia/Cosmos-Reason2-2B">nvidia/Cosmos-Reason2-2B ↗</a>，请先按模型页面提示开通访问。基础权重需要任务示范微调。' if zh else 'Processor resources come from <a href="https://huggingface.co/nvidia/Cosmos-Reason2-2B">nvidia/Cosmos-Reason2-2B ↗</a>. Follow its access requirements. The base weights need task demonstrations for fine-tuning.') if groot else ('按下方命令准备模型；生成画面需检查后才能用于训练。' if zh else 'Prepare the model using the commands below; inspect generated frames before training.')
        return f'<article class="course-model-card"><div class="model-card-header"><div><p class="model-card-family">NVIDIA · {title}</p><h3 class="model-card-title">{name}</h3></div><a class="model-card-hub" href="{url}" target="_blank" rel="noopener">🤗 Hugging Face ↗</a></div><dl class="model-card-facts">'+''.join(f'<div><dt>{a}</dt><dd>{v}</dd></div>' for a,v in facts)+f'</dl><p class="model-card-note">{note}</p></article>'
    if b["type"] == "benchmark-chart":
        table = b["table"]
        rows = table["rows"][lang]
        bars = []
        import re
        for row in rows:
            counts = [re.search(r'(\d+)\s*/\s*(\d+)', cell) for cell in row[1:3]]
            if len(counts) != 2 or not all(counts):
                continue
            tracks = []
            for i,m in enumerate(counts):
                a,n = map(int,m.groups())
                tracks.append(f'<div class="benchmark-track" role="img" aria-label="{e(table["headers"][lang][i+1])}: {a}/{n}"><div class="benchmark-bar {"secondary" if i==0 else ""}" style="width:{100*a/n:.2f}%"></div><span>{a}/{n}</span></div>')
            bars.append(f'<div class="benchmark-row"><strong>{e(row[0])}</strong><div class="benchmark-bars">{"".join(tracks)}</div></div>')
        if not bars:
            return ""
        return '<figure class="course-chart"><div class="chart-header"><h3>'+('各坡度的任务通过数' if zh else 'Task passes by slope')+'</h3></div><div class="chart-legend"><span><i class="chart-swatch secondary"></i>'+e(table['headers'][lang][1])+'</span><span><i class="chart-swatch"></i>'+e(table['headers'][lang][2])+'</span></div>'+''.join(bars)+'<figcaption>'+('每格保留通过数与完整分母；通过需要满足该评测的完整条件，不能只看是否跌倒。' if zh else 'Pass counts retain their full denominators and require the full evaluation criteria, beyond avoiding falls.')+'</figcaption></figure>'
    return None
