"""Generate accessible SVG diagrams with explicit ports and orthogonal connectors."""
from html import escape as e
import hashlib, unicodedata

def lines(text, limit=23):
    result=[];line='';width=0
    for char in text:
        n=2 if unicodedata.east_asian_width(char) in ('W','F') else 1
        if width+n>limit and line:result.append(line);line='';width=0
        line+=char;width+=n
    if line:result.append(line)
    return result

def flow_svg(b):
    nodes=b['nodes'];n=len(nodes);uid='flow-'+hashlib.sha1(b['label'].encode()).hexdigest()[:9]
    six=n==6;loop=b.get('loop',False);height=436 if six else (264 if loop else 228)
    positions=[(24,64),(280,64),(536,64),(536,268),(280,268),(24,268)][:n]
    colors=[('#dfeaf5','#0969da'),('#e6f3d4','#4a7a00'),('#d6f1ee','#0d8078'),('#eadfff','#6f42c1'),('#e6f3d4','#4a7a00'),('#f7e9cf','#9a6700')]
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 {height}" role="img" aria-labelledby="{uid}-title {uid}-desc"><title id="{uid}-title">{e(b["label"])}</title><desc id="{uid}-desc">'+e('；'.join(a+'：'+c for a,c in nodes))+'</desc>',f'<defs><marker id="{uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L6 3.5L0 7" fill="none" stroke="#57606a" stroke-width="1.3"/></marker></defs>']
    groups=b.get('groups',[])
    if six:
        for y,text in [(20,groups[0]if groups else'任务执行'),(224,groups[1]if len(groups)>1 else'结果与反馈')]:
            out.append(f'<rect x="8" y="{y}" width="744" height="176" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-dasharray="4 4"/><text x="24" y="{y+25}" fill="#57606a" font-size="13">{e(text)}</text>')
    else:out.append(f'<text x="24" y="32" fill="#57606a" font-size="13">{e(b["label"])}</text>')
    labels=b.get('edges',[])
    for i in range(n-1):
        x,y=positions[i];nx,ny=positions[i+1]
        if y==ny:
            right=nx>x;start=x+200 if right else x;end=nx-6 if right else nx+206
            path=f'M{start} {y+56}H{end}';lx=(start+end)/2;ly=y+40
        else:path=f'M{x+100} {y+112}V{ny-6}';lx=x+112;ly=(y+112+ny)/2
        out.append(f'<path d="{path}" fill="none" stroke="#57606a" stroke-width="1.5" stroke-linejoin="round" marker-end="url(#{uid}-arrow)"/>')
        if i<len(labels)and labels[i]:out.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" fill="#57606a" font-size="12">{e(labels[i])}</text>')
    if loop and not six:
        out.append(f'<path d="M636 176V214Q636 222 628 222H132Q124 222 124 214V182" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#{uid}-arrow)"/><rect x="290" y="209" width="180" height="24" fill="white"/><text x="380" y="226" text-anchor="middle" fill="#57606a" font-size="13">{e(b.get("loop_label","下一次观测"))}</text>')
    for i,((title,body),(x,y)) in enumerate(zip(nodes,positions)):
        fill,stroke=colors[i];out.append(f'<rect x="{x}" y="{y}" width="200" height="112" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
        for j,t in enumerate(lines(title,23)):out.append(f'<text x="{x+100}" y="{y+29+j*21}" text-anchor="middle" fill="#24292f" font-size="16" font-weight="600">{e(t)}</text>')
        for j,t in enumerate(sum((lines(part,26) for part in body.split('，')),[])):out.append(f'<text x="{x+100}" y="{y+66+j*20}" text-anchor="middle" fill="#424a53" font-size="13">{e(t)}</text>')
    return ''.join(out)+'</svg>'

def platform_svg():
    return '''<svg xmlns="http://www.w3.org/2000/svg" class="architecture-svg" viewBox="0 0 760 384" role="img" aria-labelledby="platform-title platform-desc"><title id="platform-title">Isaac Lab、Newton 与 Isaac Sim 的职责</title><desc id="platform-desc">Isaac Lab 定义机器人学习任务，Newton 在 GPU 上推进物理状态。需要图像或交互界面时，由 Isaac Sim 提供渲染和传感器能力。Newton 物理执行不必依赖 Kit。</desc><defs><marker id="platform-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto-start-reverse"><path d="M0 0L6 3.5L0 7" fill="none" stroke="#57606a" stroke-width="1.3"/></marker></defs>
<rect x="16" y="16" width="728" height="352" rx="10" fill="#f6f8fa" stroke="#d0d7de" stroke-dasharray="4 4"/>
<text x="36" y="42" fill="#57606a" font-size="13">任务接口、物理后端与渲染能力分工</text>
<path marker-start="url(#platform-arrow)" d="M310 150V176Q310 184 302 184H180Q172 184 172 192V226" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<path d="M588 232V192Q588 184 580 184H458Q450 184 450 176V156" fill="none" stroke="#57606a" stroke-width="1.5" marker-end="url(#platform-arrow)"/>
<text x="176" y="174" fill="#57606a" font-size="13">动作 ⇄ 物理状态</text><text x="488" y="174" fill="#57606a" font-size="13">按需接入图像与界面</text>
<rect x="240" y="62" width="280" height="88" rx="7" fill="#e6f3d4" stroke="#4a7a00"/><text x="380" y="95" text-anchor="middle" fill="#24292f" font-size="20" font-weight="600">Isaac Lab</text><text x="380" y="124" text-anchor="middle" fill="#424a53" font-size="14">任务 · 观测 · 奖励 · 并行环境</text>
<rect x="40" y="232" width="264" height="94" rx="7" fill="#d6f1ee" stroke="#0d8078"/><text x="172" y="264" text-anchor="middle" fill="#24292f" font-size="19" font-weight="600">Newton</text><text x="172" y="292" text-anchor="middle" fill="#424a53" font-size="14">Warp / CUDA · 动力学与接触</text><text x="172" y="348" text-anchor="middle" fill="#57606a" font-size="13">物理执行可不依赖 Kit</text>
<rect x="456" y="232" width="264" height="94" rx="7" fill="#dfeaf5" stroke="#0969da"/><text x="588" y="264" text-anchor="middle" fill="#24292f" font-size="19" font-weight="600">Isaac Sim</text><text x="588" y="292" text-anchor="middle" fill="#424a53" font-size="14">OpenUSD · RTX · 相机与交互</text><text x="588" y="348" text-anchor="middle" fill="#57606a" font-size="13">本课程可视化任务使用此路径</text></svg>'''
