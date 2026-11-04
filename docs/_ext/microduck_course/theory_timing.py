"""Teaching timelines: configured simulation periods, never fabricated measurements."""
from html import escape

def render(kind, lang):
    zh = lang == 'zh'
    t = lambda a,b: a if zh else b
    walk = kind == 'walking-theory'
    total = 60 if walk else 320
    rows = []
    def row(label, items):
        cells=''.join(f'<span class="cycle-block cycle-tone-{tone}" style="left:{start/total*100:.4f}%;width:{duration/total*100:.4f}%" title="{escape(text)}">{escape(text)}</span>' for start,duration,text,tone in items)
        rows.append(f'<div class="cycle-row"><strong>{label}</strong><div class="cycle-track">{cells}</div></div>')
    if walk:
        title=t('控制周期 · 50 Hz 行走闭环','Control cycle · 50 Hz walking loop')
        row(t('读取观测 → 策略推理','Read state → run policy'),[(i,20,t('第 ','Cycle ')+str(i//20+1)+t(' 拍',''),i//20) for i in range(0,60,20)])
        row(t('应用新的关节目标','Apply new joint targets'),[(i,20,'14 '+t('维动作','joint actions'),i//20) for i in range(0,60,20)])
        row(t('Newton 物理步进','Newton physics steps'),[(i,5,'5 ms',i//20) for i in range(0,60,5)])
        note=t('每 20 ms 读取反馈、计算并应用一个新动作；该动作保持四个 5 ms 物理步，再读取更新后的状态。区块表示配置的仿真周期，不是策略推理的实测耗时。','Every 20 ms, read feedback, compute and apply a new action. Hold it for four 5 ms physics steps, then read the updated state. Blocks show configured simulation periods, not measured inference duration.')
        detail=t('这里不包含球、球门或射门阶段。物理以 200 Hz 更新，策略以 50 Hz 更新；4.2 节射门的物理配置是 100 Hz，但策略仍为 50 Hz。','This loop has no ball, goal or shooting stages. Walking physics runs at 200 Hz and the policy at 50 Hz; section 4.2 uses 100 Hz physics with the same 50 Hz policy.')
    else:
        title=t('控制周期 · 分层执行','Control cycle · hierarchical execution')
        row(t('高层请求 · 12.5 Hz','High-level requests · 12.5 Hz'),[(i,80,t('请求 ','Request ')+str(i//80+1),i//80%3) for i in range(0,320,80)])
        row(t('底层策略 · 50 Hz','Low-level policy · 50 Hz'),[(i,20,'20',i//80%3) for i in range(0,320,20)])
        note=t('一次预测 16 条请求，只执行前 4 条，共 320 ms 仿真时间。每条高层请求下，底层策略仍根据新身体反馈更新 4 次。','Predict 16 requests, execute the first four over 320 ms of simulation time. For each request, the low-level policy uses fresh body feedback for four updates.')
        detail=t('同步示例：等待新推理返回，再执行这段请求。DGX Spark 服务端新推理约 300 ms，是实测中位数 332 ms 的取整示意，不属于上方仿真时间轴。RTC 拓展：执行当前块时提前推理下一块，以延迟感知的前缀约束衔接动作；不是当前录像已实现的功能，也不保证所有延迟都能被隐藏。','Synchronous example: wait for fresh inference, then execute these requests. About 300 ms of DGX Spark server computation is a rounded illustration of the measured 332 ms median, separate from the simulation-time axis above. RTC extention: predict the next chunk during current execution and connect chunks with a delay-aware prefix constraint. This is not implemented in the recording and does not guarantee all latency is hidden.')
    overview = '' if walk else '<div class="cycle-overview"><span>'+t('观察：图像 + 状态 + 指令','Observe: images + state + instruction')+'</span><b aria-hidden="true">→</b><span>'+t('GR00T 推理 ≈ 300 ms','GR00T inference ≈ 300 ms')+'<small>'+t('DGX Spark · 服务端中位数取整','DGX Spark · rounded server median')+'</small></span><b aria-hidden="true">→</b><span>'+t('执行前 4 条请求','Execute the first 4 requests')+'<small>'+t('4 × 80 ms 仿真时间','4 × 80 ms simulation time')+'</small></span><b aria-hidden="true">↻</b></div>'
    ticks=''.join(f'<span>{i} ms</span>' for i in ([0,20,40,60] if walk else [0,80,160,240,320]))
    return f'''<figure class="theory-cycle" data-theory-cycle="{kind}" data-duration="{total}">
<h3>{title}</h3>{overview}<div class="cycle-grid">{''.join(rows)}<div class="cycle-row"><span>{t('仿真时间','Simulation time')}</span><div class="cycle-ticks">{ticks}</div></div></div>
<div class="cycle-controls" hidden><button type="button" class="cycle-play">{t('播放','Play')}</button><label>{t('查看周期','Inspect cycle')} <input type="range" min="0" max="{total}" step="1" value="0"/></label><output>0 ms</output></div>
<figcaption>{note}</figcaption><details><summary>{t('详解','Details')}</summary><p>{detail}</p></details></figure>'''
