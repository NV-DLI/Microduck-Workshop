"""Teaching timelines for configured periods; host inference is shown separately."""
from html import escape


def render(kind, lang):
    zh = lang == 'zh'
    t = lambda a, b: a if zh else b
    if kind == 'comparison-theory':
        return ('<div class="cycle-comparison" data-cycle-comparison>'
                '<div class="cycle-tabs" hidden role="group" aria-label="' + t('选择控制方式', 'Select controller') + '">'
                '<button type="button" data-cycle-choice="0" aria-pressed="true">' + t('板载策略 · 4.3', 'Onboard policy · 4.3') + '</button>'
                '<button type="button" data-cycle-choice="1" aria-pressed="false">GR00T</button></div>'
                + render('onboard-theory', lang) + render('groot-theory', lang) + '</div>')
    walk = kind == 'walking-theory'
    onboard = kind == 'onboard-theory'
    total = 60 if walk else 200 if onboard else 320
    rows = []

    def row(label, items):
        cells = ''.join(f'<span class="cycle-block cycle-tone-{tone}" style="left:{start/total*100:.4f}%;width:{duration/total*100:.4f}%" title="{escape(text)}">{escape(text)}</span>' for start, duration, text, tone in items)
        rows.append(f'<div class="cycle-row"><strong>{label}</strong><div class="cycle-track">{cells}</div></div>')

    def observations(label, times):
        cells = ''.join(f'<span class="cycle-observation" style="left:{start/total*100:.4f}%" title="{escape(t("采集新观察", "Capture a new observation"))}"><i></i><small>{i+1}</small></span>' for i, start in enumerate(times))
        rows.append(f'<div class="cycle-row"><strong>{label}</strong><div class="cycle-track cycle-observations">{cells}</div></div>')

    overview = ''
    if walk:
        title = t('控制周期 · 50 Hz 行走闭环', 'Control cycle · 50 Hz walking loop')
        row(t('读取观测 → 策略推理', 'Read state → run policy'), [(i,20,t('周期 ', 'Cycle ')+str(i//20+1),i//20) for i in range(0,60,20)])
        row(t('应用新的关节目标', 'Apply new joint targets'), [(i,20,'14 '+t('维动作', 'joint actions'),i//20) for i in range(0,60,20)])
        row(t('Newton 物理步进', 'Newton physics steps'), [(i,5,'5 ms',i//20) for i in range(0,60,5)])
        note = t('每 20 ms 读取反馈、计算并应用一个新动作；该动作保持四个 5 ms 物理步，再读取更新后的状态。区块表示配置的仿真周期，不是策略推理的实测耗时。', 'Every 20 ms, read feedback, compute and apply a new action. Hold it for four 5 ms physics steps, then read the updated state. Blocks show configured simulation periods, not measured inference duration.')
        detail = t('这里不包含球、球门或射门阶段。物理以 200 Hz 更新，策略以 50 Hz 更新；4.3 节射门的物理配置是 100 Hz，但策略仍为 50 Hz。', 'This loop has no ball, goal or shooting stages. Walking physics runs at 200 Hz and the policy at 50 Hz; section 4.3 uses 100 Hz physics with the same 50 Hz policy.')
    elif onboard:
        title = t('控制周期 · 板载策略', 'Control cycle · onboard policy')
        observations(t('RGB + ToF · 15 Hz', 'RGB + ToF · 15 Hz'), [0,70,140])
        row(t('运动请求 · 50 Hz', 'Motion requests · 50 Hz'), [(i,20,str(i//20+1),i//70%3) for i in range(0,200,20)])
        row(t('身体反馈 → 底层策略 · 50 Hz', 'Body feedback → low-level policy · 50 Hz'), [(i,20,'20',i//70%3) for i in range(0,200,20)])
        row(t('Newton 物理步进 · 100 Hz', 'Newton physics · 100 Hz'), [(i,10,'',i//70%3) for i in range(0,200,10)])
        note = t('RGB 与 ToF 约每 67 ms 采集一次；运动请求和底层策略每 20 ms 更新，期间使用最近有效的感知信息与新的身体反馈。一个控制周期包含两个 10 ms 物理步。', 'RGB and ToF are captured about every 67 ms. Motion requests and the low-level policy update every 20 ms using the latest valid perception and fresh body feedback. Each control cycle contains two 10 ms physics steps.')
        detail = t('这是配置周期示意；0 ms 取自稳态的一次采集。菱形表示采集时刻，不是渲染耗时；15 Hz 采集对齐 100 Hz 物理步后，间隔为 60 或 70 ms。阶段观测随新采集刷新，运动请求每 20 ms 更新。阶段逻辑依据有效的球位、ToF 与身体状态生成移动或踢球请求，例如“向前走并逐渐减速”。底层策略是训练好的小型神经网络，例如官方行走 Actor：它读取当前关节角、角速度和重力方向，将请求变为 14 个关节动作。新的身体反馈每个控制周期都会更新，不必等下一张图像。', 'This schematic shows configured periods, with 0 ms at a steady-state capture. Diamonds mark capture instants, not rendering durations. Aligning 15 Hz capture to 100 Hz physics gives intervals of 60 or 70 ms. Stage observations refresh with each capture; motion requests update every 20 ms. Stage logic uses valid ball estimates, ToF and body state to request motion or a kick, such as “move forward while slowing down”. The low-level policy is a trained small neural network, such as the official walking Actor: it reads current joints, angular velocity and gravity direction and turns the request into 14 joint actions. Body feedback is updated every control cycle without waiting for the next image.')
    else:
        title = t('控制周期 · 分层执行', 'Control cycle · hierarchical execution')
        observations(t('RGB 采集 · 15 Hz', 'RGB capture · 15 Hz'), [0, 200/3, 400/3, 200, 800/3])
        row(t('整理观察 · 12.5 Hz', 'Assemble observations · 12.5 Hz'), [(i,80,t('图像 + 状态', 'Images + state'),i//80%3) for i in range(0,320,80)])
        row(t('高层请求 · 12.5 Hz', 'High-level requests · 12.5 Hz'), [(i,80,t('请求 ', 'Request ')+str(i//80+1),i//80%3) for i in range(0,320,80)])
        row(t('身体反馈 → 底层策略 · 50 Hz', 'Body feedback → low-level policy · 50 Hz'), [(i,20,'20',i//80%3) for i in range(0,320,20)])
        note = t('一次预测 16 条请求，只执行前 4 条，共 320 ms 仿真时间。每条高层请求下，底层策略仍根据新的身体反馈更新 4 次；执行完这一段，再用新观察预测下一段。', 'Predict 16 requests and execute the first four over 320 ms of simulation time. For each request, the low-level policy uses fresh body feedback for four updates. After this segment, predict the next one from new observations.')
        detail = t('高层请求说明“想做什么”：例如向前移动、向右转、低头看球，或请求右脚踢球。GR00T 输出的 8 个数就是这些请求，不是 14 个关节角。底层策略说明“怎样落实”：例如行走 Actor 每 20 ms 根据关节和 IMU 反馈调整 14 个关节，以维持平衡并执行移动请求。观察在每个 80 ms 请求步整理，但只有预测新动作块时才运行一次新的 GR00T 推理；其余步使用缓存请求。图像来自最近的相机采集，ToF 只供观察、不输入 GR00T。同步流程先等待推理，再执行请求。RTC 可在执行当前块时提前预测下一块；当前录像尚未接入 RTC。', 'High-level requests describe what to do: move forward, turn right, look down at the ball, or request a right-foot kick. The eight GR00T outputs are these requests, not 14 joint angles. A low-level policy handles how to act: for example, the walking Actor adjusts 14 joints every 20 ms using joint and IMU feedback to balance and follow the motion request. Observations are assembled at each 80 ms request step, but fresh GR00T inference runs only for a new action chunk; other steps use cached requests. Images come from the latest camera captures. ToF is displayed but is not a GR00T input. The synchronous flow waits for inference before executing requests. RTC can predict the next chunk during execution; it is not integrated in the recording.')
        overview = '<div class="cycle-overview"><span>'+t('观察：图像 + 状态 + 指令','Observe: images + state + instruction')+'</span><b aria-hidden="true">→</b><span>'+t('GR00T 推理 ≈ 300 ms','GR00T inference ≈ 300 ms')+'<small>'+t('DGX Spark · 服务端中位数','DGX Spark · server median')+'</small></span><b aria-hidden="true">→</b><span>'+t('执行前 4 条请求','Execute the first 4 requests')+'<small>'+t('4 × 80 ms 仿真时间','4 × 80 ms simulation time')+'</small></span></div><div class="cycle-repeat"><svg viewBox="0 0 500 30" aria-hidden="true"><path d="M485 2 V15 H15 V2 M8 9 L15 2 L22 9" fill="none" stroke="currentColor" stroke-width="2"/></svg><span>'+t('执行结束 → 获取新观察 → 下一控制循环','Finish execution → observe again → next control loop')+'</span></div>'
    ticks = ''.join(f'<span>{i} ms</span>' for i in ([0,20,40,60] if walk else [0,50,100,150,200] if onboard else [0,80,160,240,320]))
    return f'''<figure class="theory-cycle" data-theory-cycle="{kind}" data-duration="{total}">
<h3>{title}</h3>{overview}<div class="cycle-grid">{''.join(rows)}<div class="cycle-row"><span>{t('仿真时间','Simulation time')}</span><div class="cycle-ticks">{ticks}</div></div></div>
<div class="cycle-controls" hidden><button type="button" class="cycle-play">{t('播放','Play')}</button><label>{t('查看周期','Inspect cycle')} <input type="range" min="0" max="{total}" step="1" value="0"/></label><output>0 ms</output></div>
<figcaption>{note}</figcaption><details><summary>{t('详解','Details')}</summary><p>{detail}</p></details></figure>'''
