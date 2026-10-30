"""Editable lesson illustrations with separate simulation and wall clocks."""
from html import escape as e


def control_timing(lang):
    zh = lang == 'zh'
    def t(a,b): return a if zh else b
    lanes = [
        ('camera', t('相机 + ToF','Camera + ToF'), t('约 15 Hz · 每 13 / 14 个物理步','≈15 Hz · every 13 / 14 physics ticks')),
        ('decision', t('高层决策','High-level decision'), t('传统：检测、几何与任务状态','Traditional: detection, geometry, task state')),
        ('actor', t('低层动作策略','Low-level actor'), '50 Hz · 20 ms'),
        ('physics', 'Newton', '200 Hz · 5 ms'),
    ]
    rows=''.join(f'<div class="timing-lane" data-lane="{key}"><div><strong>{name}</strong><small>{sub}</small></div><div class="timing-track" aria-hidden="true"><span class="timing-cursor"></span></div></div>' for key,name,sub in lanes)
    facts=[
        t('传统方法：相机与 ToF 更新后，经检测、几何估计和任务状态选择命令；Nav2 按任务阶段调用。低层策略每 20 ms 输出关节动作，每次覆盖 4 个 Newton 物理步。','Traditional control: camera and ToF updates feed detection, geometry and task-state logic. Nav2 is called by task stage. The actor produces joint actions every 20 ms, covering four Newton physics ticks.'),
        t('GR00T：4 帧因果历史 RGB + 127 维状态 + 指令 → 远端 HTTP 推理 → 16 × 8 动作块。只取前 4 个动作，每个保持 80 ms；其余 12 个丢弃。8 维高层动作经技能与交接检查，再进入 50 Hz 低层策略，不是 14 个关节角。','GR00T: four causal RGB history frames, a 127-D state and an instruction → remote HTTP inference → a 16 × 8 action chunk. Keep its first four actions, hold each for 80 ms, and discard the remaining twelve. The 8-D high-level action passes skill and handoff checks before the 50 Hz actor; it is not fourteen joint angles.'),
        t('每 80 ms 仿真时间请求一次动作；服务器每 4 次请求产生一个新块，其余请求取队列。4 帧图像距当前为 0、1.92、3.84、5.76 秒，起始不足时重复首帧。状态为当前 58 维 + 3 × 23 维历史。','An action request occurs every 80 ms of simulation time. The server predicts a new chunk every four requests and otherwise serves its queue. Images are aged 0, 1.92, 3.84 and 5.76 s; initialization repeats the first frame. State contains 58 current values plus 3 × 23 history values.'),
        t('本课同步评估在等待 HTTP 返回时暂停仿真。0.32 秒是一个块实际使用部分覆盖的仿真时间，不是墙钟周期。真实部署若要持续控制，需要单独设计异步执行与超时处理。','The synchronous evaluator pauses simulation while awaiting HTTP. The used part of a chunk covers 0.32 s of simulation time; that is not a wall-clock cycle. Continuous deployment requires a separate asynchronous execution and timeout design.'),
    ]
    controls=f'''<div class="timing-controls"><label>{t('控制方式','Controller')}<select class="timing-mode"><option value="traditional">{t('传统方法','Traditional')}</option><option value="groot">GR00T</option></select></label><label>{t('一次新预测的等待示意','Illustrative wait for a new prediction')}<input class="timing-delay" type="range" min="0.1" max="1" step="0.01" value="0.33"><output class="timing-delay-value">330 ms</output></label><nve-button size="sm" class="timing-replay">{t('播放一个周期','Play one cycle')}</nve-button></div>'''
    return f'''<figure class="course-chart control-timing" data-control-timing><div class="chart-header"><h3>{t('一次控制周期，两个时钟','One control cycle, two clocks')}</h3><span>{t('结构示意 · 网格为仿真时间刻度','Structure · grid marks simulation time')}</span></div>{controls}<div class="timing-clocks"><span>{t('仿真时间','Simulation')} <output class="sim-clock">0 ms</output></span><span>{t('示意墙钟','Illustrative wall clock')} <output class="wall-clock">0 ms</output></span></div>{rows}<p class="timing-status" aria-live="polite">{t('选择控制方式，再播放或拖动下方进度。','Choose a controller, then play or scrub the progress below.')}</p><label class="timing-scrub-label">{t('周期进度','Cycle progress')}<input class="timing-scrub" type="range" min="0" max="1000" value="0"></label><ol class="timing-facts">{''.join('<li>'+v+'</li>' for v in facts)}</ol><figcaption>{t('参考实测：899 次新块预测（含 10 次首请求）的服务端计时中位数为 332 ms，P95 为 351 ms，只含预处理、动作选择和后处理；不含 PNG 解码、网络、客户端编码与画面采集。滑块控制示意等待；队列请求和传统方法的端到端耗时未计入，不能据此比较速度。','Reference measurement: 899 new-chunk predictions, including ten first requests, had a server-side median of 332 ms and P95 of 351 ms. This covers preprocessing, action selection and postprocessing, excluding PNG decode, network, client encoding and capture. The slider sets an illustrative wait; queued requests and traditional end-to-end costs are omitted, so this is not a speed comparison.')}</figcaption></figure>'''


def code_walkthrough(b, lang):
    code = b.get('code','')
    zh = lang == 'zh'
    def t(a,b): return a if zh else b
    if code.startswith('action = policy('):
        cuts=[(0,3,t('组装观察，再运行策略','Assemble observations and run the policy'),t('更新观察历史，把 policy 观察送入运动策略。','Update observation history and feed the policy observations to the movement policy.')),(3,4,t('处理策略动作','Process the policy action'),t('动作管理器接收本次策略输出，准备驱动仿真。','The action manager receives this policy output for application in simulation.')),(4,9,t('推进 4 个物理步','Advance four physics ticks'),t('策略动作跨越 4 × 5 ms；每步写入动作并更新场景，形成 50 Hz 控制与 200 Hz 物理的关系。','Hold this policy action across 4 × 5 ms; write actions and update the scene each tick, giving 50 Hz control and 200 Hz physics.'))]
    elif code.startswith('base = gym.make('):
        cuts=[(0,2,t('创建训练环境','Create the training environment'),t('构建任务并包装向量环境，让算法同时读取多个环境的观察和奖励。','Build the task and wrap vector environments so the algorithm can consume observations and rewards from many environments.')),(2,5,t('连接 PPO 训练器','Connect the PPO runner'),t('交给训练器环境、算法参数、输出目录和运行设备。','Pass the environment, algorithm settings, output directory and device to the runner.')),(5,8,t('开始有限轮次的学习','Learn for a fixed iteration budget'),t('迭代次数决定本次预算；随机初始回合长度可分散各环境的终止时刻。','The iteration count sets the budget. Random initial episode lengths spread termination times across environments.'))]
    elif code.startswith('ratio = torch.exp('):
        cuts=[(0,3,t('计算新旧策略概率比','Compute the probability ratio'),t('对数概率相减再取指数，得到同一动作在新旧策略下的概率比。','Exponentiate the log-probability difference to compare the same action under the new and old policies.')),(3,7,t('限制更新幅度','Limit the update'),t('优势估计与概率比共同决定代理目标；裁剪概率比，限制过大的策略变化。','Advantages and the probability ratio define the surrogate. Clipping the ratio limits overly large policy changes.')),(7,8,t('聚合损失','Aggregate the loss'),t('代码在负号形式的损失中取较大项，再对样本求平均；优化器最小化这个损失。','With the negative-loss convention, select the larger term and average over samples. The optimizer minimizes this loss.'))]
    else: return None
    lines=code.splitlines(); cards=[]; chunks=[]
    for i,(a,z,title,body) in enumerate(cuts):
        snippet=e('\n'.join(lines[a:z]))
        cards.append(f'<article class="story-step" data-story-step="{i}" tabindex="0"><h4><span>{i+1}</span>{title}</h4><p>{body}</p><pre class="story-mobile-code"><code class="language-python">{snippet}</code></pre></article>')
        chunks.append(f'<div class="story-code-chunk" data-story-target="{i}"><span class="story-code-number">{i+1}</span><pre><code class="language-python">{snippet}</code></pre></div>')
    link=f'<a class="source-link" href="{e(b["source"])}" target="_blank" rel="noopener">{t("完整代码 ↗","Full source ↗")}</a>' if b.get('source') else ''
    return f'<div class="code-story" data-story><div class="story-steps">{"".join(cards)}<p class="story-mobile-source">{link}</p></div><div class="story-code"><div class="code-toolbar"><span>{t("Python · 核心代码节选","Python · source excerpt")}</span>{link}</div>{"".join(chunks)}</div></div>'
