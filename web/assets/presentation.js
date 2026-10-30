(() => {
  const zh = document.body.dataset.lang === "zh";
  const root = document.documentElement;
  const theme = document.querySelector(".theme-toggle");
  // One fixed slotted icon; the label names the theme the button switches to.
  function refreshTheme() {
    const dark = root.getAttribute("nve-theme") === "dark";
    const label = zh ? (dark ? "切换为浅色主题" : "切换为深色主题") : (dark ? "Switch to light theme" : "Switch to dark theme");
    theme.setAttribute("aria-label", label);
    theme.setAttribute("title", label);
  }
  theme?.addEventListener("click", () => {
    const next = root.getAttribute("nve-theme") === "dark" ? "light" : "dark";
    root.setAttribute("nve-theme", next);
    try { localStorage.setItem("microduck-theme", next); } catch {}
    refreshTheme();
  });
  if (theme) refreshTheme();
  // Expose each diagram's viewBox width so CSS can cap it at reading size.
  for (const svg of document.querySelectorAll("main figure svg[viewBox]")) {
    const width = Number(svg.getAttribute("viewBox").trim().split(/[\s,]+/)[2]);
    if (!(width > 48)) continue;
    svg.style.setProperty("--vb-w", width);
    const figure = svg.closest("figure");
    if (figure && !figure.style.getPropertyValue("--vb-w")) figure.style.setProperty("--vb-w", width);
  }
  for (const chart of document.querySelectorAll("[data-chart]")) {
    const output = chart.querySelector(".chart-readout");
    for (const point of chart.querySelectorAll(".chart-dot")) {
      const show = () => { output.textContent = point.dataset.value; };
      for (const event of ["pointerenter", "focus", "click"]) point.addEventListener(event, show);
    }
  }
})();

(() => {
  const zh = document.body.dataset.lang === 'zh';
  for (const root of document.querySelectorAll('[data-story]')) {
    const steps=[...root.querySelectorAll('[data-story-step]')];
    const activate = (index) => {
      steps.forEach(s=>s.classList.toggle('is-active',s.dataset.storyStep===index));
      root.querySelectorAll('[data-story-target]').forEach(t=>t.classList.toggle('is-active',t.dataset.storyTarget===index));
      root.querySelectorAll('.story-edge').forEach(e=>e.classList.toggle('is-active', e.dataset.from===index||e.dataset.to===index));
    };
    activate('0');
    steps.forEach(s=>['pointerenter','focus','click'].forEach(event=>s.addEventListener(event,()=>activate(s.dataset.storyStep))));
    // A code chunk shares its row with its explanation; pointing at either selects the pair.
    root.querySelectorAll('.story-code-chunk[data-story-target]').forEach(c=>['pointerenter','click'].forEach(event=>c.addEventListener(event,()=>activate(c.dataset.storyTarget))));
    root.querySelectorAll('.story-node').forEach(n=>{
      n.addEventListener('click',()=>activate(n.dataset.storyTarget));
      n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate(n.dataset.storyTarget);}});
    });
    const observer=new IntersectionObserver(entries=>{
      // Keep the student's hovered or keyboard-focused step in control while scrolling.
      if (root.querySelector('.story-step:hover, .story-step:focus-visible, .story-code-chunk:hover')) return;
      const visible=entries.filter(x=>x.isIntersecting);
      if(visible.length) activate(visible[visible.length-1].target.dataset.storyStep);
    },{rootMargin:'-20% 0px -55% 0px'});
    steps.forEach(s=>observer.observe(s));
  }
  for (const root of document.querySelectorAll('[data-control-timing]')) {
    const mode=root.querySelector('.timing-mode'), delay=root.querySelector('.timing-delay'), scrub=root.querySelector('.timing-scrub'), play=root.querySelector('.timing-replay');
    let frame=0, start=0, running=false;
    const label=(a,b)=>zh?a:b;
    function render(){
      const groot=mode.value==='groot', wait=groot?Number(delay.value)*1000:0;
      const wall=Number(scrub.value)/1000*(320+wait), sim=Math.max(0,wall-wait), waiting=groot&&wall<wait;
      root.classList.toggle('is-waiting',waiting);
      delay.disabled=!groot;
      root.querySelector('.timing-delay-value').textContent=Math.round(Number(delay.value)*1000)+' ms';
      root.querySelector('.sim-clock').textContent=Math.round(sim)+' ms';
      root.querySelector('.wall-clock').textContent=Math.round(wall)+' ms';
      root.querySelector('[data-lane=decision] small').textContent=groot?label('12.5 Hz 请求 · 每 4 次新预测','12.5 Hz requests · predict every four'):label('检测、几何与任务状态','Detection, geometry, task state');
      root.querySelector('[data-lane=camera] strong').textContent=groot?label('相机 / 本体状态','Camera / robot state'):label('相机 + ToF','Camera + ToF');
      root.querySelector('[data-lane=camera] small').textContent=groot?label('4 帧因果历史 + 127 维状态','4 causal images + 127-D state'):label('约 15 Hz · 每 13 / 14 个物理步','≈15 Hz · every 13 / 14 physics ticks');
      root.querySelectorAll('.timing-lane').forEach(lane=>{
        const type=lane.dataset.lane;
        let time=sim;
        if(type==='physics') time=Math.floor(sim/5)*5;
        if(type==='actor') time=Math.floor(sim/20)*20;
        if(type==='decision'&&groot) time=Math.min(240,Math.floor(sim/80)*80);
        if(type==='camera') time=groot?Math.min(240,Math.floor(sim/80)*80):Math.ceil(Math.floor(sim/(1000/15))*200/15)*5;
        lane.querySelector('.timing-cursor').style.left=(time/320*100)+'%';
      });
      const action=Math.min(4,1+Math.floor(sim/80));
      root.querySelector('.timing-status').textContent=waiting?label('等待新动作块：HTTP 请求阻塞，两个低层仿真循环都暂停。','Awaiting a new chunk: HTTP blocks; both low-level simulation loops pause.'):
        groot?label(`使用动作 ${action}/4；低层控制与物理步恢复。队列请求额外耗时未画入。`,`Use action ${action}/4; actor and physics ticks resume. Additional queue-request costs are omitted.`):label('传感器 → 检测与几何 → 任务决策 → 低层策略 → 物理步 → 新观察。计算耗时未测量，示意中未计入。','Sensors → detection and geometry → task decision → actor → physics → new observation. Unmeasured compute costs are omitted.');
    }
    function stop(){cancelAnimationFrame(frame);running=false;play.textContent=label('播放一个周期','Play one cycle');}
    function tick(now){if(!running)return;scrub.value=Math.min(1000,(now-start)/5);render();if(Number(scrub.value)>=1000)stop();else frame=requestAnimationFrame(tick);}
    play.addEventListener('click',()=>{if(running){stop();return;}running=true;start=performance.now();play.textContent=label('暂停','Pause');frame=requestAnimationFrame(tick);});
    [mode,delay,scrub].forEach(el=>el.addEventListener('input',()=>{stop();render();}));
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
    render();
  }
})();
