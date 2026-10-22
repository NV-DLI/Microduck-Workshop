"""Accessible, local course diagrams, mathematical notation and seekable media."""
from html import escape as e

def render_teaching_block(b,lang):
 k=b['type'];zh=lang=='zh';caption=b.get(lang,'')
 if k=='equation':
  return '<div class="formula"><div class="formula-label">'+e(b.get('label',''))+'</div><div>\\['+e(b['tex'])+'\\]</div></div>'
 if k=='sequence':
  cards=''.join('<div><a class="still-link" href="../assets/'+e(x['file'])+'"><img loading="lazy" src="../assets/'+e(x['file'])+'" alt="'+e(x['label'])+'"></a><p>'+e(x['label'])+'</p></div>' for x in b['items'])
  return '<figure class="media"><div class="pose-sequence" aria-label="'+e(b['label'])+'">'+cards+'</div><figcaption>'+caption+'</figcaption></figure>'
 if k=='image':
  return '<figure class="media"><a class="still-link" href="../assets/'+e(b['file'])+'" aria-label="'+('放大图片'if zh else'Enlarge image')+'"><img loading="lazy" src="../assets/'+e(b['file'])+'" alt="'+e(b.get('alt',{}).get(lang,caption))+'"></a><figcaption>'+caption+'</figcaption></figure>'
 if k=='media':
  f=b['file']
  if 'exploded-cycle' in f:return ''
  if f=='detections.gif':return render_teaching_block(dict(type='image',file='media/detection-contact-v2.webp',alt={'zh':'固定检测器在八张图像上的检测框与置信度','en':'Fixed detector boxes and scores on eight separate images'},zh='固定检测器的八组实际输出。每格展示一张图像，框旁标有类别与置信度。点击图片可放大，对比远近、视角和目标大小对检测结果的影响；这组示例不用于估计整体准确率。',en='Eight actual fixed-detector outputs on separate images, with class labels and confidence scores. Click to enlarge. These examples do not estimate overall accuracy.'),lang)
  name=f.removesuffix('.gif')+'-v2'
  return '<figure class="media"><video controls playsinline preload="metadata" poster="../assets/media/'+name+'.jpg" aria-label="'+e(b['alt'][lang])+'"><source src="../assets/media/'+name+'.mp4" type="video/mp4"></video><figcaption>'+caption+'</figcaption></figure>'
 if k=='flow':
  nodes=''.join('<div class="diagram-node"><strong>'+e(n[0])+'</strong><small>'+e(n[1])+'</small></div>' for n in b['nodes'])
  return '<figure class="concept-figure" aria-label="'+e(b['label'])+'"><div class="diagram-flow '+('flow-six' if len(b['nodes'])==6 else '')+'">'+nodes+'</div><figcaption class="diagram-caption">'+caption+'</figcaption></figure>'
 if k=='diagram':
  name=b['kind'];controls='';svg=''
  start='<svg viewBox="0 0 640 320" role="img" aria-label="'+e(b['label'])+'" font-family="system-ui, sans-serif" font-size="15"><defs><marker id="arrow-'+name+'" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#577642"/></marker></defs>'
  if name=='ppo':
   svg=start+'''<rect x="233" y="30" width="164" height="230" fill="#e3efd7" class="clip-band"/><path d="M64 30V260H574" stroke="#8a9981" fill="none"/><path class="ppo-original" d="" stroke="#87958b" stroke-width="2" stroke-dasharray="5 5" fill="none"/><path class="ppo-clipped" d="" stroke="#527d22" stroke-width="3" fill="none"/><circle class="ppo-point" r="6" fill="#d48125" stroke="white" stroke-width="2"/><text x="65" y="18">单个样本的代理目标</text><text x="58" y="281">0.5</text><text x="312" y="281">1.0</text><text x="565" y="281">1.5</text><text x="257" y="308">新旧动作概率之比 ρ</text><text class="ppo-low" x="205" y="50" fill="#526a41">0.8</text><text class="ppo-high" x="404" y="50" fill="#526a41">1.2</text><text class="ppo-ymax" x="54" y="38" text-anchor="end">1.6</text><text class="ppo-ymin" x="54" y="263" text-anchor="end">0.4</text></svg>'''
   controls='<div class="concept-controls"><label>优势 <select class="advantage"><option value="1">A = +1（好于预期）</option><option value="-1">A = −1（差于预期）</option></select></label><label>概率比 ρ <input class="prob-ratio" type="range" min="0.5" max="1.5" step="0.01" value="1.3"></label><label>裁剪范围 ε <input class="clip-epsilon" type="range" min="0.05" max="0.4" step="0.05" value="0.2"></label></div><p class="concept-output" aria-live="polite"></p>'
  elif name=='timing':
   svg=start+'''<text x="36" y="43">策略：每 20 ms 输出一次动作（50 Hz）</text><rect x="60" y="70" width="520" height="52" rx="5" fill="#e6f0da" stroke="#91aa79"/><text x="270" y="102">aₜ：关节目标</text><path d="M320 123V169" stroke="#65844f" marker-end="url(#arrow-timing)"/><text x="36" y="156">物理：每 5 ms 推进一步（200 Hz）</text>'''+''.join(f'<rect x="{60+i*130}" y="182" width="124" height="48" rx="4" fill="white" stroke="#91aa79"/><text x="{85+i*130}" y="213">第 {i+1} 步</text><text x="{60+i*130}" y="255">{i*5} ms</text>'for i in range(4))+'''<text x="550" y="255">20 ms</text><text x="90" y="298">动作保持不变；位置、速度与接触力仍在逐步变化</text></svg>'''
  elif name=='projection':
   svg=start+'''<path d="M45 265H610" stroke="#768b64" stroke-width="2"/><text x="495" y="292">地面 z = 0</text><circle cx="102" cy="65" r="8" fill="#4d742e"/><text x="28" y="40">相机原点 o</text><path d="M102 65L475 265" stroke="#5f873d" stroke-width="2" marker-end="url(#arrow-projection)"/><path d="M205 65V210" stroke="#8e9c86" stroke-width="3"/><text x="180" y="237">图像平面</text><circle cx="205" cy="120" r="6" fill="#d48125"/><text x="230" y="105">检测像素 (u, v)</text><circle cx="475" cy="265" r="8" fill="#d48125"/><text x="410" y="235">地面交点 p</text><path d="M102 75V265H460" stroke="#a3afa0" stroke-dasharray="5 5" fill="none"/><text x="320" y="155">沿射线 d 延伸</text><text x="25" y="285">地面投影</text></svg>'''
  elif name=='stance':
   svg=start+'''<path d="M60 270H570M60 270V35" stroke="#8b9b82" fill="none"/><text x="579" y="275">x</text><text x="53" y="25">y</text><path d="M234 196L545 72" stroke="#527b32" stroke-width="2" marker-end="url(#arrow-stance)"/><rect x="526" y="37" width="22" height="76" rx="2" stroke="#4d6c34" fill="none" stroke-width="3"/><text x="550" y="65">球门 g</text><circle cx="353" cy="149" r="12" fill="#d48125"/><text x="363" y="182">足球 b</text><circle class="stance-robot" cx="234" cy="224" r="14" fill="#406c8c"/><text class="stance-label" x="117" y="246">出脚位置 p*</text><path class="stance-offset" d="M234 196L234 224" stroke="#406c8c" stroke-dasharray="4 4"/><text x="180" y="162">沿射门方向后退 d</text><text x="305" y="110">单位方向 u</text><text x="88" y="303">俯视示意：先留出出脚距离，再调整横向偏移</text></svg>'''
   controls='<div class="concept-controls"><label>后退距离 d / m <input class="stance-distance" type="range" min="0.12" max="0.5" step="0.01" value="0.26"></label><label>横向偏移 ℓ / m <input class="stance-lateral" type="range" min="-0.14" max="0.14" step="0.01" value="0.04"></label></div><p class="concept-output" aria-live="polite"></p>'
  else:raise ValueError('Unknown diagram '+name)
  return '<figure class="concept-figure" data-diagram="'+name+'">'+svg+controls+'<figcaption class="diagram-caption">'+caption+'</figcaption></figure>'
 return None
