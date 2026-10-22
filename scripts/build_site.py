#!/usr/bin/env python3
"""Build a local-first, bilingual static course with no server-side dependencies."""
from pathlib import Path
from html import escape as e
import json,re
ROOT=Path(__file__).resolve().parents[1]
PAGES=json.loads((ROOT/'content/course.json').read_text())
I18N={
'zh':dict(lang='zh-CN',course='Microduck 机器人学习实战',search='搜索课程',menu='目录',onpage='本页内容',prev='上一页',next='下一页',copy='复制命令',copied='已复制',learn='开始学习',refs='参考资料',check='检查理解',answer='查看解析',mark='标记已学完',marked='已学完',reset='重置',swap='English',overview='课程概览',skip='跳至正文'),
'en':dict(lang='en',course='Robot Learning with Microduck',search='Search the course',menu='Contents',onpage='On this page',prev='Previous',next='Next',copy='Copy command',copied='Copied',learn='Start learning',refs='References',check='Check your understanding',answer='Show explanation',mark='Mark complete',marked='Completed',reset='Reset',swap='中文',overview='Course overview',skip='Skip to content')}
def render_block(b,lang,t):
 k=b['type'];text=b.get(lang,b.get('text',''))
 if k=='html':return text
 if k=='p':return '<p>'+text+'</p>'
 if k=='note':return '<aside class="note"><strong>'+b['title'][lang]+'</strong><p>'+text+'</p></aside>'
 if k=='code':return '<div class="code-block"><div class="code-toolbar"><span>'+e(b.get('label','Terminal · Bash'))+'</span><button type="button" class="copy" data-done="'+t['copied']+'">'+t['copy']+'</button></div><pre><code>'+e(b['code'])+'</code></pre></div>'
 if k=='list':return '<ul>'+''.join('<li>'+x+'</li>' for x in b[lang])+'</ul>'
 if k=='table':return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>'for x in b['headers'][lang])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>'for x in row)+'</tr>'for row in b['rows'][lang])+'</tbody></table></div>'
 if k=='quiz':return '<div class="quiz"><h3>'+t['check']+'</h3><p>'+b['question'][lang]+'</p><details><summary>'+t['answer']+'</summary><p>'+b['answer'][lang]+'</p></details></div>'
 if k=='media':return '<figure class="media"><div class="motion" data-gif="../assets/media/'+e(b['file'])+'"><img loading="lazy" src="../assets/media/'+e(b['poster'])+'" alt="'+e(b['alt'][lang])+'"><button type="button" class="play-motion" aria-pressed="false">'+('播放过程'if lang=='zh'else'Play demonstration')+'</button></div><figcaption>'+text+'</figcaption></figure>'
 if k=='anatomy':return '''<div class="anatomy" data-robot-viewer><div class="viewer-stage"><img class="robot-fallback" src="../assets/robot/assembled.png" alt="Microduck"><div class="viewer-status" role="status"></div><button type="button" class="load-robot">'''+('打开交互模型' if lang=='zh' else 'Open interactive model')+'''</button></div><div class="viewer-toolbar"><label>'''+('装配'if lang=='zh'else'Assembled')+''' <input type="range" min="0" max="100" value="0" class="explode" aria-label="'''+('爆炸图展开程度'if lang=='zh'else'Explosion amount')+'''"> '''+('拆解'if lang=='zh'else'Exploded')+'''</label><button type="button" class="view-reset">'''+t['reset']+'''</button><button type="button" class="view-focus" disabled>'''+('查看部件'if lang=='zh'else'Focus part')+'''</button></div><div class="component-info" aria-live="polite"><strong>'''+('旋转、缩放，点击一个组件'if lang=='zh'else'Rotate, zoom, and select a component')+'''</strong><p>'''+('左键拖动旋转，滚轮缩放；也可使用下方组件列表。'if lang=='zh'else'Drag to orbit and scroll to zoom, or use the component list below.')+'''</p></div><label class="component-label">'''+('选择组件'if lang=='zh'else'Select component')+''' <select class="component-select"><option value="">—</option></select></label></div>'''
 if k=='pipeline':return '<ol class="pipeline">'+''.join('<li><span>'+str(i+1)+'</span>'+v+'</li>'for i,v in enumerate(b[lang]))+'</ol>'
 raise ValueError(k)
def build():
 for lang,t in I18N.items():
  out=ROOT/'web'/lang;out.mkdir(parents=True,exist_ok=True)
  search=[]
  for idx,p in enumerate(PAGES):
   title=p['title'][lang];other='en'if lang=='zh'else'zh'
   nav='';group=None
   for n in PAGES:
    if n['group'][lang]!=group:group=n['group'][lang];nav+='<p class="nav-group">'+group+'</p>'
    nav+='<a '+('aria-current="page" 'if n['id']==p['id']else'')+'href="'+n['id']+'.html"><span class="nav-number">'+n.get('number','')+'</span>'+n['title'][lang]+'</a>'
   toc=''.join('<a href="#'+s['id']+'">'+s['title'][lang]+'</a>'for s in p['sections'])
   content=''.join('<section id="'+s['id']+'"><h2>'+s['title'][lang]+'</h2>'+''.join(render_block(b,lang,t)for b in s['blocks'])+'</section>'for s in p['sections'])
   hero=''
   if p['id']=='index':hero='<figure class="hero-image"><img fetchpriority="high" src="../assets/robot/assembled.png" alt="Microduck robot"></figure>'
   bottom=''
   if idx:bottom+='<a href="'+PAGES[idx-1]['id']+'.html"><small>← '+t['prev']+'</small>'+PAGES[idx-1]['title'][lang]+'</a>'
   if idx+1<len(PAGES):bottom+='<a href="'+PAGES[idx+1]['id']+'.html"><small>'+t['next']+' →</small>'+PAGES[idx+1]['title'][lang]+'</a>'
   refs=''
   if p.get('references'):refs='<section id="references"><h2>'+t['refs']+'</h2><ul class="references">'+''.join('<li><a href="'+e(r['url'])+'" target="_blank" rel="noopener">'+r[lang]+' ↗</a></li>'for r in p['references'])+'</ul></section>'
   html=f'''<!doctype html>
<html lang="{t['lang']}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{e(p['summary'][lang])}"><meta name="theme-color" content="#76b900"><title>{e(title)} · Microduck Workshop</title><link rel="icon" href="../assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="../assets/course.css"><script type="importmap">{{"imports":{{"three":"../assets/vendor/three.module.js","three/addons/":"../assets/vendor/addons/"}}}}</script><script defer src="../assets/course.js"></script></head>
<body data-lang="{lang}" data-page="{p['id']}"><a class="skip" href="#main">{t['skip']}</a>
<header><button class="menu-toggle" aria-label="{t['menu']}" aria-expanded="false">☰</button><a class="brand" href="index.html"><span class="brand-mark">NVIDIA</span><span class="brand-course">Microduck <b>Workshop</b></span></a><div class="header-actions"><button class="search-open">{t['search']} <kbd>/</kbd></button><a class="language" hreflang="{other}" href="../{other}/{p['id']}.html">{t['swap']}</a><a class="repo-link" href="https://github.com/NV-DLI/Microduck-Workshop">GitHub ↗</a></div></header>
<div class="layout"><nav class="sidebar" aria-label="{t['menu']}"><div class="sidebar-intro">{t['course']}</div>{nav}<div class="local-progress"><span></span><progress max="7" value="0"></progress></div></nav>
<main id="main"><div class="breadcrumb">Microduck Workshop <span>/</span> {p['group'][lang]}</div><div class="lesson-heading {'is-home'if p['id']=='index'else''}"><div><p class="eyebrow">{p.get('eyebrow',{}).get(lang,'SELF-PACED WORKSHOP')}</p><h1>{title}</h1><p class="lead">{p['summary'][lang]}</p><div class="lesson-meta">{p['meta'][lang]}</div></div>{hero}</div>{content}{refs}
{('<button class="mark-complete" data-mark="'+t['mark']+'" data-marked="'+t['marked']+'">'+t['mark']+'</button>')if p.get('number')else''}<nav class="pager" aria-label="Pagination">{bottom}</nav><footer>Microduck Workshop · <a href="references.html">{t['refs']}</a><span>© 2026 · <a href="https://github.com/NV-DLI/Microduck-Workshop/blob/main/THIRD_PARTY_NOTICES.md">Attribution & licenses</a></span></footer></main>
<aside class="toc"><p>{t['onpage']}</p>{toc}{'<a href="#references">'+t['refs']+'</a>'if refs else''}</aside></div>
<dialog class="search-dialog"><form method="dialog"><label for="search-input">{t['search']}</label><button aria-label="Close">×</button></form><input id="search-input" type="search" autocomplete="off" placeholder="{t['search']}…"><div class="search-results"></div></dialog>
</body></html>'''
   (out/(p['id']+'.html')).write_text(html)
   search.append(dict(id=p['id'],title=title,text=re.sub('<[^>]+>',' ',content)))
  (out/'search.json').write_text(json.dumps(search,ensure_ascii=False))
 (ROOT/'web/index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=zh/index.html"><title>Microduck Workshop</title></head><body><a href="zh/index.html">中文课程</a> · <a href="en/index.html">English course</a></body></html>')
 (ROOT/'web/.nojekyll').touch()
 print(f'Built {len(PAGES)*2} lesson pages')
if __name__=='__main__':build()
