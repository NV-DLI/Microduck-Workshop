#!/usr/bin/env python3
"""Build a local-first, bilingual static course with no server-side dependencies."""
from pathlib import Path
from html import escape as e
import json,re
from teaching_blocks import render_teaching_block
ROOT=Path(__file__).resolve().parents[1]
PAGES=json.loads((ROOT/'content/course.json').read_text())
DEPTH=json.loads((ROOT/'content/zh-depth.json').read_text()) if (ROOT/'content/zh-depth.json').exists() else {}
I18N={
'zh':dict(lang='zh-CN',course='Microduck 机器人学习实战',search='搜索课程',menu='目录',onpage='本页内容',prev='上一页',next='下一页',copy='复制命令',copied='已复制',learn='开始学习',refs='参考资料',check='检查理解',answer='查看解析',reset='重置',swap='English',overview='课程概览',skip='跳至正文'),
'en':dict(lang='en',course='Robot Learning with Microduck',search='Search the course',menu='Contents',onpage='On this page',prev='Previous',next='Next',copy='Copy command',copied='Copied',learn='Start learning',refs='References',check='Check your understanding',answer='Show explanation',reset='Reset',swap='中文',overview='Course overview',skip='Skip to content')}
def render_block(b,lang,t):
 k=b['type'];text=b.get(lang,b.get('text',''))
 if b.get('languages') and lang not in b['languages']:return ''
 extra=render_teaching_block(b,lang)
 if extra is not None:return extra
 if k=='html':return text
 if k=='p':return '<p>'+text+'</p>'
 if k=='note':return '<aside class="note"><strong>'+b['title'][lang]+'</strong><p>'+text+'</p></aside>'
 if k=='code':
  terminal=b.get('label','Terminal · Bash').startswith('Terminal')
  control=('<button type="button" class="copy" data-done="'+t['copied']+'">'+t['copy']+'</button>') if terminal else ('<a class="source-link" href="'+e(b['source'])+'" target="_blank" rel="noopener">'+('完整代码 ↗' if lang=='zh' else 'Full source ↗')+'</a>' if b.get('source') else '')
  return '<div class="code-block '+('terminal' if terminal else 'source-excerpt')+'"><div class="code-toolbar"><span>'+e(b.get('label','Terminal · Bash'))+'</span>'+control+'</div><pre><code class="language-'+('bash' if terminal else 'python')+'">'+e(b['code'])+'</code></pre></div>'
 if k=='list':return '<ul>'+''.join('<li>'+x+'</li>' for x in b[lang])+'</ul>'
 if k=='table':return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>'for x in b['headers'][lang])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>'for x in row)+'</tr>'for row in b['rows'][lang])+'</tbody></table></div>'
 if k=='quiz':return '<div class="quiz"><h3>'+t['check']+'</h3><p>'+b['question'][lang]+'</p><details><summary>'+t['answer']+'</summary><p>'+b['answer'][lang]+'</p></details></div>'
 if k=='anatomy':return '<div class="anatomy" data-robot-viewer><div class="viewer-stage"><img class="robot-fallback" src="../assets/robot/studio-hd.webp" alt="Microduck"><div class="viewer-status" role="status"></div><button type="button" class="load-robot">'+('加载交互模型' if lang=='zh' else 'Open interactive model')+'</button></div><div class="viewer-toolbar"><label>'+('装配'if lang=='zh'else'Assembled')+' <input type="range" min="0" max="100" value="0" class="explode" aria-label="'+('结构展开程度'if lang=='zh'else'Explosion amount')+'"> '+('展开'if lang=='zh'else'Exploded')+'</label><button type="button" class="view-reset">'+('返回总览'if lang=='zh'else'Reset view')+'</button></div><div class="component-info" aria-live="polite"><strong>'+('点击部件或名称，即可放大查看'if lang=='zh'else'Click a part or its label to inspect it')+'</strong><p>'+('左键拖动旋转，按住滚轮拖动平移，滚轮或双指缩放；移动滑块可展开结构。'if lang=='zh'else'Left-drag to orbit, middle-drag to pan, scroll or pinch to zoom. Use the slider to separate the parts.')+'</p></div></div>'
 if k=='pipeline':return '<ol class="pipeline">'+''.join('<li><span>'+str(i+1)+'</span>'+v+'</li>'for i,v in enumerate(b[lang]))+'</ol>'
 raise ValueError(k)
def build():
 for lang,t in I18N.items():
  out=ROOT/'web'/lang;out.mkdir(parents=True,exist_ok=True)
  search=[]
  for idx,p in enumerate(PAGES):
   title=p['title'][lang];other='en'if lang=='zh'else'zh'
   sections=list(p['sections'])
   if lang=='zh':
    for addition in DEPTH.get(p['id'],[]):
     if addition.get('replace'):
      pos=next(i for i,x in enumerate(sections) if x['id']==addition['replace']);sections[pos]=addition;continue
     before=addition.get('before');pos=next((i for i,x in enumerate(sections) if x['id']==before),len(sections))
     sections.insert(pos,addition)
   nav='';group=None
   for n in PAGES:
    if n['group'][lang]!=group:group=n['group'][lang];nav+='<p class="nav-group">'+group+'</p>'
    nav+='<a '+('aria-current="page" 'if n['id']==p['id']else'')+'href="'+n['id']+'.html"><span class="nav-number">'+n.get('number','')+'</span>'+n['title'][lang]+'</a>'
   toc=''.join('<a href="#'+s['id']+'">'+s['title'][lang]+'</a>'for s in sections)
   content=''.join('<section id="'+s['id']+'"><h2>'+s['title'][lang]+'</h2>'+''.join(render_block(b,lang,t)for b in s['blocks'])+'</section>'for s in sections)
   hero=''
   if p['id']=='index':hero='<figure class="hero-image"><img fetchpriority="high" src="../assets/robot/studio-hd.webp" alt="'+('Microduck 机器人' if lang=='zh' else 'Microduck robot')+'"></figure>'
   bottom=''
   if idx:bottom+='<a href="'+PAGES[idx-1]['id']+'.html"><small>← '+t['prev']+'</small>'+PAGES[idx-1]['title'][lang]+'</a>'
   if idx+1<len(PAGES):bottom+='<a href="'+PAGES[idx+1]['id']+'.html"><small>'+t['next']+' →</small>'+PAGES[idx+1]['title'][lang]+'</a>'
   refs=''
   if p.get('references'):refs='<section id="references"><h2>'+t['refs']+'</h2><ul class="references">'+''.join('<li><a href="'+e(r['url'])+'" target="_blank" rel="noopener">'+r[lang]+' ↗</a></li>'for r in p['references'])+'</ul></section>'
   html=f'''<!doctype html>
<html lang="{t['lang']}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{e(p['summary'][lang])}"><meta name="theme-color" content="#76b900"><title>{e(title)} · Microduck Workshop</title><link rel="icon" href="../assets/favicon.svg" type="image/svg+xml"><link rel="preload" href="../assets/fonts/OpenSans.woff2" as="font" type="font/woff2" crossorigin><link rel="stylesheet" href="../assets/vendor/highlight/highlight-github-dark-11.10.0.min.css"><link rel="stylesheet" href="../assets/course.css"><script type="importmap">{{"imports":{{"three":"../assets/vendor/three.module.js","three/addons/":"../assets/vendor/addons/"}}}}</script><link rel="stylesheet" href="../assets/vendor/katex/katex.min.css"><script defer src="../assets/vendor/katex/katex.min.js"></script><script defer src="../assets/vendor/katex/auto-render.min.js"></script><script defer src="../assets/vendor/highlight/highlight-11.10.0.min.js"></script><script defer src="../assets/course.js"></script><script defer src="../assets/teaching.js"></script></head>
<body data-lang="{lang}" data-page="{p['id']}"><a class="skip" href="#main">{t['skip']}</a>
<header><button class="menu-toggle" aria-label="{t['menu']}" aria-expanded="false">☰</button><a class="brand" href="index.html"><span class="brand-course">Microduck <b>Workshop</b></span></a><div class="header-actions"><button class="search-open">{t['search']} <kbd>/</kbd></button><details class="language-menu"><summary aria-label="{'选择语言' if lang=='zh' else 'Choose language'}"><svg viewBox="0 0 24 24" width="23" height="23" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18M5 7h14M5 17h14"/></svg></summary><div class="language-options"><a lang="zh-CN" href="../zh/{p['id']}.html" {'aria-current="true"' if lang=='zh' else ''}>简体中文</a><a lang="en" href="../en/{p['id']}.html" {'aria-current="true"' if lang=='en' else ''}>English</a></div></details><a class="repo-link" href="https://github.com/NV-DLI/Microduck-Workshop">GitHub ↗</a></div></header>
<div class="layout"><nav class="sidebar" aria-label="{t['menu']}"><div class="sidebar-intro">{t['course']}</div>{nav}</nav>
<main id="main"><div class="breadcrumb">Microduck Workshop <span>/</span> {p['group'][lang]}</div><div class="lesson-heading {'is-home'if p['id']=='index'else''}"><div><p class="eyebrow">{p.get('eyebrow',{}).get(lang,'SELF-PACED WORKSHOP')}</p><h1>{title}</h1><p class="lead">{p['summary'][lang]}</p><div class="lesson-meta">{p['meta'][lang]}</div></div>{hero}</div>{content}{refs}
<nav class="pager" aria-label="Pagination">{bottom}</nav><footer>Microduck Workshop · <a href="references.html">{t['refs']}</a><span>© 2026 · <a href="https://github.com/NV-DLI/Microduck-Workshop/blob/main/THIRD_PARTY_NOTICES.md">{'来源与许可' if lang=='zh' else 'Attribution & licenses'}</a></span></footer></main>
<aside class="toc"><p>{t['onpage']}</p>{toc}{'<a href="#references">'+t['refs']+'</a>'if refs else''}</aside></div>
<dialog class="search-dialog"><form method="dialog"><label for="search-input">{t['search']}</label><button aria-label="{'关闭' if lang=='zh' else 'Close'}">×</button></form><input id="search-input" type="search" autocomplete="off" placeholder="{t['search']}…"><div class="search-results"></div></dialog>
</body></html>'''
   (out/(p['id']+'.html')).write_text(html)
   search.append(dict(id=p['id'],title=title,text=re.sub('<[^>]+>',' ',content)))
  (out/'search.json').write_text(json.dumps(search,ensure_ascii=False))
 (ROOT/'web/index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=zh/index.html"><title>Microduck Workshop</title></head><body><a href="zh/index.html">中文课程</a> · <a href="en/index.html">English course</a></body></html>')
 for old,new in json.loads((ROOT/'content/redirects.json').read_text()).items():
  for lang in I18N:
   (ROOT/'web'/lang/(old+'.html')).write_text('<!doctype html><html lang="'+lang+'"><head><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+new+'.html"><title>Microduck Workshop</title></head><body><a href="'+new+'.html">'+('查看课程章节' if lang=='zh' else 'Open lesson')+'</a></body></html>')
 (ROOT/'web/.nojekyll').touch()
 print(f'Built {len(PAGES)*2} lesson pages')
if __name__=='__main__':build()
