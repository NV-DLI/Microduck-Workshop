#!/usr/bin/env python3
"""Check published links, bilingual lessons, lab commands, assets and immutable weights."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import ast,hashlib,json,re,subprocess
R=Path(__file__).resolve().parents[1];W=R/'web';errors=[];count=0
class Page(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id'in a:self.ids.append(a['id'])
  for key in ('href','src','data-gif'):
   if a.get(key):self.links.append(a[key])
parsed={}
for f in W.rglob('*.html'):
 p=Page();p.feed(f.read_text());parsed[f.resolve()]=p
 if len(p.ids)!=len(set(p.ids)):errors.append(f'Duplicate IDs: {f}')
 for pattern in ['/home/yuki/','@COURSE@','@REPO@','localhost:8018']:
  if pattern in f.read_text():errors.append(f'Internal path in page: {f}')
for f,p in parsed.items():
 for link in p.links:
  u=urlsplit(link)
  if u.scheme or u.netloc:continue
  target=(f.parent/unquote(u.path)).resolve()if u.path else f
  if not target.is_relative_to(W):errors.append(f'Outside web root: {f.name} {link}')
  elif not target.exists():errors.append(f'Missing: {f.name} {link}')
  elif u.fragment and target in parsed and unquote(u.fragment)not in parsed[target].ids:errors.append(f'Missing fragment: {f.name} {link}')
  count+=1
pages=json.loads((R/'content/course.json').read_text());chapters=[p for p in pages if p.get('number')]
assert len(chapters)==7
for p in pages:
 for lang in ['zh','en']:
  if not (W/lang/(p['id']+'.html')).exists():errors.append('Missing language page '+p['id'])
 if p.get('number')and not any(b['type']=='media'for s in p['sections']for b in s['blocks']):errors.append('Missing demonstration '+p['id'])
 for s in p['sections']:
  for b in s['blocks']:
   if b['type']=='code'and b.get('label','').startswith('Terminal'):
    result=subprocess.run(['bash','-n'],input=b['code'],capture_output=True,text=True)
    if result.returncode:errors.append('Invalid shell block '+p['id'])
    for script in re.findall(r'python3 (scripts/[\w.-]+\.py)',b['code']):
     if not (R/script).is_file():errors.append('Missing command '+script)
for item in json.loads((R/'models/manifest.json').read_text())['files']:
 f=R/'models'/item['file']
 if hashlib.sha256(f.read_bytes()).hexdigest()!=item['sha256']:errors.append('Model mismatch '+f.name)
for folder in ['scripts','lab']:
 for f in (R/folder).rglob('*.py'):
  try:ast.parse(f.read_text(),filename=str(f))
  except SyntaxError:errors.append('Python syntax '+str(f))
for item in json.loads((R/'lab/bundle-manifest.json').read_text())['files']:
 f=R/'lab/frozen'/item['path']
 if hashlib.sha256(f.read_bytes()).hexdigest()!=item['template_sha256']:errors.append('Bundle mismatch '+str(f))
meta=json.loads((W/'assets/robot/components.json').read_text())
assert len(meta['parts'])==70
ids=[p['id']for p in meta['parts']]
assert len(set(ids))==70
if errors:raise SystemExit('\n'.join(errors))
print(json.dumps(dict(pages=len(parsed),local_links=count,chapters=7,models=6,components=70,passed=True)))
