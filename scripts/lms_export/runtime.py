"""Inline runtime that makes flat assets load under every Open edX static route.

An iframe page can be served as ``/static/p.html`` (preview), ``/c4x/ORG/COURSE/asset/p.html``,
``/asset-v1:ORG+COURSE+RUN+type@asset+block/p.html`` or
``/asset-v1:ORG+COURSE+RUN+type@asset+block@p.html``. In the last form a relative URL such as
``image.webp`` resolves to the site root, so exported pages never rely on relative URL resolution.
The bootstrap derives the asset base from the page's own address (never from build-time course
metadata) and every exported reference goes through ``__lmsAsset(name)``:

* head/body ``<script src>`` and ``<link href>`` are written by ``__lms.write`` in document order;
* an import map is written for module graphs (``@lms/<file>`` and the site's bare specifiers);
* element URL attributes are stored in ``data-lms-url`` and set by ``__lms.fix()`` before
  deferred scripts run;
* JavaScript string literals naming assets become ``__lmsAsset("file")`` expressions.
"""

from __future__ import annotations

import json

BOOTSTRAP_JS = r"""(function(w,d,C){
var p=location.pathname,m=/^(.*?)\/(asset-v1:[^\/]*\+type@asset\+block)@[^\/]*$/.exec(p),b;
if(m){b=location.origin+m[1].replace(/\/assets\/courseware\/v\d+\/[^\/]+$/,"")+"/"+m[2]+"@";}
else{b=new URL("./",location.href).href;}
function a(n){n=String(n);return /^(?:[a-zA-Z][a-zA-Z0-9+.\-]*:|\/|#)/.test(n)?n:b+n;}
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/</g,"&lt;");}
function write(tag,attrs,urls){var s="<"+tag;for(var i=0;i<attrs.length;i++){var k=attrs[i][0],v=attrs[i][1];if(urls.indexOf(k)>=0)v=a(v);s+=" "+k+(v===null?"":'="'+esc(v)+'"');}s+=">";if(tag==="script")s+="<\/script>";d.write(s);}
function fix(root){var els=(root||d).querySelectorAll("[data-lms-url]"),media=[];
for(var i=0;i<els.length;i++){var el=els[i],map;try{map=JSON.parse(el.getAttribute("data-lms-url"));}catch(e){continue;}
el.removeAttribute("data-lms-url");
for(var k in map){var v=map[k];
if(k==="srcset"||k==="imagesrcset"){v=v.split(",").map(function(x){x=x.trim();var j=x.search(/\s/);return j<0?a(x):a(x.slice(0,j))+x.slice(j);}).join(", ");}else{v=a(v);}
if(k==="xlink:href"){el.setAttributeNS("http://www.w3.org/1999/xlink",k,v);}else{el.setAttribute(k,v);}
var t=el.tagName.toLowerCase(),par=el.parentNode;
if((t==="source"||t==="track")&&par&&typeof par.load==="function"&&media.indexOf(par)<0){media.push(par);}}}
for(var j=0;j<media.length;j++){media[j].load();}}
w.__lmsAsset=a;
w.__lms={asset:a,base:b,route:m?"asset-key":"directory",write:write,fix:fix};
var L=C.lang;
if(L&&L.remember){var K="lms-export-lang:"+L.key;
try{var want=w.localStorage.getItem(K);if(L.isDefault&&want&&want!==L.cur&&L.pages[want]){location.replace(a(L.pages[want])+location.hash);return;}}catch(e){}
d.addEventListener("click",function(ev){var t=ev.target&&ev.target.closest&&ev.target.closest("[data-lms-lang]");if(t){try{w.localStorage.setItem(K,t.getAttribute("data-lms-lang"));}catch(e){}}});}
if(C.imports){var o={},n=0;for(var k in C.imports){o[k]=a(C.imports[k]);n++;}
if(n){d.write('<script type="importmap">'+JSON.stringify({imports:o}).replace(/</g,"\\u003c")+"<\/script>");}}
d.addEventListener("DOMContentLoaded",function(){fix();});
})(window,document,__CONFIG__);"""


def json_for_script(value: object) -> str:
    """JSON safe to embed inside an inline <script>."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def bootstrap(config: dict) -> str:
    return BOOTSTRAP_JS.replace("__CONFIG__", json_for_script(config))


def write_call(tag: str, attrs: list[tuple[str, str | None]], url_attrs: list[str]) -> str:
    return "__lms.write(%s,%s,%s)" % (
        json_for_script(tag),
        json_for_script([[k, v] for k, v in attrs]),
        json_for_script(url_attrs),
    )


FIX_CALL = "__lms.fix()"
