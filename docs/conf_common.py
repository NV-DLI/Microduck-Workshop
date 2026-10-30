"""Sphinx settings shared by the Chinese and English course builds.

Each language has its own source directory (``docs/zh``, ``docs/en``) whose
``conf.py`` calls :func:`configure`. Pages are written to ``web/<lang>/`` by
``scripts/build_site.py``; large media, the 3D robot, fonts and vendor
scripts stay in ``web/assets`` and are referenced as ``../assets/...``.
"""

import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_ext"))

LANGS = {
    "zh": dict(
        language="zh_CN",
        project="使用 NVIDIA Isaac 仿真并训练 Microduck",
        language_menu="选择语言",
        dark_mode="深色模式",
    ),
    "en": dict(
        language="en",
        project="Simulate and train Microduck with NVIDIA Isaac",
        language_menu="Choose language",
        dark_mode="Dark mode",
    ),
}


def configure(namespace: dict, lang: str) -> None:
    ui = LANGS[lang]
    settings = dict(
        project=ui["project"],
        author="NVIDIA",
        copyright="2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved",
        language=ui["language"],
        # Header title and browser-tab suffix: the official course name in each language.
        html_title=ui["project"],
        extensions=["myst_parser", "sphinx_copybutton", "microduck_course"],
        myst_enable_extensions=["colon_fence", "attrs_block", "attrs_inline", "dollarmath", "html_image"],
        myst_links_external_new_tab=True,
        myst_heading_anchors=0,
        # Keep authored punctuation exactly; Chinese and English quotes are deliberate.
        smartquotes=False,
        exclude_patterns=["_build"],
        templates_path=[str(DOCS / "_templates")],
        html_static_path=[str(DOCS / "_static")],
        html_theme="nvidia_sphinx_theme",
        html_copy_source=False,
        html_show_sourcelink=False,
        html_last_updated_fmt=None,
        html_math_renderer="course_katex",
        html_css_files=["microduck.css", "course-theme-toggle.css"],
        html_js_files=[("course-widgets.js", {"defer": "defer"}), ("course-theme-toggle.js", {"defer": "defer"})],
        # Page shell: the theme's own header, sidebars and colors. Course additions:
        # a language menu, a two-state light/dark switch driving the theme's mode
        # buttons and a GitHub link. The footer is the theme's own, with the links
        # of the NVIDIA docs site (docs.nvidia.com sim-to-real SO-101 course).
        html_theme_options={
            "navbar_end": ["course-language", "course-theme-toggle", "navbar-icon-links"],
            "icon_links": [
                {"name": "GitHub", "url": "https://github.com/NV-DLI/Microduck-Workshop", "icon": "fa-brands fa-github", "type": "fontawesome"}
            ],
            # The theme's default article header holds an NVIDIA-docs survey link.
            "article_header_end": [],
            "footer_links": [
                {"name": "Privacy Policy", "url": "https://www.nvidia.com/en-us/about-nvidia/privacy-policy/"},
                {"name": "Manage My Privacy", "url": "https://www.nvidia.com/en-us/about-nvidia/privacy-center/"},
                {"name": "Do Not Sell or Share My Data", "url": "https://www.nvidia.com/en-us/preferences/start/"},
                {"name": "Terms of Service", "url": "https://www.nvidia.com/en-us/about-nvidia/terms-of-service/"},
                {"name": "Accessibility", "url": "https://www.nvidia.com/en-us/about-nvidia/accessibility/"},
                {"name": "Corporate Policies", "url": "https://www.nvidia.com/en-us/about-nvidia/company-policies/"},
                {"name": "Product Security", "url": "https://www.nvidia.com/en-us/product-security/"},
                {"name": "Contact", "url": "https://www.nvidia.com/en-us/contact/"},
            ],
        },
        html_context={"course_ui_strings": ui, "default_mode": "dark"},
        copybutton_selector="div.terminal div.highlight pre",
        copybutton_exclude=".linenos, .gp, .go",
        course_lang=lang,
        course_assets_dir=str(ROOT / "web/assets"),
        course_assets_url="../assets",
        course_redirects=json.loads((DOCS / "_data/redirects.json").read_text()),
    )
    namespace.update(settings)
