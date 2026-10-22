---
name: microduck-course
description: Write, revise, and validate the bilingual Microduck self-study website, terminal labs, and fixed-model download instructions in this repository.
---
# Student course workflow

Write to the learner: explain why an operation matters, what to do, what to observe, and how to recover from a likely failure. Keep development chronology and internal acceptance machinery outside student pages. Chinese and English share lesson identifiers, commands, and artifact references.

Use short continuous lessons, a persistent chapter navigation, and local references. Each practical lesson names its prerequisites, parameters, expected outputs, and a question that checks understanding. Copy blocks contain executable commands without terminal prompts; explain parameters beside the command. Never offer a command whose entry point is absent.

Fixed models in `models/` are the downstream default. Training is a separate practice route with a new output directory, seed, budget, and explicit custom-checkpoint opt-in. Never retrain a preserved model to produce course media. Distinguish detection scores, survival rates, local kicks and complete autonomous task success.

Blender component assets retain CAD provenance and licensing. Check assembly and explosion renders from the same camera; validate mesh identities, transforms and browser selection. Do not claim electronic components from one hardware revision represent another revision.

Build with `python3 scripts/build_site.py`. Run `python3 scripts/check_course.py` for links, command references, bilingual topology and fixed-model identities. Inspect desktop and mobile pages, copy controls, quiz feedback, local search, language links, media loading, and the actual 3D viewer in a browser before publication. Check the deployed revision after publishing; a local build is not evidence of remote publication.
