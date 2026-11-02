// "Inside the little duck" (web/assets/robot-explorer.js) loads by itself as soon as the page is
// ready. Until the 3D duck is ready the transparent stage shows only a small progress bar:
// "preparing the 3D view" while the code starts, the real download percentage once the model's
// size is known (no estimate otherwise), then "preparing the model" while it is decoded and set
// up. Screen readers hear "loading" once; the progress bar exposes its value without announcing
// every step. A failed load hides the bar and shows a short message and one "try again" button,
// never a still image. Automatic loading never moves focus or scrolls; after a retry the
// learner asked for, focus stays inside the figure.
// The page names the exact module version (data-dx-module), so a module cached from an earlier
// version of the page is never paired with this page's markup and stylesheet.
(() => {
  for (const root of document.querySelectorAll("[data-duck-explorer]")) {
    const button = root.querySelector(".dx-open"),
      launch = root.querySelector(".dx-launch"),
      status = root.querySelector(".dx-status"),
      progress = root.querySelector(".dx-progress"),
      bar = progress?.querySelector(".dx-progress-track"),
      label = progress?.querySelector(".dx-progress-label");
    let busy = false, attempt = 0, generation = 0, shown = null;
    // phase: "preparing" | "downloading" | "finishing"; percent only while downloading.
    const show = (phase, percent) => {
      if (!progress || shown === `${phase}${percent}`) return;
      shown = `${phase}${percent}`;
      const name = progress.dataset[phase], known = phase === "downloading";
      progress.hidden = false;
      progress.dataset.phase = phase;
      label.textContent = known ? `${name} ${percent}%` : `${name}…`;
      bar.setAttribute("aria-label", name);
      for (const [key, value] of [["aria-valuemin", 0], ["aria-valuemax", 100], ["aria-valuenow", percent]])
        known ? bar.setAttribute(key, value) : bar.removeAttribute(key);
      bar.style.setProperty("--dx-progress", known ? `${percent}%` : "");
    };
    const load = async (requested) => {
      if (busy || root.dataset.state === "ready") return; // one attempt at a time, never twice
      busy = true;
      const keepFocus = requested && root.contains(document.activeElement);
      root.dataset.state = "loading";
      launch.hidden = true;
      status.textContent = status.dataset.loading;
      shown = null;
      show("preparing");
      // The model keeps downloading even if a data file fails first, so its progress reports may
      // arrive after this attempt ended: only the current, still-loading attempt updates the bar.
      const current = ++generation;
      const onProgress = (fraction) => {
        if (current !== generation || root.dataset.state !== "loading" || !Number.isFinite(fraction)) return;
        fraction = Math.min(1, Math.max(0, fraction));
        fraction >= 1 ? show("finishing") : show("downloading", Math.min(99, Math.floor(fraction * 100)));
      };
      try {
        const src = new URL(root.dataset.dxModule || "../assets/robot-explorer.js", document.baseURI);
        // A failed module fetch stays failed for that URL, so a retry asks for a fresh copy.
        if (attempt++) src.searchParams.set("attempt", attempt);
        const { mountExplorer } = await import(src.href);
        await mountExplorer(root, document.body.dataset.lang, onProgress);
        root.dataset.state = "ready";
        launch.remove();
        progress?.remove();
        status.textContent = "";
        if (keepFocus) root.querySelector(".dx-chapter")?.focus({ preventScroll: true });
      } catch (error) {
        root.dataset.state = "error";
        if (progress) progress.hidden = true;
        status.textContent = status.dataset.error;
        launch.hidden = false;
        if (keepFocus) button.focus({ preventScroll: true });
        console.error(error);
      } finally {
        generation++; // this attempt's progress reports are ignored from now on
        busy = false;
      }
    };
    button.addEventListener("click", () => load(true));
    load(false);
  }
})();
