// "Inside the little duck" (web/assets/robot-explorer.js) loads by itself as soon as the page is
// ready. Until the 3D duck is ready the stage stays transparent and shows nothing: loading is
// announced to screen readers only. A failed load shows a short message and one "try again"
// button, never a still image. Automatic loading never moves focus or scrolls; after a retry the
// learner asked for, focus stays inside the figure.
// The page names the exact module version (data-dx-module), so a module cached from an earlier
// version of the page is never paired with this page's markup and stylesheet.
(() => {
  for (const root of document.querySelectorAll("[data-duck-explorer]")) {
    const button = root.querySelector(".dx-open"),
      launch = root.querySelector(".dx-launch"),
      status = root.querySelector(".dx-status");
    let busy = false, attempt = 0;
    const load = async (requested) => {
      if (busy || root.dataset.state === "ready") return; // one attempt at a time, never twice
      busy = true;
      const keepFocus = requested && root.contains(document.activeElement);
      root.dataset.state = "loading";
      launch.hidden = true;
      status.textContent = status.dataset.loading;
      try {
        const src = new URL(root.dataset.dxModule || "../assets/robot-explorer.js", document.baseURI);
        // A failed module fetch stays failed for that URL, so a retry asks for a fresh copy.
        if (attempt++) src.searchParams.set("attempt", attempt);
        const { mountExplorer } = await import(src.href);
        await mountExplorer(root, document.body.dataset.lang);
        root.dataset.state = "ready";
        launch.remove();
        status.textContent = "";
        if (keepFocus) root.querySelector(".dx-chapter")?.focus({ preventScroll: true });
      } catch (error) {
        root.dataset.state = "error";
        status.textContent = status.dataset.error;
        launch.hidden = false;
        if (keepFocus) button.focus({ preventScroll: true });
        console.error(error);
      } finally {
        busy = false;
      }
    };
    button.addEventListener("click", () => load(true));
    load(false);
  }
})();
