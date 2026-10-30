// Light/dark switch in the header (and its copy in the narrow-screen drawer).
// A click presses the theme's own hidden light/dark button, so the theme stores
// the explicit choice. The course defaults to dark until a visitor chooses.
(() => {
  const root = document.documentElement;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
  const current = () => {
    const theme = root.dataset.theme;
    if (theme === "dark" || theme === "light") return theme;
    return prefersDark.matches ? "dark" : "light";
  };
  const sync = () => {
    const dark = String(current() === "dark");
    document.querySelectorAll(".course-theme-switch").forEach((el) => el.setAttribute("aria-checked", dark));
  };
  function toggle(event) {
    const target = current() === "dark" ? "light" : "dark";
    event.currentTarget.parentElement.querySelector(`.theme-change-button[data-mode="${target}"]`).click();
    sync();
  }
  // The theme binds its hidden buttons when its script runs; clicks come later.
  const init = () => {
    document.querySelectorAll(".course-theme-switch").forEach((el) => el.addEventListener("click", toggle));
    new MutationObserver(sync).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    sync();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
