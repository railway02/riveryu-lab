(() => {
  "use strict";

  const timers = new WeakMap();

  async function copyCode(button) {
    const code = button.closest("[data-code-block]")?.querySelector("code");
    if (!code) return;

    try {
      await navigator.clipboard.writeText(code.textContent || "");
    } catch {
      return;
    }

    button.classList.add("copied");
    button.setAttribute("aria-label", "Copied");
    window.clearTimeout(timers.get(button));
    timers.set(button, window.setTimeout(() => {
      button.classList.remove("copied");
      button.setAttribute("aria-label", "Copy code");
    }, 2000));
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-code-copy]");
    if (button) copyCode(button);
  });
})();
