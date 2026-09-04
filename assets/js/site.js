(() => {
  "use strict";

  const root = document.documentElement;
  const themeToggle = document.getElementById("theme-toggle");
  themeToggle?.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = nextTheme;
    localStorage.setItem("pref-theme", nextTheme);
  });

  const topLink = document.getElementById("top-link");
  function updateTopLink() {
    topLink?.classList.toggle("is-visible", window.scrollY > 720);
  }
  window.addEventListener("scroll", updateTopLink, { passive: true });
  updateTopLink();

  document.addEventListener("click", (event) => {
    const anchor = event.target.closest('a[href^="#"]');
    if (!anchor) return;
    const hash = anchor.getAttribute("href");
    const target = hash === "#top" ? document.documentElement : document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    if (hash === "#top") history.replaceState(null, "", `${location.pathname}${location.search}`);
    else history.pushState(null, "", hash);
  });
})();
