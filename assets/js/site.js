(() => {
  "use strict";

  const root = document.documentElement;
  const header = document.querySelector("[data-site-header]");
  const menuToggle = document.getElementById("menu-toggle");
  const themeToggle = document.getElementById("theme-toggle");
  const themeColor = document.querySelector('meta[name="theme-color"]');

  function updateThemeColor() {
    themeColor?.setAttribute("content", root.dataset.theme === "dark" ? "#091116" : "#F4F9FC");
  }

  themeToggle?.addEventListener("click", () => {
    const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = nextTheme;
    localStorage.setItem("pref-theme", nextTheme);
    updateThemeColor();
  });

  updateThemeColor();

  let previousScrollY = window.scrollY;
  function updateHeader() {
    const currentScrollY = window.scrollY;
    header?.classList.toggle("not-top", currentScrollY > 20);
    if (header) {
      header.dataset.show = String(currentScrollY < 350 || currentScrollY < previousScrollY || header.classList.contains("expanded"));
    }
    previousScrollY = currentScrollY;
  }

  function setMenu(open) {
    header?.classList.toggle("expanded", open);
    menuToggle?.setAttribute("aria-expanded", String(open));
    updateHeader();
  }

  menuToggle?.addEventListener("click", () => {
    setMenu(!header?.classList.contains("expanded"));
  });

  header?.querySelectorAll("#menu a").forEach((link) => {
    link.addEventListener("click", () => setMenu(false));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") setMenu(false);
  });

  const topLink = document.getElementById("top-link");
  function updateScrollState() {
    updateHeader();
    topLink?.classList.toggle("is-visible", window.scrollY > 720);
  }
  window.addEventListener("scroll", updateScrollState, { passive: true });
  updateScrollState();

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
