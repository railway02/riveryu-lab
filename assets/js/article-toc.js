(() => {
  "use strict";

  const toc = document.getElementById("article-toc");
  const content = document.getElementById("content");
  if (!toc || !content) return;

  const links = Array.from(toc.querySelectorAll('a[href^="#"]')).map((link) => {
    const slug = decodeURIComponent(link.hash.slice(1));
    const heading = document.getElementById(slug);
    const progress = document.createElement("span");
    progress.className = "toc-progress";
    link.parentElement?.classList.add("toc-entry");
    link.parentElement?.insertBefore(progress, link);
    link.classList.add("toc-item");
    return { link, heading, progress };
  }).filter(({ heading }) => heading);

  if (!links.length) return;

  const articleHeadings = links.map(({ heading }) => heading);
  let frame = 0;
  let activeIndex = -1;

  function keepLinkVisible(link) {
    if (!link || window.getComputedStyle(toc).display === "none") return;

    const tocRect = toc.getBoundingClientRect();
    const linkRect = link.getBoundingClientRect();
    const inset = 16;

    if (linkRect.top < tocRect.top + inset) {
      toc.scrollBy({ top: linkRect.top - tocRect.top - inset, behavior: "auto" });
    } else if (linkRect.bottom > tocRect.bottom - inset) {
      toc.scrollBy({ top: linkRect.bottom - tocRect.bottom + inset, behavior: "auto" });
    }
  }

  function update() {
    frame = 0;
    const viewportTop = window.scrollY;
    const viewportBottom = viewportTop + window.innerHeight;
    const contentRect = content.getBoundingClientRect();
    const contentBottom = contentRect.bottom + window.scrollY;
    const ranges = articleHeadings.map((heading, index) => {
      const start = heading.getBoundingClientRect().top + window.scrollY;
      const nextHeading = articleHeadings[index + 1];
      const end = nextHeading
        ? nextHeading.getBoundingClientRect().top + window.scrollY
        : contentBottom;
      const visible = Math.max(0, Math.min(end, viewportBottom) - Math.max(start, viewportTop));
      const progress = Math.max(0, Math.min(1, (viewportBottom - start) / Math.max(1, end - start)));
      return { visible, progress };
    });

    let nextActiveIndex = 0;
    if (viewportBottom >= contentBottom - 1) {
      nextActiveIndex = links.length - 1;
    } else {
      ranges.forEach(({ visible }, index) => {
        if (visible > ranges[nextActiveIndex].visible) nextActiveIndex = index;
      });
    }

    links.forEach(({ link, progress }, index) => {
      const isActive = index === nextActiveIndex;
      link.classList.toggle("highlight", isActive);
      link.classList.toggle("rounded-top", isActive);
      link.classList.toggle("rounded-bottom", isActive);
      link.toggleAttribute("aria-current", isActive);
      progress.classList.toggle("is-read", !isActive && ranges[index].progress === 1);
      progress.classList.toggle("highlight", isActive);
      progress.style.height = `${ranges[index].progress * 90}%`;
    });

    if (activeIndex !== nextActiveIndex) {
      activeIndex = nextActiveIndex;
      keepLinkVisible(links[activeIndex].link);
    }
  }

  function scheduleUpdate() {
    if (!frame) frame = window.requestAnimationFrame(update);
  }

  const toggle = document.getElementById("toc-toggle");
  const shade = document.getElementById("toc-shade");

  function closeDrawer() {
    toc.classList.remove("show");
    toggle?.setAttribute("aria-expanded", "false");
    if (shade) shade.hidden = true;
  }

  links.forEach(({ link, heading }) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      history.pushState(null, heading.textContent || "", link.getAttribute("href"));
      heading.scrollIntoView({ behavior: "smooth" });
      closeDrawer();
    });
  });

  toggle?.addEventListener("click", () => {
    const open = toc.classList.toggle("show");
    toggle.setAttribute("aria-expanded", String(open));
    if (shade) shade.hidden = !open;
    if (open) window.requestAnimationFrame(() => keepLinkVisible(links[activeIndex]?.link));
  });
  shade?.addEventListener("click", closeDrawer);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate);
  update();
})();
