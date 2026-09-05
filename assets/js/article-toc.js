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

  function update() {
    frame = 0;
    const viewportHeight = window.innerHeight;
    const contentTop = content.offsetTop;
    const pageOffset = window.scrollY - contentTop;
    const postOffset = content.offsetHeight + 127;
    const state = new Map();

    articleHeadings.forEach((heading, index) => {
      const nextTop = articleHeadings[index + 1]?.offsetTop || postOffset;
      const start = heading.offsetTop - pageOffset;
      const end = nextTop - pageOffset - heading.offsetHeight;
      const progress = Math.max(0, Math.min(1, (viewportHeight - start) / Math.max(1, end - start)));
      state.set(heading.id, { inView: start < viewportHeight && end > 0, progress });
    });

    links.forEach(({ link, heading, progress }, index) => {
      const item = state.get(heading.id);
      const previous = links[index - 1] ? state.get(links[index - 1].heading.id) : null;
      const next = links[index + 1] ? state.get(links[index + 1].heading.id) : null;
      link.classList.toggle("highlight", item.inView);
      link.classList.toggle("rounded-top", item.inView && !previous?.inView);
      link.classList.toggle("rounded-bottom", item.inView && !next?.inView);
      link.toggleAttribute("aria-current", item.inView);
      progress.classList.toggle("is-read", !item.inView && item.progress === 1);
      progress.classList.toggle("highlight", item.inView);
      progress.style.height = `${item.progress * 90}%`;
    });
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
      history.pushState(null, heading.textContent || "", link.getAttribute("href"));
      heading.scrollIntoView({ behavior: "smooth" });
      closeDrawer();
    });
  });

  toggle?.addEventListener("click", () => {
    const open = toc.classList.toggle("show");
    toggle.setAttribute("aria-expanded", String(open));
    if (shade) shade.hidden = !open;
  });
  shade?.addEventListener("click", closeDrawer);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeDrawer();
  });
  window.addEventListener("scroll", scheduleUpdate, { passive: true });
  window.addEventListener("resize", scheduleUpdate);
  update();
})();
