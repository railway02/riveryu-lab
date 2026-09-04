(() => {
  const toc = document.querySelector(".post-single > .toc");
  const content = document.querySelector(".post-single > .post-content");

  if (!toc || !content) return;

  const details = toc.querySelector("details");
  const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
  const headings = links
    .map((link) => {
      const id = decodeURIComponent(link.hash.slice(1));
      return { link, heading: document.getElementById(id) };
    })
    .filter(({ heading }) => heading);

  if (!headings.length) return;

  const desktop = window.matchMedia("(min-width: 1120px)");
  if (details && !desktop.matches) details.open = false;

  let currentLink = null;
  let frame = 0;

  const revealInToc = (link) => {
    if (!desktop.matches) return;
    const top = link.offsetTop;
    const bottom = top + link.offsetHeight;
    if (top < toc.scrollTop + 12) toc.scrollTop = Math.max(0, top - 12);
    if (bottom > toc.scrollTop + toc.clientHeight - 12) {
      toc.scrollTop = bottom - toc.clientHeight + 12;
    }
  };

  const activate = (link) => {
    if (!link || link === currentLink) return;
    links.forEach((item) => {
      item.classList.remove("is-active");
      item.removeAttribute("aria-current");
    });
    link.classList.add("is-active");
    link.setAttribute("aria-current", "location");
    currentLink = link;
    revealInToc(link);
  };

  const update = () => {
    frame = 0;
    const offset = Math.min(180, window.innerHeight * 0.24);
    let active = headings[0].link;

    for (const item of headings) {
      if (item.heading.getBoundingClientRect().top <= offset) active = item.link;
      else break;
    }

    activate(active);
  };

  const requestUpdate = () => {
    if (!frame) frame = window.requestAnimationFrame(update);
  };

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  window.addEventListener("hashchange", requestUpdate);
  desktop.addEventListener?.("change", (event) => {
    if (details) details.open = event.matches;
    requestUpdate();
  });
  update();
})();
