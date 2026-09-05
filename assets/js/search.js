(() => {
  "use strict";

  const input = document.getElementById("searchInput");
  const results = document.getElementById("searchResults");
  const status = document.getElementById("searchStatus");
  if (!input || !results || !status) return;

  const normalize = (value) => value.normalize("NFKC").toLocaleLowerCase();
  let index = [];
  let activeIndex = -1;

  function fieldScore(field, terms, weight) {
    const normalized = normalize(field || "");
    if (!terms.every((term) => normalized.includes(term))) return 0;
    return terms.reduce((score, term) => score + weight * (normalized.startsWith(term) ? 2 : 1), 0);
  }

  function rank(item, terms, phrase) {
    const title = normalize(item.title || "");
    let score = title.includes(phrase) ? 40 : 0;
    score += fieldScore(item.title, terms, 12);
    score += fieldScore(item.summary, terms, 5);
    score += fieldScore(item.content, terms, 1);
    return score;
  }

  function createResult(item) {
    const entry = document.createElement("li");
    const link = document.createElement("a");
    const title = document.createElement("span");
    const summary = document.createElement("span");

    link.href = item.permalink;
    title.className = "search-result__title";
    title.textContent = item.title;
    summary.className = "search-result__summary";
    summary.textContent = item.summary || "";
    link.append(title, summary);
    entry.append(link);
    return entry;
  }

  function render() {
    const phrase = normalize(input.value.trim());
    results.replaceChildren();
    activeIndex = -1;
    if (!phrase) {
      status.textContent = "";
      return;
    }

    const terms = phrase.split(/\s+/u).filter(Boolean);
    const matches = index
      .map((item) => ({ item, score: rank(item, terms, phrase) }))
      .filter(({ score }) => score > 0)
      .sort((a, b) => b.score - a.score || a.item.title.localeCompare(b.item.title))
      .slice(0, 12);

    const fragment = document.createDocumentFragment();
    matches.forEach(({ item }) => fragment.append(createResult(item)));
    results.append(fragment);
    status.textContent = matches.length ? `${matches.length} results` : "No matching writing";
  }

  function moveActive(direction) {
    const links = Array.from(results.querySelectorAll("a"));
    if (!links.length) return;
    activeIndex = (activeIndex + direction + links.length) % links.length;
    links[activeIndex].focus();
  }

  input.addEventListener("input", render);
  input.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowDown") return;
    event.preventDefault();
    moveActive(1);
  });
  results.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveActive(event.key === "ArrowDown" ? 1 : -1);
    }
  });

  fetch("../index.json")
    .then((response) => {
      if (!response.ok) throw new Error(`Search index request failed: ${response.status}`);
      return response.json();
    })
    .then((data) => {
      index = Array.isArray(data) ? data : [];
      render();
    })
    .catch(() => {
      status.textContent = "Search is temporarily unavailable";
    });
})();
