(() => {
  "use strict";

  if (!document.querySelector(".post-single .code-block")) return;

  const copiedTimers = new WeakMap();

  function copyFallback(text) {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();

    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (_) {
      copied = false;
    }

    textarea.remove();
    return copied;
  }

  async function copyCode(button) {
    const block = button.closest("[data-code-block]");
    const code = block?.querySelector("code");
    if (!code) return;

    let copied = false;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(code.textContent || "");
        copied = true;
      } catch (_) {
        copied = copyFallback(code.textContent || "");
      }
    } else {
      copied = copyFallback(code.textContent || "");
    }

    if (!copied) return;

    button.textContent = "Copied";
    window.clearTimeout(copiedTimers.get(button));
    copiedTimers.set(button, window.setTimeout(() => {
      button.textContent = "Copy";
    }, 1800));
  }

  function toggleWrap(button) {
    const block = button.closest("[data-code-block]");
    if (!block) return;

    const wrapped = block.classList.toggle("is-wrapped");
    button.setAttribute("aria-pressed", String(wrapped));
    button.textContent = wrapped ? "No wrap" : "Wrap";
  }

  function syncToc(mediaQuery) {
    document.querySelectorAll(".post-single > .toc details").forEach((details) => {
      details.open = !mediaQuery.matches;
    });
  }

  document.addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-code-copy]");
    if (copyButton) {
      copyCode(copyButton);
      return;
    }

    const wrapButton = event.target.closest("[data-code-wrap]");
    if (wrapButton) toggleWrap(wrapButton);
  });

  const mobileToc = window.matchMedia("(max-width: 1119px)");
  syncToc(mobileToc);
  if (mobileToc.addEventListener) {
    mobileToc.addEventListener("change", () => syncToc(mobileToc));
  } else {
    mobileToc.addListener(() => syncToc(mobileToc));
  }
})();
