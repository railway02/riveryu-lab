(() => {
  "use strict";

  const canvas = document.querySelector("[data-river-flow]");
  const hero = canvas?.closest("[data-river-hero]");
  const context = canvas?.getContext("2d", { alpha: true });
  if (!canvas || !hero || !context) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const pointer = { x: 0, y: 0, active: false, movedAt: 0 };
  const word = canvas.dataset.word || "RiverYu";

  let width = 1;
  let height = 1;
  let pixelRatio = 1;
  let wordParticles = [];
  let streamParticles = [];
  let animationFrame = 0;
  let lastFrame = 0;
  let startedAt = performance.now();
  let isVisible = true;
  let hasBuilt = false;
  let fontsReady = false;

  const randomBetween = (minimum, maximum) => minimum + Math.random() * (maximum - minimum);

  // A divergence-free velocity field derived from a smooth stream function.
  // Its coherent bends read as a current rather than independent particle noise.
  const currentAt = (x, y, time, phase = 0) => {
    const firstX = x * 0.0041 + time * 0.12 + phase;
    const firstY = y * 0.0053 - time * 0.09 - phase * 0.37;
    const secondX = x * 0.0023 - time * 0.07 - phase * 0.22;
    const secondY = y * 0.0031 + time * 0.08 + phase * 0.19;

    const velocityX = (
      0.0053 * Math.sin(firstX) * Math.cos(firstY)
      + 0.0031 * 0.64 * Math.cos(secondX) * Math.cos(secondY)
    );
    const velocityY = -(
      0.0041 * Math.cos(firstX) * Math.sin(firstY)
      - 0.0023 * 0.64 * Math.sin(secondX) * Math.sin(secondY)
    );
    const magnitude = Math.hypot(velocityX, velocityY) || 1;

    return {
      x: velocityX / magnitude,
      y: velocityY / magnitude,
    };
  };

  const buildWordParticles = (settled) => {
    const sample = document.createElement("canvas");
    const sampleContext = sample.getContext("2d", { willReadFrequently: true });
    if (!sampleContext) return [];

    const sampleScale = Math.min(1, 1440 / width, 900 / height);
    sample.width = Math.max(1, Math.round(width * sampleScale));
    sample.height = Math.max(1, Math.round(height * sampleScale));

    const styles = getComputedStyle(document.documentElement);
    const fontFamily = styles.getPropertyValue("--river-font-display").trim() || "sans-serif";
    const maximumWidth = sample.width * (width < 680 ? 0.9 : 0.84);
    let fontSize = Math.min(sample.height * 0.32, sample.width * 0.235);

    sampleContext.font = `650 ${fontSize}px ${fontFamily}`;
    const measuredWidth = sampleContext.measureText(word).width;
    if (measuredWidth > maximumWidth) fontSize *= maximumWidth / measuredWidth;

    const centerX = sample.width / 2;
    const centerY = sample.height * 0.505;
    sampleContext.clearRect(0, 0, sample.width, sample.height);
    sampleContext.fillStyle = "#fff";
    sampleContext.font = `650 ${fontSize}px ${fontFamily}`;
    sampleContext.textAlign = "center";
    sampleContext.textBaseline = "middle";
    sampleContext.fillText(word, centerX, centerY);

    const pixels = sampleContext.getImageData(0, 0, sample.width, sample.height).data;
    const gap = 3;
    const targets = [];

    for (let y = 0; y < sample.height; y += gap) {
      for (let x = 0; x < sample.width; x += gap) {
        const alpha = pixels[(y * sample.width + x) * 4 + 3];
        if (alpha > 80) {
          targets.push({
            x: x / sampleScale,
            y: y / sampleScale,
            alpha: alpha / 255,
          });
        }
      }
    }

    return targets.map((target, index) => {
      const entersFromLeft = index % 5 !== 0;
      const startX = settled
        ? target.x + randomBetween(-2.4, 2.4)
        : entersFromLeft
          ? randomBetween(-width * 0.24, -24)
          : randomBetween(width + 24, width * 1.16);
      const band = Math.sin(target.y * 0.018 + index * 0.037) * height * 0.1;
      const startY = settled
        ? target.y + randomBetween(-1.8, 1.8)
        : target.y + band + randomBetween(-height * 0.16, height * 0.16);

      return {
        x: startX,
        y: startY,
        previousX: startX,
        previousY: startY,
        targetX: target.x,
        targetY: target.y,
        velocityX: entersFromLeft ? randomBetween(0.25, 0.9) : randomBetween(-0.6, -0.2),
        velocityY: randomBetween(-0.2, 0.2),
        ease: randomBetween(0.018, 0.06),
        friction: 0.2,
        phase: randomBetween(0, Math.PI * 2),
        delay: settled ? 0 : randomBetween(0, 1150),
        size: gap - 1,
        tone: index % 3,
        alpha: target.alpha,
      };
    });
  };

  const buildStreamParticles = () => {
    const count = Math.min(150, Math.max(width < 680 ? 42 : 72, Math.round(width * height / 14500)));
    return Array.from({ length: count }, (_, index) => ({
      x: Math.random() * width,
      y: Math.random() * height,
      previousX: 0,
      previousY: 0,
      speed: randomBetween(0.18, 0.52),
      phase: randomBetween(0, Math.PI * 2),
      tone: index % 3,
    })).map((particle) => ({
      ...particle,
      previousX: particle.x,
      previousY: particle.y,
    }));
  };

  const applyPointerCollision = (particle, timestamp) => {
    if (!pointer.active || timestamp - pointer.movedAt > 1200) return;

    const distanceX = particle.x - pointer.x;
    const distanceY = particle.y - pointer.y;
    const distanceSquared = distanceX * distanceX + distanceY * distanceY;
    const interactionArea = width < 680 ? 200 : 300;
    if (distanceSquared <= 0 || distanceSquared >= interactionArea) return;

    const distance = Math.sqrt(distanceSquared);
    const force = (interactionArea * 2000) / distanceSquared;
    particle.velocityX += (distanceX / distance) * force;
    particle.velocityY += (distanceY / distance) * force;
  };

  const updateWordParticles = (time, elapsed, timestamp) => {
    for (const particle of wordParticles) {
      particle.previousX = particle.x;
      particle.previousY = particle.y;

      if (reducedMotion.matches) {
        particle.x = particle.targetX;
        particle.y = particle.targetY;
        continue;
      }

      const current = currentAt(particle.x, particle.y, time, particle.phase);
      const targetCurrent = currentAt(particle.targetX, particle.targetY, time * 0.74, particle.phase);
      const targetX = particle.targetX + targetCurrent.x * 2.1;
      const targetY = particle.targetY + targetCurrent.y * 2.7;

      applyPointerCollision(particle, timestamp);
      particle.velocityX *= particle.friction;
      particle.velocityY *= particle.friction;

      if (elapsed >= particle.delay) {
        particle.x += particle.velocityX + (targetX - particle.x) * particle.ease + current.x * 0.06;
        particle.y += particle.velocityY + (targetY - particle.y) * particle.ease + current.y * 0.06;
      } else {
        particle.x += particle.velocityX + current.x * 0.06;
        particle.y += particle.velocityY + current.y * 0.06;
      }
    }
  };

  const updateStreamParticles = (time) => {
    for (const particle of streamParticles) {
      particle.previousX = particle.x;
      particle.previousY = particle.y;
      const current = currentAt(particle.x, particle.y, time, particle.phase);
      particle.x += (current.x + 0.72) * particle.speed;
      particle.y += current.y * particle.speed * 0.78;

      if (particle.x > width + 24 || particle.y < -24 || particle.y > height + 24) {
        particle.x = -24;
        particle.y = Math.random() * height;
        particle.previousX = particle.x;
        particle.previousY = particle.y;
      }
    }
  };

  const palette = () => {
    const dark = document.documentElement.dataset.theme === "dark";
    return dark
      ? {
        word: ["rgba(83, 187, 217, .38)", "rgba(125, 211, 232, .46)", "rgba(178, 231, 242, .4)"],
        stream: ["rgba(91, 187, 211, .055)", "rgba(129, 207, 225, .07)", "rgba(182, 230, 240, .055)"],
        composite: "screen",
      }
      : {
        word: ["rgba(27, 128, 169, .31)", "rgba(47, 153, 191, .38)", "rgba(100, 190, 216, .42)"],
        stream: ["rgba(36, 137, 174, .045)", "rgba(71, 171, 205, .06)", "rgba(117, 198, 220, .05)"],
        composite: "multiply",
      };
  };

  const drawStreamParticles = (particles, colors) => {
    context.lineCap = "round";

    for (let tone = 0; tone < colors.length; tone += 1) {
      context.strokeStyle = colors[tone];
      context.lineWidth = 0.55 + tone * 0.12;
      context.beginPath();

      for (const particle of particles) {
        if (particle.tone !== tone) continue;
        context.moveTo(particle.previousX, particle.previousY);
        context.lineTo(particle.x, particle.y);
      }

      context.stroke();
    }
  };

  const drawWordParticles = (particles, colors) => {
    for (let tone = 0; tone < colors.length; tone += 1) {
      context.fillStyle = colors[tone];

      for (const particle of particles) {
        if (particle.tone !== tone) continue;
        const size = particle.size * (0.78 + particle.alpha * 0.22);
        context.fillRect(particle.x - size / 2, particle.y - size / 2, size, size);
      }
    }
  };

  const draw = () => {
    context.clearRect(0, 0, width, height);
    const colors = palette();
    drawStreamParticles(streamParticles, colors.stream);
    context.globalCompositeOperation = colors.composite;
    drawWordParticles(wordParticles, colors.word);
    context.globalCompositeOperation = "source-over";
  };

  const render = (timestamp) => {
    if (!isVisible || document.hidden) {
      animationFrame = 0;
      return;
    }

    const frameInterval = 1000 / 30;
    if (timestamp - lastFrame >= frameInterval || reducedMotion.matches) {
      const time = timestamp * 0.001;
      updateStreamParticles(time);
      updateWordParticles(time, timestamp - startedAt, timestamp);
      draw();
      lastFrame = timestamp;
      hero.classList.add("is-flow-ready");
    }

    animationFrame = reducedMotion.matches ? 0 : requestAnimationFrame(render);
  };

  const start = () => {
    if (hasBuilt && !animationFrame && isVisible && !document.hidden) {
      animationFrame = requestAnimationFrame(render);
    }
  };

  const stop = () => {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  };

  const resize = (force = false) => {
    const bounds = hero.getBoundingClientRect();
    const nextWidth = Math.max(1, Math.round(bounds.width));
    const nextHeight = Math.max(1, Math.round(bounds.height));
    if (!force && hasBuilt && nextWidth === width && nextHeight === height) return;

    width = nextWidth;
    height = nextHeight;
    pixelRatio = Math.min(
      window.devicePixelRatio || 1,
      width < 680 ? 1.15 : 1.35,
      2400 / width,
      1350 / height,
    );
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    wordParticles = buildWordParticles(hasBuilt || reducedMotion.matches);
    streamParticles = buildStreamParticles();
    startedAt = performance.now();
    lastFrame = 0;
    hasBuilt = true;
    start();
  };

  hero.addEventListener("pointermove", (event) => {
    const bounds = hero.getBoundingClientRect();
    pointer.x = event.clientX - bounds.left;
    pointer.y = event.clientY - bounds.top;
    pointer.active = true;
    pointer.movedAt = performance.now();
  }, { passive: true });

  hero.addEventListener("pointerleave", () => {
    pointer.active = false;
  }, { passive: true });

  new ResizeObserver(() => {
    if (fontsReady) resize();
  }).observe(hero);
  new IntersectionObserver(([entry]) => {
    isVisible = entry.isIntersecting;
    if (isVisible) start(); else stop();
  }).observe(hero);

  new MutationObserver(() => {
    if (reducedMotion.matches) draw();
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop(); else start();
  });

  reducedMotion.addEventListener("change", () => resize(true));
  document.fonts.ready.then(() => {
    fontsReady = true;
    resize(true);
  });
})();
