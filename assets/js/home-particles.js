(() => {
  "use strict";

  const canvas = document.querySelector("[data-river-particles]");
  const surface = canvas?.closest(".river-home");
  if (!canvas || !surface) return;

  const context = canvas.getContext("2d", { alpha: true });
  if (!context) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const title = canvas.dataset.title;
  const tau = Math.PI * 2;
  const pointer = {
    x: 0,
    y: 0,
    active: false,
    movedAt: 0,
  };

  let width = 1;
  let height = 1;
  let pixelRatio = 1;
  let titleParticles = [];
  let currentParticles = [];
  let graphNodes = [];
  let animationFrame = 0;
  let visible = true;
  let firstBuild = true;
  let startedAt = performance.now();

  const randomBetween = (minimum, maximum) => minimum + Math.random() * (maximum - minimum);

  const flowAngle = (x, y, time, phase = 0) => (
    Math.sin(y * 0.0062 + time * 0.23 + phase) * 0.72
    + Math.cos(x * 0.0038 - time * 0.17 - phase * 0.4) * 0.46
    + Math.sin((x + y) * 0.0022 + time * 0.11) * 0.24
  );

  const wrapParticle = (particle, margin = 24) => {
    if (particle.x < -margin) particle.x = width + margin;
    if (particle.x > width + margin) particle.x = -margin;
    if (particle.y < -margin) particle.y = height + margin;
    if (particle.y > height + margin) particle.y = -margin;
  };

  const buildTitleParticles = (settled) => {
    const sampleCanvas = document.createElement("canvas");
    const sampleContext = sampleCanvas.getContext("2d", { willReadFrequently: true });
    const sampleScale = Math.min(1, 1920 / width, 1080 / height);
    sampleCanvas.width = Math.ceil(width * sampleScale);
    sampleCanvas.height = Math.ceil(height * sampleScale);

    const styles = getComputedStyle(surface);
    const fontFamily = styles.getPropertyValue("--river-display-font").trim();
    const maximumWidth = sampleCanvas.width * (width < 680 ? 0.88 : 0.82);
    let fontSize = Math.min(sampleCanvas.height * 0.33, sampleCanvas.width * 0.3);

    sampleContext.font = `600 ${fontSize}px ${fontFamily}`;
    const measuredWidth = sampleContext.measureText(title).width;
    if (measuredWidth > maximumWidth) fontSize *= maximumWidth / measuredWidth;

    const titleY = sampleCanvas.height * (width < 680 ? 0.445 : 0.46);
    sampleContext.clearRect(0, 0, sampleCanvas.width, sampleCanvas.height);
    sampleContext.fillStyle = "#fff";
    sampleContext.font = `600 ${fontSize}px ${fontFamily}`;
    sampleContext.textAlign = "center";
    sampleContext.textBaseline = "middle";
    sampleContext.fillText(title, sampleCanvas.width / 2, titleY);

    const pixels = sampleContext.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height).data;
    const gap = width < 680 ? 3 : 4;
    const targets = [];

    for (let y = 0; y < sampleCanvas.height; y += gap) {
      for (let x = 0; x < sampleCanvas.width; x += gap) {
        const alpha = pixels[(y * sampleCanvas.width + x) * 4 + 3];
        if (alpha > 72) {
          targets.push({
            x: x / sampleScale,
            y: y / sampleScale,
            alpha: alpha / 255,
          });
        }
      }
    }

    titleParticles = targets.map((target, index) => {
      const angle = Math.random() * tau;
      const radiusX = randomBetween(width * 0.24, width * 0.72);
      const radiusY = randomBetween(height * 0.12, height * 0.46);
      const startX = settled ? target.x + randomBetween(-8, 8) : width / 2 + Math.cos(angle) * radiusX;
      const startY = settled
        ? target.y + randomBetween(-8, 8)
        : titleY / sampleScale + Math.sin(angle) * radiusY;
      const particleScale = Math.sqrt(1 / sampleScale);

      return {
        x: startX,
        y: startY,
        targetX: target.x,
        targetY: target.y,
        velocityX: settled ? 0 : Math.cos(angle + Math.PI / 2) * randomBetween(0.2, 1.25),
        velocityY: settled ? 0 : Math.sin(angle + Math.PI / 2) * randomBetween(0.2, 1.25),
        spring: randomBetween(0.012, 0.024),
        friction: randomBetween(0.86, 0.9),
        phase: Math.random() * tau,
        flow: randomBetween(0.7, 1.25),
        delay: settled ? 0 : Math.random() * 1050,
        size: (gap === 3 ? randomBetween(1.25, 2.15) : randomBetween(1.4, 2.45)) * particleScale,
        tone: (index + Math.floor(target.x / 90)) % 3,
        alpha: target.alpha,
      };
    });
  };

  const buildCurrentParticles = () => {
    const count = Math.max(110, Math.min(440, Math.round((width * height) / 4200)));
    currentParticles = Array.from({ length: count }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      previousX: 0,
      previousY: 0,
      speed: randomBetween(0.18, 0.62),
      phase: Math.random() * tau,
      size: randomBetween(0.45, 1.35),
      tone: Math.floor(Math.random() * 3),
    }));

    for (const particle of currentParticles) {
      particle.previousX = particle.x;
      particle.previousY = particle.y;
    }
  };

  const buildGraphNodes = () => {
    const count = width < 680 ? 16 : 28;
    graphNodes = Array.from({ length: count }, (_, index) => ({
      x: randomBetween(width * 0.04, width * 0.96),
      y: randomBetween(height * 0.12, height * 0.88),
      phase: (index / count) * tau + Math.random(),
      speed: randomBetween(0.08, 0.22),
      radius: randomBetween(0.8, 1.75),
    }));
  };

  const updatePointerForce = (particle, strength, timestamp) => {
    if (!pointer.active || timestamp - pointer.movedAt > 1800) return;

    const distanceX = particle.x - pointer.x;
    const distanceY = particle.y - pointer.y;
    const distanceSquared = distanceX * distanceX + distanceY * distanceY;
    const radius = Math.min(190, Math.max(115, width * 0.115));

    if (distanceSquared === 0 || distanceSquared >= radius * radius) return;

    const distance = Math.sqrt(distanceSquared);
    const force = (1 - distance / radius) * strength;
    const normalX = distanceX / distance;
    const normalY = distanceY / distance;
    particle.velocityX += normalX * force - normalY * force * 0.42;
    particle.velocityY += normalY * force + normalX * force * 0.42;
  };

  const updateTitleParticles = (time, elapsed, timestamp) => {
    for (const particle of titleParticles) {
      if (reducedMotion.matches) {
        particle.x = particle.targetX;
        particle.y = particle.targetY;
        continue;
      }

      const waveX = Math.sin(particle.targetY * 0.019 + time * 0.62 + particle.phase) * 0.9;
      const waveY = Math.sin(particle.targetX * 0.012 - time * 0.78 + particle.phase) * 1.25;
      const angle = flowAngle(particle.x, particle.y, time, particle.phase);

      if (elapsed > particle.delay) {
        particle.velocityX += (particle.targetX + waveX - particle.x) * particle.spring;
        particle.velocityY += (particle.targetY + waveY - particle.y) * particle.spring;
      }

      particle.velocityX += Math.cos(angle) * 0.012 * particle.flow;
      particle.velocityY += Math.sin(angle) * 0.012 * particle.flow;
      updatePointerForce(particle, 1.7, timestamp);

      particle.velocityX *= particle.friction;
      particle.velocityY *= particle.friction;
      particle.x += particle.velocityX;
      particle.y += particle.velocityY;
    }
  };

  const updateCurrentParticles = (time) => {
    for (const particle of currentParticles) {
      particle.previousX = particle.x;
      particle.previousY = particle.y;

      const angle = flowAngle(particle.x, particle.y, time, particle.phase);
      particle.x += (Math.cos(angle) + 0.36) * particle.speed;
      particle.y += Math.sin(angle) * particle.speed * 0.78;
      wrapParticle(particle);

      if (Math.abs(particle.x - particle.previousX) > width / 2) particle.previousX = particle.x;
      if (Math.abs(particle.y - particle.previousY) > height / 2) particle.previousY = particle.y;
    }
  };

  const updateGraphNodes = (time) => {
    for (const node of graphNodes) {
      const angle = flowAngle(node.x, node.y, time, node.phase);
      node.x += (Math.cos(angle) + 0.12) * node.speed;
      node.y += Math.sin(angle) * node.speed * 0.65;
      wrapParticle(node, 80);
    }
  };

  const drawGraph = () => {
    const connectionDistance = Math.min(190, Math.max(120, width * 0.11));
    const distanceSquaredLimit = connectionDistance * connectionDistance;
    context.lineWidth = 0.65;

    for (let first = 0; first < graphNodes.length; first += 1) {
      const source = graphNodes[first];

      for (let second = first + 1; second < graphNodes.length; second += 1) {
        const target = graphNodes[second];
        const distanceX = source.x - target.x;
        const distanceY = source.y - target.y;
        const distanceSquared = distanceX * distanceX + distanceY * distanceY;
        if (distanceSquared >= distanceSquaredLimit) continue;

        const alpha = (1 - Math.sqrt(distanceSquared) / connectionDistance) * 0.12;
        context.strokeStyle = `rgba(89, 209, 204, ${alpha})`;
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.lineTo(target.x, target.y);
        context.stroke();
      }
    }

    context.fillStyle = "rgba(112, 231, 224, 0.28)";
    context.beginPath();
    for (const node of graphNodes) {
      context.moveTo(node.x + node.radius, node.y);
      context.arc(node.x, node.y, node.radius, 0, tau);
    }
    context.fill();
  };

  const drawCurrentParticles = () => {
    const colors = [
      "rgba(41, 146, 153, 0.18)",
      "rgba(78, 205, 199, 0.2)",
      "rgba(145, 242, 232, 0.16)",
    ];

    context.lineCap = "round";
    context.lineWidth = 0.7;

    for (let tone = 0; tone < colors.length; tone += 1) {
      context.strokeStyle = colors[tone];
      context.fillStyle = colors[tone];
      context.beginPath();

      for (const particle of currentParticles) {
        if (particle.tone !== tone) continue;
        context.moveTo(particle.previousX, particle.previousY);
        context.quadraticCurveTo(
          (particle.previousX + particle.x) / 2 + Math.sin(particle.phase) * 0.8,
          (particle.previousY + particle.y) / 2 + Math.cos(particle.phase) * 0.8,
          particle.x,
          particle.y,
        );
      }

      context.stroke();
      context.beginPath();
      for (const particle of currentParticles) {
        if (particle.tone !== tone) continue;
        context.moveTo(particle.x + particle.size, particle.y);
        context.arc(particle.x, particle.y, particle.size, 0, tau);
      }
      context.fill();
    }
  };

  const drawTitleParticles = () => {
    const colors = [
      "rgba(65, 205, 201, 0.82)",
      "rgba(91, 233, 222, 0.92)",
      "rgba(174, 255, 244, 0.96)",
    ];

    context.globalCompositeOperation = "lighter";
    for (let tone = 0; tone < colors.length; tone += 1) {
      context.fillStyle = colors[tone];
      for (const particle of titleParticles) {
        if (particle.tone !== tone) continue;
        const size = particle.size * (0.72 + particle.alpha * 0.28);
        context.fillRect(particle.x - size / 2, particle.y - size / 2, size, size);
      }
    }
    context.globalCompositeOperation = "source-over";
  };

  const render = (timestamp) => {
    if (!visible || document.hidden) {
      animationFrame = 0;
      return;
    }

    const time = timestamp * 0.001;
    const elapsed = timestamp - startedAt;
    context.clearRect(0, 0, width, height);

    updateGraphNodes(time);
    updateCurrentParticles(time);
    updateTitleParticles(time, elapsed, timestamp);
    drawGraph();
    drawCurrentParticles();
    drawTitleParticles();

    surface.classList.add("is-ready");
    animationFrame = reducedMotion.matches ? 0 : requestAnimationFrame(render);
  };

  const start = () => {
    if (!animationFrame && visible && !document.hidden) {
      animationFrame = requestAnimationFrame(render);
    }
  };

  const stop = () => {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    animationFrame = 0;
  };

  const resize = (force = false) => {
    const bounds = surface.getBoundingClientRect();
    const nextWidth = Math.max(1, Math.round(bounds.width));
    const nextHeight = Math.max(1, Math.round(bounds.height));
    if (!force && !firstBuild && nextWidth === width && nextHeight === height) return;

    width = nextWidth;
    height = nextHeight;
    pixelRatio = Math.min(
      window.devicePixelRatio || 1,
      1.75,
      3200 / width,
      1800 / height,
    );

    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    buildTitleParticles(!firstBuild);
    buildCurrentParticles();
    buildGraphNodes();
    startedAt = performance.now();
    firstBuild = false;
    start();
  };

  surface.addEventListener("pointermove", (event) => {
    const bounds = surface.getBoundingClientRect();
    pointer.x = event.clientX - bounds.left;
    pointer.y = event.clientY - bounds.top;
    pointer.active = true;
    pointer.movedAt = performance.now();
  }, { passive: true });

  surface.addEventListener("pointerleave", () => {
    pointer.active = false;
  }, { passive: true });

  new ResizeObserver(() => resize()).observe(surface);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) start(); else stop();
  }).observe(surface);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop(); else start();
  });

  reducedMotion.addEventListener("change", () => resize(true));
  document.fonts.ready.then(resize);
})();
