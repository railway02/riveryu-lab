(() => {
  "use strict";

  const canvas = document.querySelector("[data-ocean-current]");
  const landing = canvas?.closest(".home-landing");
  if (!canvas || !landing) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const startedAt = performance.now();
  let targetX = 0.5;
  let targetY = 0.5;
  let pointerX = 0.5;
  let pointerY = 0.5;

  const easeOutCubic = (value) => 1 - Math.pow(1 - value, 3);

  const bindPointer = () => {
    landing.addEventListener("pointermove", (event) => {
      const rect = landing.getBoundingClientRect();
      targetX = (event.clientX - rect.left) / rect.width;
      targetY = (event.clientY - rect.top) / rect.height;
    }, { passive: true });

    landing.addEventListener("pointerleave", () => {
      targetX = 0.5;
      targetY = 0.5;
    }, { passive: true });
  };

  const runCanvasOcean = () => {
    const context = canvas.getContext("2d");
    if (!context) return;

    let width = 1;
    let height = 1;
    let frame = 0;
    let visible = true;
    let lastFrame = 0;

    const resize = () => {
      const rect = landing.getBoundingClientRect();
      const mobile = window.matchMedia("(max-width: 640px)").matches;
      const scale = Math.min(window.devicePixelRatio || 1, mobile ? 1 : 1.25);
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      context.setTransform(scale, 0, 0, scale, 0, 0);
    };

    const draw = (timestamp) => {
      if (!visible || document.hidden) {
        frame = 0;
        return;
      }
      if (!reducedMotion.matches && timestamp - lastFrame < 32) {
        frame = requestAnimationFrame(draw);
        return;
      }

      lastFrame = timestamp;
      pointerX += (targetX - pointerX) * 0.05;
      pointerY += (targetY - pointerY) * 0.05;

      const elapsed = reducedMotion.matches ? 4200 : timestamp - startedAt;
      const progress = reducedMotion.matches ? 1 : Math.min(1, elapsed / 1850);
      const arrival = easeOutCubic(progress);
      const horizon = height * 0.235;
      const front = horizon + (height - horizon + 28) * arrival;
      const phase = elapsed * 0.001;

      const sky = context.createRadialGradient(width * 0.5, horizon, 0, width * 0.5, horizon, width * 0.72);
      sky.addColorStop(0, "#082a35");
      sky.addColorStop(0.28, "#03151c");
      sky.addColorStop(1, "#02070a");
      context.fillStyle = sky;
      context.fillRect(0, 0, width, height);

      context.save();
      context.beginPath();
      context.moveTo(0, horizon);
      context.lineTo(width, horizon);
      for (let x = width; x >= 0; x -= 16) {
        const frontWave = Math.sin(x * 0.014 - phase * 4.8) * (4 + (1 - progress) * 15);
        context.lineTo(x, front + frontWave);
      }
      context.closePath();
      context.clip();

      const water = context.createLinearGradient(0, horizon, 0, height);
      water.addColorStop(0, "#0d5263");
      water.addColorStop(0.28, "#0a3f52");
      water.addColorStop(0.7, "#07283a");
      water.addColorStop(1, "#041622");
      context.fillStyle = water;
      context.fillRect(0, horizon, width, height - horizon);

      const nearCenter = width * (0.5 + (pointerX - 0.5) * 0.22);
      const channel = context.createLinearGradient(0, horizon, 0, height);
      channel.addColorStop(0, "rgba(72, 205, 217, .12)");
      channel.addColorStop(0.52, "rgba(39, 142, 166, .22)");
      channel.addColorStop(1, "rgba(29, 102, 134, .07)");
      context.beginPath();
      context.moveTo(width * 0.485, horizon);
      context.bezierCurveTo(width * 0.44, height * 0.48, nearCenter - width * 0.18, height * 0.76, nearCenter - width * 0.31, height + 20);
      context.lineTo(nearCenter + width * 0.31, height + 20);
      context.bezierCurveTo(nearCenter + width * 0.18, height * 0.76, width * 0.56, height * 0.48, width * 0.515, horizon);
      context.closePath();
      context.fillStyle = channel;
      context.fill();

      for (let index = 0; index < 30; index += 1) {
        const depth = index / 29;
        const y = horizon + (front - horizon) * Math.pow(depth, 1.58);
        if (y > front + 24 || y > height + 30) continue;
        const amplitude = 1.3 + depth * 17;
        const frequency = 0.009 - depth * 0.004;
        const pointerWake = (pointerX - 0.5) * depth * width * 0.08;

        context.beginPath();
        for (let x = -30; x <= width + 30; x += 14) {
          const wave = Math.sin(x * frequency + phase * (0.72 + depth * 1.1) + index * 0.62) * amplitude;
          const crossWave = Math.sin(x * 0.0028 - phase * 0.42 + index) * amplitude * 0.42;
          const wakeDistance = (x / width - pointerX) * 4.6;
          const wake = Math.exp(-(wakeDistance * wakeDistance)) * (pointerY - 0.52) * depth * 36;
          const pointY = y + wave + crossWave + wake;
          const pointX = x + pointerWake;
          if (x === -30) context.moveTo(pointX, pointY); else context.lineTo(pointX, pointY);
        }

        const alpha = 0.075 + depth * 0.13;
        context.strokeStyle = index % 6 === 0
          ? `rgba(124, 229, 236, ${alpha + 0.06})`
          : `rgba(89, 176, 196, ${alpha})`;
        context.lineWidth = 0.55 + depth * 1.45;
        context.stroke();
      }

      context.restore();

      if (progress < 0.985) {
        context.save();
        context.beginPath();
        for (let x = -20; x <= width + 20; x += 10) {
          const y = front + Math.sin(x * 0.014 - phase * 4.8) * (4 + (1 - progress) * 15);
          if (x === -20) context.moveTo(x, y); else context.lineTo(x, y);
        }
        context.strokeStyle = `rgba(188, 244, 245, ${0.16 + (1 - progress) * 0.42})`;
        context.lineWidth = 1.2 + (1 - progress) * 2.6;
        context.shadowColor = "rgba(99, 218, 229, .7)";
        context.shadowBlur = 12 + (1 - progress) * 16;
        context.stroke();

        for (let index = 0; index < 34; index += 1) {
          const seed = (index * 73.17) % 101;
          const x = (seed / 101) * width;
          const lift = 8 + Math.abs(Math.sin(index * 9.31 + phase * 2.3)) * 52 * (1 - progress * 0.5);
          const y = front - lift;
          context.beginPath();
          context.arc(x, y, 0.7 + (index % 4) * 0.45, 0, Math.PI * 2);
          context.fillStyle = `rgba(169, 237, 241, ${0.08 + (1 - progress) * 0.26})`;
          context.fill();
        }
        context.restore();
      }

      const vignette = context.createRadialGradient(width * 0.5, height * 0.5, width * 0.05, width * 0.5, height * 0.5, width * 0.72);
      vignette.addColorStop(0, "rgba(0, 0, 0, 0)");
      vignette.addColorStop(0.7, "rgba(0, 4, 8, .08)");
      vignette.addColorStop(1, "rgba(0, 3, 6, .58)");
      context.fillStyle = vignette;
      context.fillRect(0, 0, width, height);

      landing.classList.add("is-ocean-ready");
      frame = reducedMotion.matches ? 0 : requestAnimationFrame(draw);
    };

    const start = () => {
      if (!frame && visible && !document.hidden) frame = requestAnimationFrame(draw);
    };
    const stop = () => {
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
    };

    bindPointer();
    if ("ResizeObserver" in window) {
      new ResizeObserver(() => {
        resize();
        if (reducedMotion.matches) start();
      }).observe(landing);
    } else {
      window.addEventListener("resize", resize, { passive: true });
    }
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        if (visible) start(); else stop();
      }).observe(landing);
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop(); else start();
    });

    resize();
    start();
  };

  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    powerPreference: "high-performance",
  });

  if (!gl) {
    runCanvasOcean();
    return;
  }

  const vertexSource = `
    attribute vec2 a_position;
    varying vec2 v_uv;
    void main() {
      v_uv = a_position * 0.5 + 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  const fragmentSource = `
    precision mediump float;
    varying vec2 v_uv;
    uniform vec2 u_resolution;
    uniform vec2 u_pointer;
    uniform float u_time;
    uniform float u_reduced;

    float hash(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
    }

    float fbm(vec2 p) {
      float value = 0.0;
      float amplitude = 0.5;
      for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p);
        p = mat2(1.62, 1.18, -1.18, 1.62) * p + 0.17;
        amplitude *= 0.48;
      }
      return value;
    }

    float waveHeight(vec2 p, float travel) {
      float largeWave = sin(p.x * 0.62 + p.y * 1.15 - travel * 1.8) * 0.44;
      float crossing = sin(p.x * 1.18 - p.y * 0.83 + travel * 1.22) * 0.28;
      float chop = sin(p.x * 2.8 + p.y * 2.15 - travel * 2.35) * 0.10;
      return largeWave + crossing + chop + (fbm(p * 0.72 - travel * 0.09) - 0.5) * 0.42;
    }

    void main() {
      vec2 uv = v_uv;
      float aspect = u_resolution.x / max(u_resolution.y, 1.0);
      float time = u_reduced > 0.5 ? 4.2 : u_time;
      float progress = u_reduced > 0.5 ? 1.0 : clamp(time / 1.85, 0.0, 1.0);
      float arrival = 1.0 - pow(1.0 - progress, 3.0);
      float horizon = 0.765;
      float belowHorizon = horizon - uv.y;
      float waterMask = 1.0 - smoothstep(horizon - 0.012, horizon + 0.012, uv.y);
      float front = mix(horizon, -0.08, arrival);
      float reveal = smoothstep(front - 0.055, front + 0.022, uv.y) * waterMask;

      float depth = 0.46 / max(belowHorizon, 0.018);
      float nearField = 1.0 - clamp((depth - 0.62) / 18.0, 0.0, 1.0);
      float travel = time * 0.72 + 7.8 * (1.0 - exp(-time * 1.65));
      vec2 world = vec2((uv.x - 0.5) * depth * 2.75 * aspect, depth - travel);
      world.x += (u_pointer.x - 0.5) * nearField * 1.15;

      float height = waveHeight(world, travel);
      float dx = waveHeight(world + vec2(0.035, 0.0), travel) - height;
      float dz = waveHeight(world + vec2(0.0, 0.035), travel) - height;
      vec3 normal = normalize(vec3(-dx * 5.2, 0.18, -dz * 4.6));
      vec3 lightDirection = normalize(vec3(-0.45, 0.82, -0.34));
      float diffuse = max(dot(normal, lightDirection), 0.0);
      float fresnel = pow(1.0 - max(normal.y, 0.0), 2.2);

      float distanceGlow = clamp(depth / 16.0, 0.0, 1.0);
      vec3 deep = vec3(0.008, 0.075, 0.115);
      vec3 teal = vec3(0.018, 0.29, 0.36);
      vec3 blue = vec3(0.025, 0.20, 0.34);
      vec3 water = mix(deep, teal, 0.28 + diffuse * 0.56);
      water = mix(water, blue, fresnel * 0.72 + distanceGlow * 0.18);

      float crest = smoothstep(0.48, 0.94, height + diffuse * 0.22);
      float causticA = 1.0 - smoothstep(0.02, 0.12, abs(sin(world.x * 1.18 + sin(world.y * 0.74 - travel))));
      float causticB = 1.0 - smoothstep(0.025, 0.13, abs(sin(world.y * 1.05 - world.x * 0.31 - travel * 1.4)));
      float caustic = causticA * causticB * (0.25 + nearField * 0.75);
      water += vec3(0.18, 0.62, 0.67) * crest * 0.36;
      water += vec3(0.10, 0.48, 0.58) * caustic * 0.22;

      float riverCenter = 0.5 + sin(depth * 0.42 - time * 0.28) * 0.036 + (u_pointer.x - 0.5) * nearField * 0.11;
      float riverWidth = mix(0.022, 0.29, pow(nearField, 1.34));
      float river = 1.0 - smoothstep(riverWidth, riverWidth + 0.075, abs(uv.x - riverCenter));
      water = mix(water, water + vec3(0.015, 0.12, 0.15), river * 0.56);

      vec3 sky = vec3(0.004, 0.018, 0.026);
      float horizonGlow = exp(-abs(uv.y - horizon) * 8.5) * (1.0 - abs(uv.x - 0.5) * 0.72);
      sky += vec3(0.012, 0.10, 0.125) * max(horizonGlow, 0.0);
      float entryFoam = exp(-abs(uv.y - front) * 105.0) * (1.0 - smoothstep(0.66, 1.0, progress));
      water += vec3(0.50, 0.93, 0.95) * entryFoam * 0.82;

      vec3 color = mix(sky, water, reveal);
      float vignette = 1.0 - smoothstep(0.24, 1.08, length((uv - 0.5) * vec2(1.15, 0.82)));
      color *= 0.68 + vignette * 0.32;
      color += (hash(gl_FragCoord.xy + floor(time * 18.0)) - 0.5) * 0.012;
      gl_FragColor = vec4(color, 1.0);
    }
  `;

  const compileShader = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.warn("Ocean shader unavailable.");
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  };

  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) return;

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return;

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const uniforms = {
    resolution: gl.getUniformLocation(program, "u_resolution"),
    pointer: gl.getUniformLocation(program, "u_pointer"),
    time: gl.getUniformLocation(program, "u_time"),
    reduced: gl.getUniformLocation(program, "u_reduced"),
  };

  let width = 1;
  let height = 1;
  let frame = 0;
  let visible = true;
  let lastFrame = 0;

  const resize = () => {
    const rect = landing.getBoundingClientRect();
    const mobile = window.matchMedia("(max-width: 640px)").matches;
    const scale = Math.min(window.devicePixelRatio || 1, mobile ? 1 : 1.4);
    width = Math.max(1, Math.round(rect.width * scale));
    height = Math.max(1, Math.round(rect.height * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  const draw = (timestamp) => {
    if (!visible || document.hidden) {
      frame = 0;
      return;
    }
    if (!reducedMotion.matches && timestamp - lastFrame < 32) {
      frame = requestAnimationFrame(draw);
      return;
    }
    lastFrame = timestamp;
    pointerX += (targetX - pointerX) * 0.045;
    pointerY += (targetY - pointerY) * 0.045;
    gl.uniform2f(uniforms.resolution, width, height);
    gl.uniform2f(uniforms.pointer, pointerX, pointerY);
    gl.uniform1f(uniforms.time, (timestamp - startedAt) * 0.001);
    gl.uniform1f(uniforms.reduced, reducedMotion.matches ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    landing.classList.add("is-ocean-ready");
    frame = reducedMotion.matches ? 0 : requestAnimationFrame(draw);
  };

  const start = () => {
    if (!frame && visible && !document.hidden) frame = requestAnimationFrame(draw);
  };
  const stop = () => {
    if (frame) cancelAnimationFrame(frame);
    frame = 0;
  };

  bindPointer();
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => {
      resize();
      if (reducedMotion.matches) start();
    }).observe(landing);
  } else {
    window.addEventListener("resize", resize, { passive: true });
  }
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start(); else stop();
    }).observe(landing);
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop(); else start();
  });
  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    stop();
  });

  resize();
  start();
})();
