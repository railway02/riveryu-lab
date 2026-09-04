(() => {
  "use strict";

  const canvas = document.querySelector("[data-ocean-current]");
  const landing = canvas?.closest(".home-landing");
  if (!canvas || !landing) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const startCanvasFallback = () => {
    const context = canvas.getContext("2d");
    if (!context) return;

    let cssWidth = 1;
    let cssHeight = 1;
    let frame = 0;
    let visible = true;
    let lastFrame = 0;
    let pointerX = 0.5;
    let pointerY = 0.5;
    let targetX = 0.5;
    let targetY = 0.5;

    const resize = () => {
      const rect = landing.getBoundingClientRect();
      const scale = Math.min(window.devicePixelRatio || 1, 1.25);
      cssWidth = Math.max(1, rect.width);
      cssHeight = Math.max(1, rect.height);
      canvas.width = Math.round(cssWidth * scale);
      canvas.height = Math.round(cssHeight * scale);
      context.setTransform(scale, 0, 0, scale, 0, 0);
    };

    const waveY = (x, index, time) => {
      const base = cssHeight * (0.2 + index / 24 * 0.6);
      const primary = Math.sin(x * 0.0065 + time * 0.00036 + index * 0.57) * cssHeight * 0.018;
      const secondary = Math.sin(x * 0.0021 - time * 0.00021 + index * 0.83) * cssHeight * 0.025;
      const dx = (x / cssWidth - pointerX) * 4.4;
      const wake = Math.exp(-(dx * dx)) * (pointerY - base / cssHeight) * cssHeight * 0.12;
      return base + primary + secondary + wake;
    };

    const draw = (time) => {
      if (!visible || document.hidden) {
        frame = 0;
        return;
      }
      if (!reducedMotion.matches && time - lastFrame < 32) {
        frame = requestAnimationFrame(draw);
        return;
      }

      lastFrame = time;
      pointerX += (targetX - pointerX) * 0.045;
      pointerY += (targetY - pointerY) * 0.045;
      context.clearRect(0, 0, cssWidth, cssHeight);

      const dark = document.documentElement.dataset.theme === "dark";
      const gradient = context.createLinearGradient(cssWidth * 0.12, 0, cssWidth * 0.88, 0);
      gradient.addColorStop(0, dark ? "rgba(74, 205, 214, .16)" : "rgba(34, 120, 139, .11)");
      gradient.addColorStop(0.52, dark ? "rgba(126, 139, 234, .18)" : "rgba(78, 102, 177, .105)");
      gradient.addColorStop(1, dark ? "rgba(215, 114, 180, .13)" : "rgba(163, 86, 139, .09)");

      const animatedTime = reducedMotion.matches ? 6000 : time;
      for (let index = 0; index < 25; index += 1) {
        context.beginPath();
        for (let x = -24; x <= cssWidth + 24; x += 18) {
          const y = waveY(x, index, animatedTime);
          if (x === -24) context.moveTo(x, y); else context.lineTo(x, y);
        }
        context.strokeStyle = gradient;
        context.lineWidth = index % 5 === 0 ? 1.15 : 0.68;
        context.stroke();
      }

      for (let index = 0; index < 18; index += 1) {
        const progress = ((animatedTime * (0.013 + index % 3 * 0.003) + index * 83) % (cssWidth + 140)) - 70;
        const line = 3 + index % 19;
        const y = waveY(progress, line, animatedTime);
        context.beginPath();
        context.arc(progress, y, index % 4 === 0 ? 1.6 : 0.9, 0, Math.PI * 2);
        context.fillStyle = dark ? "rgba(146, 225, 232, .24)" : "rgba(45, 118, 147, .16)";
        context.fill();
      }

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

    landing.addEventListener("pointermove", (event) => {
      const rect = landing.getBoundingClientRect();
      targetX = (event.clientX - rect.left) / rect.width;
      targetY = (event.clientY - rect.top) / rect.height;
    }, { passive: true });
    landing.addEventListener("pointerleave", () => {
      targetX = 0.5;
      targetY = 0.5;
    }, { passive: true });

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

    const handleMotionPreference = () => {
      stop();
      start();
    };
    if ("addEventListener" in reducedMotion) {
      reducedMotion.addEventListener("change", handleMotionPreference);
    } else {
      reducedMotion.addListener(handleMotionPreference);
    }
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop(); else start();
    });

    resize();
    start();
  };

  const gl = canvas.getContext("webgl", {
    alpha: true,
    antialias: false,
    depth: false,
    powerPreference: "low-power",
    premultipliedAlpha: true,
  });

  if (!gl) {
    startCanvasFallback();
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
    uniform float u_dark;

    float hash(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    float noise(vec2 p) {
      vec2 i = floor(p);
      vec2 f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x),
        f.y
      );
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

    void main() {
      float aspect = u_resolution.x / max(u_resolution.y, 1.0);
      vec2 p = (v_uv - 0.5) * vec2(aspect, 1.0);
      vec2 pointer = (u_pointer - 0.5) * vec2(aspect, 1.0);
      vec2 delta = p - pointer;
      float distanceToPointer = dot(delta, delta) + 0.12;

      vec2 swirl = vec2(-delta.y, delta.x) * (0.026 / distanceToPointer);
      float baseNoise = fbm(p * 1.38 + vec2(u_time * 0.025, -u_time * 0.018));
      vec2 q = p + swirl + vec2(baseNoise * 0.085, -baseNoise * 0.045);

      float currentA = abs(sin(
        q.y * 12.0
        + sin(q.x * 3.1 - u_time * 0.24) * 1.22
        + baseNoise * 3.0
        - u_time * 0.46
      ));
      float currentB = abs(sin(
        (q.y * 0.76 - q.x * 0.19) * 17.0
        + sin(q.x * 4.3 + u_time * 0.17) * 0.82
        + u_time * 0.31
      ));
      float currentC = abs(sin(
        (q.y * 0.42 + q.x * 0.34) * 23.0
        - baseNoise * 2.4
        - u_time * 0.22
      ));

      float ribbonA = 1.0 - smoothstep(0.05, 0.29, currentA);
      float ribbonB = 1.0 - smoothstep(0.03, 0.20, currentB);
      float caustic = 1.0 - smoothstep(0.025, 0.135, currentC);

      float breathing = 0.5 + 0.5 * sin(u_time * 0.16 + baseNoise * 5.0);
      float mist = smoothstep(0.28, 0.82, baseNoise) * (0.56 + breathing * 0.22);
      float centerFade = 1.0 - smoothstep(0.16, 1.05, length(p * vec2(0.72, 1.0)));
      float edgeFade = 1.0 - smoothstep(0.66, 1.22, length(p * vec2(0.54, 1.0)));
      float pointerWake = exp(-distanceToPointer * 5.0) * 0.14;

      vec3 teal = mix(vec3(0.05, 0.31, 0.40), vec3(0.22, 0.77, 0.82), u_dark);
      vec3 blue = mix(vec3(0.18, 0.29, 0.58), vec3(0.44, 0.52, 0.92), u_dark);
      vec3 violet = mix(vec3(0.45, 0.25, 0.50), vec3(0.80, 0.45, 0.72), u_dark);

      vec3 color = mix(teal, blue, clamp(v_uv.x + baseNoise * 0.22, 0.0, 1.0));
      color = mix(color, violet, ribbonB * (0.18 + v_uv.x * 0.18));
      color += caustic * mix(vec3(0.12, 0.20, 0.22), vec3(0.15, 0.25, 0.30), u_dark);

      float alpha = (
        ribbonA * 0.19
        + ribbonB * 0.105
        + caustic * 0.075
        + mist * 0.075
        + pointerWake
      ) * edgeFade * (0.52 + centerFade * 0.48);

      gl_FragColor = vec4(color * alpha, alpha);
    }
  `;

  const compileShader = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
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
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
    gl.STATIC_DRAW
  );

  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

  const uniforms = {
    resolution: gl.getUniformLocation(program, "u_resolution"),
    pointer: gl.getUniformLocation(program, "u_pointer"),
    time: gl.getUniformLocation(program, "u_time"),
    dark: gl.getUniformLocation(program, "u_dark"),
  };

  let width = 1;
  let height = 1;
  let frame = 0;
  let visible = true;
  let lastFrame = 0;
  let pointerX = 0.5;
  let pointerY = 0.5;
  let targetX = 0.5;
  let targetY = 0.5;

  const resize = () => {
    const rect = landing.getBoundingClientRect();
    const mobileScale = window.matchMedia("(max-width: 640px)").matches ? 1 : 1.35;
    const scale = Math.min(window.devicePixelRatio || 1, mobileScale);
    width = Math.max(1, Math.round(rect.width * scale));
    height = Math.max(1, Math.round(rect.height * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
  };

  const isDark = () => document.documentElement.dataset.theme === "dark";

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
    gl.uniform1f(uniforms.time, reducedMotion.matches ? 6.0 : timestamp * 0.001);
    gl.uniform1f(uniforms.dark, isDark() ? 1 : 0);
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

  landing.addEventListener("pointermove", (event) => {
    const rect = landing.getBoundingClientRect();
    targetX = (event.clientX - rect.left) / rect.width;
    targetY = 1 - (event.clientY - rect.top) / rect.height;
  }, { passive: true });

  landing.addEventListener("pointerleave", () => {
    targetX = 0.5;
    targetY = 0.5;
  }, { passive: true });

  if ("ResizeObserver" in window) {
    const resizeObserver = new ResizeObserver(() => {
      resize();
      if (reducedMotion.matches) start();
    });
    resizeObserver.observe(landing);
  } else {
    window.addEventListener("resize", resize, { passive: true });
  }

  if ("IntersectionObserver" in window) {
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start(); else stop();
    });
    intersectionObserver.observe(landing);
  }

  const themeObserver = new MutationObserver(() => {
    if (reducedMotion.matches) start();
  });
  themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  const handleMotionPreference = () => {
    stop();
    start();
  };
  if ("addEventListener" in reducedMotion) {
    reducedMotion.addEventListener("change", handleMotionPreference);
  } else {
    reducedMotion.addListener(handleMotionPreference);
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop(); else start();
  });

  canvas.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    stop();
    landing.classList.remove("is-ocean-ready");
  });

  resize();
  start();
})();
