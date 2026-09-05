(() => {
  "use strict";

  const canvas = document.querySelector("[data-river-water]");
  const hero = canvas?.closest(".river-hero");
  if (!canvas || !hero) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const gl = canvas.getContext("webgl", {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    preserveDrawingBuffer: false,
    powerPreference: "low-power"
  });

  if (!gl) {
    hero.classList.add("is-static");
    return;
  }

  const vertexSource = `
    attribute vec2 a_position;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  const fragmentSource = `
    precision highp float;

    uniform vec2 u_resolution;
    uniform vec2 u_pointer;
    uniform float u_pointer_strength;
    uniform float u_time;
    uniform float u_intro;
    uniform float u_dark;

    float hash21(vec2 p) {
      p = fract(p * vec2(123.34, 456.21));
      p += dot(p, p + 45.32);
      return fract(p.x * p.y);
    }

    float valueNoise(vec2 p) {
      vec2 cell = floor(p);
      vec2 local = fract(p);
      local = local * local * (3.0 - 2.0 * local);
      float a = hash21(cell);
      float b = hash21(cell + vec2(1.0, 0.0));
      float c = hash21(cell + vec2(0.0, 1.0));
      float d = hash21(cell + vec2(1.0, 1.0));
      return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
    }

    float fbm(vec2 p) {
      float value = 0.0;
      float amplitude = 0.5;
      mat2 transform = mat2(1.61, 1.18, -1.18, 1.61);
      for (int octave = 0; octave < 4; octave++) {
        value += amplitude * valueNoise(p);
        p = transform * p + vec2(3.7, 1.9);
        amplitude *= 0.48;
      }
      return value;
    }

    vec2 curlField(vec2 p) {
      float epsilon = 0.055;
      float top = fbm(p + vec2(0.0, epsilon));
      float bottom = fbm(p - vec2(0.0, epsilon));
      float right = fbm(p + vec2(epsilon, 0.0));
      float left = fbm(p - vec2(epsilon, 0.0));
      return vec2(top - bottom, left - right) / (2.0 * epsilon);
    }

    float waterBody(vec2 p, float offset, float width, float seed) {
      float center = offset + (fbm(vec2(p.x * 0.78 + seed, seed * 1.7)) - 0.5) * 0.76;
      float breathing = mix(0.76, 1.16, fbm(vec2(p.x * 1.3 - seed, seed + 4.2)));
      float localWidth = width * breathing;
      return 1.0 - smoothstep(localWidth * 0.44, localWidth, abs(p.y - center));
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / u_resolution.xy;
      vec2 p = uv - 0.5;
      p.x *= u_resolution.x / u_resolution.y;

      vec2 pointer = u_pointer - 0.5;
      pointer.x *= u_resolution.x / u_resolution.y;
      vec2 pointerDelta = p - pointer;
      float pointerWeight = exp(-dot(pointerDelta, pointerDelta) * 9.0) * u_pointer_strength;
      p += vec2(-pointerDelta.y, pointerDelta.x) * pointerWeight * 0.18;

      float movingTime = u_time;
      vec2 drift = vec2(movingTime * 0.18, -movingTime * 0.07);
      vec2 curl = curlField(p * 1.08 + drift);
      vec2 warped = p + curl * 0.24;
      warped += (curlField(warped * 1.85 - drift * 0.7) * 0.1);

      float bodyA = waterBody(warped * mat2(0.99, -0.12, 0.12, 0.99), -0.08, 0.34, 2.4 + movingTime * 0.08);
      float bodyB = waterBody(warped * mat2(0.95, 0.30, -0.30, 0.95), 0.37, 0.24, 6.8 - movingTime * 0.05);
      float bodyC = waterBody(warped * mat2(0.97, -0.24, 0.24, 0.97), -0.48, 0.21, 10.2 + movingTime * 0.04);
      float water = max(bodyA, max(bodyB, bodyC));

      float textureLarge = fbm(warped * 2.2 + drift * 0.65);
      float textureFine = fbm(warped * 4.7 - drift * 0.38 + curl * 0.4);
      float translucency = 0.42 + textureLarge * 0.36 + textureFine * 0.22;

      float edgePosition = abs(uv.x - 0.5) * 2.0;
      float introEdge = 1.04 - smoothstep(0.0, 1.0, u_intro) * 1.2;
      float arrival = smoothstep(introEdge - 0.2, introEdge, edgePosition);
      float settled = smoothstep(0.28, 0.92, u_intro);
      float reveal = mix(arrival, 1.0, settled);

      vec3 lightTop = vec3(0.875, 0.957, 0.984);
      vec3 lightBottom = vec3(0.725, 0.890, 0.949);
      vec3 lightWater = vec3(0.205, 0.590, 0.735);
      vec3 lightDepth = vec3(0.075, 0.355, 0.510);
      vec3 darkTop = vec3(0.038, 0.118, 0.158);
      vec3 darkBottom = vec3(0.020, 0.075, 0.106);
      vec3 darkWater = vec3(0.110, 0.420, 0.545);
      vec3 darkDepth = vec3(0.055, 0.235, 0.330);

      vec3 background = mix(mix(lightBottom, lightTop, uv.y), mix(darkBottom, darkTop, uv.y), u_dark);
      vec3 waterColor = mix(mix(lightWater, lightDepth, textureLarge), mix(darkWater, darkDepth, textureLarge), u_dark);
      vec3 waterColorSoft = mix(mix(lightTop, lightWater, 0.58), mix(darkTop, darkWater, 0.65), u_dark);
      vec3 color = background;
      color = mix(color, waterColor, bodyA * translucency * reveal * mix(0.54, 0.62, u_dark));
      color = mix(color, waterColorSoft, bodyB * (0.30 + textureFine * 0.22) * reveal);
      color = mix(color, waterColor, bodyC * (0.24 + textureLarge * 0.24) * reveal);

      float softRefraction = smoothstep(0.57, 0.88, textureFine) * water * reveal * mix(0.16, 0.1, u_dark);
      color = mix(color, mix(vec3(0.94, 0.985, 1.0), vec3(0.30, 0.63, 0.72), u_dark), softRefraction);

      vec2 clearShape = p * vec2(0.72, 1.38);
      float clearPool = exp(-dot(clearShape, clearShape) * 3.2);
      color = mix(color, background, clearPool * mix(0.12, 0.09, u_dark));

      float vignette = smoothstep(1.1, 0.2, length((uv - 0.5) * vec2(0.72, 1.0)));
      color += vignette * mix(0.018, 0.008, u_dark);
      gl_FragColor = vec4(color, 1.0);
    }
  `;

  function compile(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = compile(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compile(gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) {
    hero.classList.add("is-static");
    return;
  }

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    hero.classList.add("is-static");
    return;
  }

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
    pointerStrength: gl.getUniformLocation(program, "u_pointer_strength"),
    time: gl.getUniformLocation(program, "u_time"),
    intro: gl.getUniformLocation(program, "u_intro"),
    dark: gl.getUniformLocation(program, "u_dark")
  };

  const pointer = { x: 0.5, y: 0.5, strength: 0 };
  const compact = window.matchMedia("(max-width: 720px)");
  let frame = 0;
  let previousFrame = 0;
  let startedAt = performance.now();
  let visible = true;

  function resize() {
    const width = Math.max(1, hero.clientWidth);
    const height = Math.max(1, hero.clientHeight);
    const requestedDpr = compact.matches ? 1 : Math.min(window.devicePixelRatio || 1, 1.35);
    const maxPixels = compact.matches ? 600000 : 1200000;
    const pixelScale = Math.min(requestedDpr, Math.sqrt(maxPixels / (width * height)));
    const renderWidth = Math.max(1, Math.round(width * pixelScale));
    const renderHeight = Math.max(1, Math.round(height * pixelScale));
    if (canvas.width !== renderWidth || canvas.height !== renderHeight) {
      canvas.width = renderWidth;
      canvas.height = renderHeight;
      gl.viewport(0, 0, renderWidth, renderHeight);
    }
  }

  function draw(now, forceSettled = false) {
    resize();
    const elapsed = Math.max(0, now - startedAt);
    const intro = forceSettled ? 1 : Math.min(1, elapsed / 2700);
    const time = Math.min(elapsed, 3200) * 0.00042 + Math.max(0, elapsed - 3200) * 0.000055;
    const dark = document.documentElement.dataset.theme === "dark" ? 1 : 0;

    pointer.strength *= 0.94;
    gl.uniform2f(uniforms.resolution, canvas.width, canvas.height);
    gl.uniform2f(uniforms.pointer, pointer.x, pointer.y);
    gl.uniform1f(uniforms.pointerStrength, pointer.strength);
    gl.uniform1f(uniforms.time, forceSettled ? 1.7 : time);
    gl.uniform1f(uniforms.intro, intro);
    gl.uniform1f(uniforms.dark, dark);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    hero.classList.add("is-rendered");
  }

  function schedule() {
    if (!frame && visible && !document.hidden && !reducedMotion.matches) {
      frame = requestAnimationFrame(loop);
    }
  }

  function loop(now) {
    frame = 0;
    const elapsed = now - startedAt;
    const entering = elapsed < 3200;
    const fps = entering ? (compact.matches ? 24 : 30) : (compact.matches ? 12 : 14);
    if (now - previousFrame >= 1000 / fps) {
      draw(now);
      previousFrame = now;
    }
    schedule();
  }

  hero.addEventListener("pointermove", (event) => {
    const bounds = hero.getBoundingClientRect();
    pointer.x = (event.clientX - bounds.left) / bounds.width;
    pointer.y = 1 - (event.clientY - bounds.top) / bounds.height;
    pointer.strength = 1;
  }, { passive: true });

  hero.addEventListener("pointerleave", () => {
    pointer.strength = 0;
  });

  const observer = new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    schedule();
  }, { threshold: 0.02 });
  observer.observe(hero);

  const themeObserver = new MutationObserver(() => {
    draw(performance.now(), reducedMotion.matches);
  });
  themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  document.addEventListener("visibilitychange", schedule);
  window.addEventListener("resize", () => {
    draw(performance.now(), reducedMotion.matches);
    schedule();
  }, { passive: true });
  reducedMotion.addEventListener?.("change", () => {
    startedAt = performance.now() - 2700;
    draw(performance.now(), reducedMotion.matches);
    schedule();
  });

  draw(performance.now(), reducedMotion.matches);
  schedule();
})();
