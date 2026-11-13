/* Survey sheets: overprint layers that drift apart, floating points, a live route to Sheet 2,
   a survey cursor with grid square and ground height, a scale bar, and the journal globe. */
(function () {
  'use strict';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fine = window.matchMedia('(pointer: fine)').matches;
  var wide = window.matchMedia('(min-width: 961px)');
  var neat = document.querySelector('#sheet .neatline');
  if (!neat) return;

  /* ---------- Elevation grid (Copernicus DEM, 200 x 125) ---------- */
  var dem = null;
  fetch(window.SHEET.dem).then(function (r) { return r.json(); }).then(function (d) {
    var bin = atob(d.z), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    dem = { w: d.w, h: d.h, vw: d.vw, vh: d.vh, kmw: d.km_w, z: new Int16Array(buf) };
    labelPoints(); scaleBar();
  }).catch(function () {});

  function fit(W, H) { var s = Math.max(W / 1600, H / 1000); return { s: s, ox: (W - 1600 * s) / 2, oy: (H - 1000 * s) / 2 }; }
  function heightAt(px, py, W, H) {          /* sheet pixel -> metres, honouring the SVG "slice" fit */
    if (!dem) return null;
    var f = fit(W, H), gx = ((px - f.ox) / f.s) / dem.vw * (dem.w - 1), gy = ((py - f.oy) / f.s) / dem.vh * (dem.h - 1);
    if (gx < 0 || gy < 0 || gx > dem.w - 1 || gy > dem.h - 1) return null;
    var x0 = Math.floor(gx), y0 = Math.floor(gy), x1 = Math.min(x0 + 1, dem.w - 1), y1 = Math.min(y0 + 1, dem.h - 1), fx = gx - x0, fy = gy - y0;
    var z = function (x, y) { return dem.z[y * dem.w + x]; };
    return z(x0, y0) * (1 - fx) * (1 - fy) + z(x1, y0) * fx * (1 - fy) + z(x0, y1) * (1 - fx) * fy + z(x1, y1) * fx * fy;
  }
  function gridRef(px, py, W, H) { return 'ABCDEFGH'.charAt(Math.min(7, Math.floor(px / W * 8))) + (Math.min(5, Math.floor(py / H * 6)) + 1); }
  function metres(z) { return Math.round(z).toLocaleString('en-GB') + ' m'; }

  var pts = Array.prototype.slice.call(neat.querySelectorAll('.pt'));
  function labelPoints() {
    var r = neat.getBoundingClientRect();
    pts.forEach(function (p) {
      var z = heightAt(+p.dataset.x / 100 * r.width, +p.dataset.y / 100 * r.height, r.width, r.height);
      var e = p.querySelector('.elev'); if (e && z != null) e.textContent = metres(z);
    });
  }
  var face = neat.querySelector('.contours');
  function scaleBar() {                        /* a 5 km bar at the drawn scale of the map face */
    if (!dem) return;
    var r = (face || neat).getBoundingClientRect(), f = fit(r.width, r.height), pxPerKm = 1600 * f.s / dem.kmw;
    var km = [10, 5, 2, 1].filter(function (k) { return k * pxPerKm <= 220; })[0] || 1;
    var sc = document.querySelector('.scale'); if (sc) { sc.style.setProperty('--bar', Math.round(km * pxPerKm) + 'px'); var e = sc.querySelector('.scale-end'); if (e) e.textContent = km + ' km'; }
  }

  /* ---------- Motion model: layer drift, point float, magnetic lean, collision guard ---------- */
  var layers = Array.prototype.slice.call(neat.querySelectorAll('.layer')).map(function (el) {
    return { el: el, d: +el.dataset.depth || 0, x: 0, y: 0, tx: 0, ty: 0 };
  });
  var POINT_DEPTH = 1.15;
  var st = pts.map(function (p, i) { return { el: p, i: i, mark: p.querySelector('.mark'), px: 0, py: 0, tx: 0, ty: 0, gx: 0, gy: 0, ph: i * 1.7 }; });
  var boxes = [neat.querySelector('.legend'), neat.querySelector('.titleblock')];
  var mx = -1e4, my = -1e4, nx = 0, ny = 0, lx = 0, ly = 0, ltx = 0, lty = 0, t0 = performance.now(), visible = true;

  function guard() {                           /* move any point whose marker or label lands under the legend or title block, by the shortest way out */
    var r = neat.getBoundingClientRect();
    st.forEach(function (s) {
      var b = s.el.getBoundingClientRect(), cur = { x: s.ox || 0, y: s.oy || 0 };
      var base = { l: b.left - cur.x, r: b.right - cur.x, t: b.top - cur.y, btm: b.bottom - cur.y };
      s.gx = s.gy = 0;
      boxes.forEach(function (bx) {
        if (!bx) return; var q = bx.getBoundingClientRect(), m = 14;
        var L = base.l + s.gx, R = base.r + s.gx, T = base.t + s.gy, B = base.btm + s.gy;
        if (R < q.left - m || L > q.right + m || B < q.top - m || T > q.bottom + m) return;
        var moves = [[0, q.top - m - B], [0, q.bottom + m - T], [q.left - m - R, 0], [q.right + m - L, 0]].filter(function (v) {
          return L + v[0] > r.left + 8 && R + v[0] < r.right - 8 && T + v[1] > r.top + 8 && B + v[1] < r.bottom - 8;
        });
        if (!moves.length) return;
        moves.sort(function (a, c) { return Math.abs(a[0]) + Math.abs(a[1]) - Math.abs(c[0]) - Math.abs(c[1]); });
        s.gx += moves[0][0]; s.gy += moves[0][1];
      });
    });
  }

  function frame(now) {
    var r = neat.getBoundingClientRect(), t = (now - t0) / 1000, sy = Math.max(0, -r.top);
    ltx = reduce ? 0 : -nx * 22; lty = reduce ? 0 : -ny * 15 - sy * 0.16;   /* no ambient motion under reduced motion */ lx += (ltx - lx) * 0.07; ly += (lty - ly) * 0.07;
    layers.forEach(function (L) { L.el.style.transform = 'translate(' + (lx * L.d).toFixed(2) + 'px,' + (ly * L.d).toFixed(2) + 'px)'; });
    st.forEach(function (s) {
      var fx = reduce ? 0 : 10 * Math.sin(t * (0.55 + s.i * 0.07) + s.ph), fy = reduce ? 0 : 8 * Math.cos(t * (0.42 + s.i * 0.05) + s.ph * 1.3);
      var cx = +s.el.dataset.x / 100 * r.width + r.left + s.gx + fx + lx * POINT_DEPTH, cy = +s.el.dataset.y / 100 * r.height + r.top + s.gy + fy + ly * POINT_DEPTH;
      var dx = mx - cx, dy = my - cy, dist = Math.sqrt(dx * dx + dy * dy), k = !reduce && dist < 170 ? (1 - dist / 170) * 0.2 : 0;
      s.tx = Math.max(-18, Math.min(18, dx * k)); s.ty = Math.max(-18, Math.min(18, dy * k));
      s.px += (s.tx - s.px) * 0.12; s.py += (s.ty - s.py) * 0.12;
      var ox = s.gx + fx + lx * POINT_DEPTH + s.px, oy = s.gy + fy + ly * POINT_DEPTH + s.py;
      s.el.style.setProperty('--mx', ox.toFixed(2) + 'px'); s.el.style.setProperty('--my', oy.toFixed(2) + 'px');
      s.ox = ox; s.oy = oy;
    });
    if (visible && wide.matches && !reduce) requestAnimationFrame(frame);
    else running = false;
  }
  var running = false;
  function start() { if (!running && wide.matches) { running = true; requestAnimationFrame(frame); } }
  function settle() { guard(); labelPoints(); scaleBar(); if (reduce || !wide.matches) { requestAnimationFrame(function (n) { frame(n); }); } else start(); }
  if ('IntersectionObserver' in window) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (visible) start(); }).observe(neat);
  window.addEventListener('resize', settle);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(settle); else settle();
  settle();
  window.addEventListener('scroll', function () { if (reduce) requestAnimationFrame(frame); }, { passive: true });

  /* ---------- Survey cursor: hairline reticle with grid square and ground height ---------- */
  var readout = neat.querySelector('.readout');
  if (fine) {
    neat.addEventListener('pointermove', function (e) {
      var r = neat.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      mx = e.clientX; my = e.clientY; nx = (px / r.width - 0.5) * 2; ny = (py / r.height - 0.5) * 2;
      neat.classList.add('hot');
      neat.style.setProperty('--cx', px + 'px'); neat.style.setProperty('--cy', py + 'px');
      var z = heightAt(px - lx * 0.45, py - ly * 0.45, r.width, r.height);
      readout.textContent = gridRef(px, py, r.width, r.height) + (z != null ? ' · ' + metres(z) : '');
      if (reduce) requestAnimationFrame(frame);
    });
    neat.addEventListener('pointerleave', function () { neat.classList.remove('hot'); mx = my = -1e4; nx = ny = 0; });
  }

  /* ---------- Sheet 2 globe: wine-ink graticule, coastlines, the route and a plane flying it ---------- */
  var canvas = document.getElementById('globe');
  if (!canvas || !('IntersectionObserver' in window)) return;
  var io = new IntersectionObserver(function (es) {
    if (es.some(function (e) { return e.isIntersecting; })) { io.disconnect(); loadGlobe(); }
  }, { rootMargin: '300px' });
  io.observe(canvas);

  function script(src) { return new Promise(function (ok, no) { var s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = no; document.head.appendChild(s); }); }
  function loadGlobe() {
    Promise.all([
      script('https://unpkg.com/three@0.160.0/build/three.min.js'),
      script('https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js'),
      fetch('https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json').then(function (r) { return r.json(); })
    ]).then(function (res) { drawGlobe(res[2]); }).catch(function () { canvas.closest('figure').classList.add('no-globe'); });
  }

  function drawGlobe(topo) {
    var T = window.THREE, route = window.SHEET.route;
    var renderer = new T.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    var scene = new T.Scene(), camera = new T.PerspectiveCamera(36, 1, 0.1, 100);
    camera.position.set(0, 0, 4.3);
    var world = new T.Group(); scene.add(world);
    var WINE = 0x8A2A3D, DEEP = 0x5B1426, PAPER = 0xEDEFEC;
    function v(lat, lng, r) {
      var phi = (90 - lat) * Math.PI / 180, th = (lng + 180) * Math.PI / 180;
      return new T.Vector3(-r * Math.sin(phi) * Math.cos(th), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(th));
    }
    function line(points, color, opacity) {
      return new T.Line(new T.BufferGeometry().setFromPoints(points), new T.LineBasicMaterial({ color: color, transparent: true, opacity: opacity }));
    }
    world.add(new T.Mesh(new T.SphereGeometry(0.995, 64, 48), new T.MeshBasicMaterial({ color: PAPER })));
    for (var la = -60; la <= 60; la += 20) { var a = []; for (var lo = -180; lo <= 180; lo += 4) a.push(v(la, lo, 1)); world.add(line(a, WINE, 0.22)); }
    for (var lo2 = -180; lo2 < 180; lo2 += 20) { var b = []; for (var la2 = -90; la2 <= 90; la2 += 4) b.push(v(la2, lo2, 1)); world.add(line(b, WINE, 0.22)); }
    window.topojson.mesh(topo, topo.objects.land).coordinates.forEach(function (ring) { world.add(line(ring.map(function (c) { return v(c[1], c[0], 1.002); }), DEEP, 0.85)); });
    var arcs = [];
    route.forEach(function (c, i) {
      var dot = new T.Mesh(new T.SphereGeometry(0.022, 12, 12), new T.MeshBasicMaterial({ color: WINE }));
      dot.position.copy(v(c.lat, c.lng, 1.004)); world.add(dot);
      if (i < route.length - 1) {
        var p0 = v(c.lat, c.lng, 1), p1 = v(route[i + 1].lat, route[i + 1].lng, 1), pts3 = [];
        for (var t = 0; t <= 1.0001; t += 1 / 64) { var p = new T.Vector3().copy(p0).lerp(p1, t).normalize(); pts3.push(p.multiplyScalar(1 + 0.15 * Math.sin(Math.PI * t))); }
        world.add(line(pts3, WINE, 0.9)); arcs.push(pts3);
      }
    });
    var plane = new T.Mesh(new T.ConeGeometry(0.03, 0.09, 3), new T.MeshBasicMaterial({ color: DEEP }));
    world.add(plane);

    function size() { var w = canvas.clientWidth; renderer.setSize(w, w, false); camera.aspect = 1; camera.updateProjectionMatrix(); }
    size(); window.addEventListener('resize', size);

    /* start with the route facing the viewer: longitude 100° E, a little north */
    var rotY = Math.PI / 2 - (100 + 180) * Math.PI / 180, rotX = 0.12, vy = reduce ? 0 : 0.0012, vx = 0, drag = false, px0 = 0, py0 = 0;
    canvas.addEventListener('pointerdown', function (e) { drag = true; px0 = e.clientX; py0 = e.clientY; vy = vx = 0; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', function (e) { if (!drag) return; vy = (e.clientX - px0) * 0.005; vx = (e.clientY - py0) * 0.005; rotY += vy; rotX = Math.max(-1.1, Math.min(1.1, rotX + vx)); px0 = e.clientX; py0 = e.clientY; });
    canvas.addEventListener('pointerup', function () { drag = false; });
    canvas.addEventListener('pointercancel', function () { drag = false; });

    var leg = 0, tt = 0;
    function place() {
      var a = arcs[leg], f = tt * (a.length - 1), k = Math.floor(f), p = a[k], q = a[Math.min(k + 1, a.length - 1)];
      plane.position.copy(p).lerp(q, f - k);
      plane.lookAt(new T.Vector3().copy(q).add(new T.Vector3().copy(q).sub(p)));
      plane.rotateX(Math.PI / 2);
    }
    if (reduce) { leg = arcs.length - 1; tt = 1; }
    (function loop() {
      if (!drag && !reduce) { vy += (0.0012 - vy) * 0.01; vx *= 0.9; rotY += vy; rotX = Math.max(-1.1, Math.min(1.1, rotX + vx)); }
      world.rotation.set(rotX, rotY, 0);
      if (!reduce) { tt += 0.004; if (tt >= 1) { tt = 0; leg = (leg + 1) % arcs.length; } }
      place(); renderer.render(scene, camera);
      if (!reduce || drag) requestAnimationFrame(loop); else setTimeout(function () { requestAnimationFrame(loop); }, 250);
    })();
  }
})();
