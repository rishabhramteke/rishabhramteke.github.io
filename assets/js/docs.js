/* Documents on project pages, drawn in the page instead of downloaded.
   PDFs: PDF.js draws each page onto a canvas. Landscape files (slide decks) show one slide at a time with
   previous/next, arrow keys and swipe; portrait files (reports) show as a stack of pages drawn as they scroll in.
   Other files (code) are fetched and shown as numbered lines. */
(function () {
  'use strict';
  var ROOT = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/';
  var BUILD = ROOT + 'legacy/build/';
  var each = function (list, fn) { Array.prototype.forEach.call(list, fn); };
  var el = function (tag, cls, attrs) { var e = document.createElement(tag); if (cls) e.className = cls; for (var k in attrs || {}) e.setAttribute(k, attrs[k]); return e; };
  var DPR = function () { return Math.min(window.devicePixelRatio || 1, 2.5); };
  var CHEV = { prev: 'M10 3 L5 8 L10 13', next: 'M6 3 L11 8 L6 13' };
  function chevron(dir) { return '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="' + CHEV[dir] + '"/></svg>'; }

  /* ---------- code files ---------- */
  each(document.querySelectorAll('.code-file'), function (pre) {
    var code = pre.querySelector('code'); code.textContent = 'Loading the file…';
    fetch(pre.getAttribute('data-src')).then(function (r) { if (!r.ok) throw new Error(r.status); return r.text(); }).then(function (t) {
      code.textContent = '';
      t.replace(/^(?:[ \t]*\r?\n)+/, '').replace(/\s+$/, '').split(/\r?\n/).forEach(function (line) {
        var s = el('span', 'ln'); s.textContent = line; code.appendChild(s); code.appendChild(document.createTextNode('\n'));
      });
    }).catch(function () { code.textContent = 'This file could not be loaded. Try reloading the page.'; });
  });

  /* ---------- PDFs ---------- */
  var pdfs = document.querySelectorAll('.pdf');
  if (!pdfs.length) return;
  each(pdfs, function (box) { box.querySelector('.pdf-status').textContent = 'Drawing the pages…'; });
  var script = el('script'); script.src = BUILD + 'pdf.min.js';
  script.onload = function () {
    var lib = window.pdfjsLib;
    if (!lib) return each(pdfs, fail);
    lib.GlobalWorkerOptions.workerSrc = BUILD + 'pdf.worker.min.js';
    each(pdfs, function (box) { open(lib, box); });
  };
  script.onerror = function () { each(pdfs, fail); };
  document.head.appendChild(script);

  function fail(box) {
    var st = box.querySelector('.pdf-status') || box.appendChild(el('p', 'pdf-status'));
    st.textContent = 'The pages could not be drawn here. Use Open PDF above to read them.';
    box.classList.add('pdf-failed');
  }

  function open(lib, box) {
    lib.getDocument({ url: box.getAttribute('data-src'), cMapUrl: ROOT + 'cmaps/', cMapPacked: true, standardFontDataUrl: ROOT + 'standard_fonts/' }).promise
      .then(function (pdf) {
        return pdf.getPage(1).then(function (p1) {
          var vp = p1.getViewport({ scale: 1 }), title = box.getAttribute('data-title') || 'Document';
          box.querySelector('.pdf-status').remove();
          var isDeck = vp.width > vp.height, fig = box.closest('.docfile');
          if (fig) fig.classList.toggle('is-deck', isDeck);
          (isDeck ? deck : stack)(box, pdf, vp.width / vp.height, title);
        });
      })
      .catch(function () { fail(box); });
  }

  /* draw page n of pdf at css width w into a fresh canvas; resolves with the canvas */
  function render(pdf, n, w) {
    return pdf.getPage(n).then(function (page) {
      var vp1 = page.getViewport({ scale: 1 }), vp = page.getViewport({ scale: w * DPR() / vp1.width });
      var c = el('canvas'); c.width = Math.floor(vp.width); c.height = Math.floor(vp.height);
      return page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise.then(function () { return c; });
    });
  }
  function blit(dst, src) { dst.width = src.width; dst.height = src.height; dst.getContext('2d').drawImage(src, 0, 0); }

  /* reports: a stack of pages, each drawn when it nears the viewport and redrawn if the column width changes */
  function stack(box, pdf, ratio, title) {
    var N = pdf.numPages, list = el('ol', 'pdf-stack'), pages = [];
    for (var i = 1; i <= N; i++) {
      var li = el('li', 'pdf-page'), sheet = el('div', 'pdf-sheet'), cv = el('canvas', '', { role: 'img', 'aria-label': title + ', page ' + i + ' of ' + N });
      sheet.style.aspectRatio = String(ratio); sheet.appendChild(cv); li.appendChild(sheet);
      if (N > 1) { var no = el('span', 'pdf-no'); no.textContent = i + ' / ' + N; li.appendChild(no); }
      list.appendChild(li); pages.push({ n: i, sheet: sheet, cv: cv, drawnAt: 0, busy: false, seen: false });
    }
    box.appendChild(list);
    function draw(p) {
      var w = p.sheet.clientWidth; if (!w || p.busy || Math.abs(w - p.drawnAt) < 2) return;
      p.busy = true;
      render(pdf, p.n, w).then(function (c) { blit(p.cv, c); p.drawnAt = w; p.sheet.classList.add('is-drawn'); })
        .catch(function () {}).then(function () { p.busy = false; if (p.seen && Math.abs(p.sheet.clientWidth - p.drawnAt) >= 2) draw(p); });
    }
    if (!('IntersectionObserver' in window)) { pages.forEach(draw); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { var p = pages[+e.target.getAttribute('data-i')]; p.seen = e.isIntersecting; if (p.seen) draw(p); });
    }, { rootMargin: '800px 0px' });
    pages.forEach(function (p, k) { p.sheet.setAttribute('data-i', k); io.observe(p.sheet); });
    if (window.ResizeObserver) {
      var lastW = list.clientWidth, t;
      new ResizeObserver(function () {
        if (Math.abs(list.clientWidth - lastW) < 2) return; lastW = list.clientWidth;
        clearTimeout(t); t = setTimeout(function () { pages.forEach(function (p) { if (p.seen) draw(p); }); }, 150);
      }).observe(list);
    }
  }

  /* slide decks: one slide at a time */
  function deck(box, pdf, ratio, title) {
    var N = pdf.numPages, cur = 1, cache = {}, order = [];
    var wrap = el('div', 'pdf-deck');
    var stage = el('div', 'pdf-stage', { tabindex: '0', role: 'group', 'aria-roledescription': 'slide deck', 'aria-label': title + '. Use the arrow keys to change slide.' });
    var cv = el('canvas', '', { role: 'img' }); stage.style.aspectRatio = String(ratio); stage.appendChild(cv);
    var ctrl = el('div', 'pdf-ctrl');
    var prev = el('button', 'pdf-btn', { type: 'button', 'aria-label': 'Previous slide' }); prev.innerHTML = chevron('prev');
    var next = el('button', 'pdf-btn', { type: 'button', 'aria-label': 'Next slide' }); next.innerHTML = chevron('next');
    var count = el('span', 'pdf-count', { 'aria-live': 'polite' });
    ctrl.appendChild(prev); ctrl.appendChild(count); ctrl.appendChild(next);
    wrap.appendChild(stage); wrap.appendChild(ctrl); box.appendChild(wrap);

    function get(n, w) {
      var key = n + '@' + w;
      if (!cache[key]) {
        cache[key] = render(pdf, n, w); order.push(key);
        while (order.length > 5) delete cache[order.shift()];
      }
      return cache[key];
    }
    function show(n) {
      cur = Math.max(1, Math.min(N, n));
      prev.disabled = cur === 1; next.disabled = cur === N;
      count.textContent = cur + ' / ' + N;
      cv.setAttribute('aria-label', title + ', slide ' + cur + ' of ' + N);
      var w = Math.round(stage.clientWidth), want = cur;
      if (!w) return;
      stage.classList.add('is-loading');
      get(cur, w).then(function (c) { if (want === cur) { blit(cv, c); stage.classList.remove('is-loading'); stage.classList.add('is-drawn'); } })
        .catch(function () { if (want === cur) stage.classList.remove('is-loading'); });
      if (cur < N) get(cur + 1, w).catch(function () {});
    }
    prev.addEventListener('click', function () { show(cur - 1); });
    next.addEventListener('click', function () { show(cur + 1); });
    stage.addEventListener('keydown', function (e) {
      var k = e.key;
      if (k === 'ArrowRight' || k === 'PageDown') { show(cur + 1); e.preventDefault(); }
      else if (k === 'ArrowLeft' || k === 'PageUp') { show(cur - 1); e.preventDefault(); }
      else if (k === 'Home') { show(1); e.preventDefault(); }
      else if (k === 'End') { show(N); e.preventDefault(); }
    });
    var x0 = null;
    stage.addEventListener('pointerdown', function (e) { x0 = e.clientX; });
    stage.addEventListener('pointerup', function (e) {
      if (x0 === null) return; var dx = e.clientX - x0; x0 = null;
      if (Math.abs(dx) > 40) show(cur + (dx < 0 ? 1 : -1));
    });
    stage.addEventListener('pointercancel', function () { x0 = null; });
    if (window.ResizeObserver) {
      var lastW = 0, t;
      new ResizeObserver(function () {
        var w = Math.round(stage.clientWidth); if (Math.abs(w - lastW) < 2) return; lastW = w;
        clearTimeout(t); t = setTimeout(function () { show(cur); }, 120);
      }).observe(stage);
    } else show(1);
  }
})();
