/*
 * Tiny hand-rolled SVG chart helpers. Each chart draws at the container's
 * current pixel width (so text stays legible on phones) and redraws on resize.
 * Colours come from CSS classes, so theme switches need no redraw.
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    var node = document.createElementNS(NS, tag);
    if (attrs) {
      for (var k in attrs) {
        if (attrs[k] !== undefined && attrs[k] !== null) node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, str, cls, extra) {
    var t = el("text", Object.assign({ x: x, y: y, class: cls }, extra || {}), parent);
    t.textContent = str;
    return t;
  }

  function svgRoot(container, width, height, title, desc) {
    container.innerHTML = "";
    var svg = el("svg", {
      width: width, height: height, viewBox: "0 0 " + width + " " + height,
      role: "img", "aria-labelledby": container.id + "-t " + container.id + "-d"
    }, container);
    var t = el("title", { id: container.id + "-t" }, svg);
    t.textContent = title;
    var d = el("desc", { id: container.id + "-d" }, svg);
    d.textContent = desc;
    return svg;
  }

  // Visually hidden data table, for screen readers
  function srTable(container, caption, headers, rows) {
    // Tables ignore width/overflow, so the hiding is done by a wrapper div
    var wrap = document.createElement("div");
    wrap.className = "sr-only";
    var table = document.createElement("table");
    wrap.appendChild(table);
    var cap = document.createElement("caption");
    cap.textContent = caption;
    table.appendChild(cap);
    var thead = document.createElement("tr");
    headers.forEach(function (h) {
      var th = document.createElement("th");
      th.scope = "col";
      th.textContent = h;
      thead.appendChild(th);
    });
    table.appendChild(thead);
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      r.forEach(function (c) {
        var td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
    container.appendChild(wrap);
  }

  // ---------- Shared tooltip ----------
  var tip = null;
  function tooltipShow(evt, html) {
    tip = tip || document.getElementById("tooltip");
    if (!tip) return;
    tip.innerHTML = html;
    tip.hidden = false;
    var x = evt.clientX, y = evt.clientY;
    if (x === undefined || (x === 0 && y === 0)) {
      var r = evt.target.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    }
    var half = 110;
    tip.style.left = Math.min(Math.max(x, half), window.innerWidth - half) + "px";
    tip.style.top = y + "px";
  }
  function tooltipHide() {
    if (tip) tip.hidden = true;
  }
  function bindTip(node, html) {
    node.setAttribute("tabindex", "0");
    node.addEventListener("pointerenter", function (e) { tooltipShow(e, html()); });
    node.addEventListener("pointermove", function (e) { tooltipShow(e, html()); });
    node.addEventListener("pointerleave", tooltipHide);
    node.addEventListener("focus", function (e) { tooltipShow(e, html()); });
    node.addEventListener("blur", tooltipHide);
  }

  // =====================================================================
  // Grouped bar chart
  // opts: { categories:[{label, short}], series:[{key, label, cls}],
  //         value(catIndex, seriesKey) -> number|null, metricLabel }
  // =====================================================================
  function GroupedBar(container, opts) {
    this.c = container;
    this.o = opts;
  }

  GroupedBar.prototype.render = function () {
    var o = this.o, c = this.c;
    var W = Math.max(280, c.clientWidth);
    var narrow = W < 560;
    var H = narrow ? 300 : 360;
    var m = { t: 22, r: 6, b: 42, l: 38 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var svg = svgRoot(c, W, H, o.title, o.desc);
    this.geom = { m: m, ph: ph, narrow: narrow };

    var yMax = 0.8;
    this.yMax = yMax;
    var step = narrow ? 0.2 : 0.1;
    var g = el("g", { class: "axis" }, svg);
    for (var v = 0; v <= yMax + 1e-9; v += step) {
      var y = m.t + ph - (v / yMax) * ph;
      el("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, class: v === 0 ? "" : "grid-line" }, g);
      text(g, m.l - 8, y + 4, v.toFixed(1), "tick-label", { "text-anchor": "end" });
    }

    var nC = o.categories.length, nS = o.series.length;
    var band = pw / nC;
    var inner = band * (narrow ? 0.84 : 0.72);
    var bw = inner / nS;
    var self = this;
    this.bars = [];
    this.labels = [];

    o.categories.forEach(function (cat, ci) {
      var gx = m.l + ci * band + (band - inner) / 2;
      text(svg, m.l + ci * band + band / 2, H - m.b + 20, narrow ? cat.short : cat.label, "cat-label");
      o.series.forEach(function (s, si) {
        var x = gx + si * bw + 1;
        var rect = el("rect", { x: x, width: Math.max(2, bw - 3), rx: 2, class: "bar " + s.cls, y: m.t + ph, height: 0 }, svg);
        rect.style.transitionDelay = (ci * 60 + si * 30) + "ms";
        var showLabel = !narrow || s.key === "federated";
        var lbl = showLabel ? text(svg, x + (bw - 3) / 2, m.t + ph, "", "val-label") : null;
        if (lbl) lbl.style.fontSize = narrow ? "10px" : "11px";
        var na = text(svg, x + (bw - 3) / 2, m.t + ph - 4, "", "val-label");
        na.style.fontSize = "9px";
        self.bars.push({ rect: rect, lbl: lbl, na: na, ci: ci, key: s.key, label: s.label });
        bindTip(rect, function () {
          var val = o.value(ci, s.key);
          return cat.label + " · " + s.label + "<br><b>" + o.metricLabel() + " " + (val == null ? "n/a" : val.toFixed(4)) + "</b>";
        });
      });
    });

    this.update();
  };

  GroupedBar.prototype.update = function () {
    var o = this.o, gm = this.geom, yMax = this.yMax;
    this.bars.forEach(function (b) {
      var v = o.value(b.ci, b.key);
      var h = v == null ? 0 : (v / yMax) * gm.ph;
      var y = gm.m.t + gm.ph - h;
      b.rect.setAttribute("y", y);
      b.rect.setAttribute("height", h);
      b.rect.setAttribute("aria-label", b.label + ": " + (v == null ? "not reported" : v.toFixed(4)));
      if (b.lbl) {
        b.lbl.textContent = v == null ? "" : v.toFixed(gm.narrow ? 2 : 3);
        b.lbl.setAttribute("y", y - 6);
      }
      b.na.textContent = v == null ? "n/a" : "";
    });
  };

  // =====================================================================
  // DP line chart: AUROC vs ε (log x, with ∞ after an axis break)
  // =====================================================================
  function DPChart(container, opts) {
    this.c = container;
    this.o = opts;
    this.selected = opts.selected || 0;
  }

  DPChart.prototype.render = function () {
    var o = this.o, c = this.c, rows = o.rows;
    var W = Math.max(280, c.clientWidth);
    var narrow = W < 560;
    var H = narrow ? 280 : 340;
    var m = { t: 20, r: narrow ? 14 : 24, b: 36, l: 44 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var svg = svgRoot(c, W, H, o.title, o.desc);

    var finitePw = pw * 0.82;
    var lmin = Math.log10(0.1), lmax = Math.log10(10);
    function x(i) {
      var e = rows[i].epsilon;
      if (e === null) return m.l + pw;
      return m.l + ((Math.log10(e) - lmin) / (lmax - lmin)) * finitePw;
    }
    var y0 = 0.46, y1 = 0.66;
    function y(v) { return m.t + ph - ((v - y0) / (y1 - y0)) * ph; }
    this.x = x;

    // Sweet-spot band (ε 5–10)
    var i5 = rows.findIndex(function (r) { return r.epsilon === 5; });
    var i10 = rows.findIndex(function (r) { return r.epsilon === 10; });
    el("rect", { x: x(i5), y: m.t, width: x(i10) - x(i5), height: ph, class: "sweet" }, svg);
    text(svg, (x(i5) + x(i10)) / 2, m.t + 14, "sweet spot", "sweet-label", { "text-anchor": "middle" });

    // Grid + y axis
    var g = el("g", { class: "axis" }, svg);
    for (var v = y0; v <= y1 + 1e-9; v += 0.04) {
      el("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: "grid-line" }, g);
      text(g, m.l - 8, y(v) + 4, v.toFixed(2), "tick-label", { "text-anchor": "end" });
    }
    el("line", { x1: m.l, x2: m.l + finitePw + 6, y1: m.t + ph, y2: m.t + ph }, g);
    el("line", { x1: m.l + pw - 14, x2: m.l + pw + 6, y1: m.t + ph, y2: m.t + ph }, g);
    // Axis break
    var bx = (m.l + finitePw + m.l + pw) / 2;
    el("path", { d: "M" + (bx - 7) + "," + (m.t + ph + 5) + " l6,-10 M" + (bx) + "," + (m.t + ph + 5) + " l6,-10", class: "axis-break" }, svg);

    rows.forEach(function (r, i) {
      text(svg, x(i), H - m.b + 20, r.epsilon === null ? "∞" : String(r.epsilon), "tick-label", { "text-anchor": "middle" });
    });
    text(svg, 12, m.t + ph / 2, "AUROC", "tick-label", { "text-anchor": "middle", transform: "rotate(-90 12 " + (m.t + ph / 2) + ")" });

    var finite = rows.map(function (r, i) { return i; }).filter(function (i) { return rows[i].epsilon !== null; });
    var inf = rows.findIndex(function (r) { return r.epsilon === null; });

    var self = this;
    this.dots = [];

    [["gaussian", "gauss", "Gaussian NB"], ["categorical", "cat", "Categorical NB"]].forEach(function (s) {
      var key = s[0], cls = s[1], name = s[2];
      // ± std band
      var up = finite.map(function (i) { return x(i) + "," + y(rows[i][key] + rows[i][key + "Std"]); });
      var dn = finite.slice().reverse().map(function (i) { return x(i) + "," + y(rows[i][key] - rows[i][key + "Std"]); });
      el("polygon", { points: up.concat(dn).join(" "), class: "band--" + cls }, svg);
      // No-noise baseline
      el("line", { x1: m.l, x2: x(inf), y1: y(rows[inf][key]), y2: y(rows[inf][key]), class: "baseline baseline--" + cls }, svg);
      // Main line
      var d = finite.map(function (i, k) { return (k ? "L" : "M") + x(i) + "," + y(rows[i][key]); }).join(" ");
      var path = el("path", { d: d, class: "line line-path line--" + cls }, svg);
      var len = path.getTotalLength ? path.getTotalLength() : 1000;
      path.style.setProperty("--len", Math.ceil(len));
      // Dashed connector across the break
      var last = finite[finite.length - 1];
      el("line", { x1: x(last), y1: y(rows[last][key]), x2: x(inf), y2: y(rows[inf][key]), class: "line line--" + cls, "stroke-dasharray": "3 5", opacity: 0.6 }, svg);
    });

    // Selection cursor
    this.cursor = el("line", { x1: 0, x2: 0, y1: m.t, y2: m.t + ph, class: "cursor" }, svg);

    [["gaussian", "gauss", "Gaussian NB"], ["categorical", "cat", "Categorical NB"]].forEach(function (s) {
      var key = s[0], cls = s[1], name = s[2];
      rows.forEach(function (r, i) {
        var dot = el("circle", { cx: x(i), cy: y(r[key]), r: 4.5, class: "dot dot--" + cls }, svg);
        dot.setAttribute("aria-label", name + " at ε " + (r.epsilon === null ? "∞" : r.epsilon) + ": " + r[key].toFixed(4));
        bindTip(dot, function () {
          return name + " · ε = " + (r.epsilon === null ? "∞" : r.epsilon) +
            "<br><b>AUROC " + r[key].toFixed(4) + (r[key + "Std"] ? " ± " + r[key + "Std"].toFixed(4) : "") + "</b>";
        });
        dot.addEventListener("click", function () { if (o.onPick) o.onPick(i); });
        self.dots.push({ node: dot, i: i });
      });
    });

    this.select(this.selected);
  };

  DPChart.prototype.select = function (i) {
    this.selected = i;
    if (!this.cursor) return;
    var xx = this.x(i);
    this.cursor.setAttribute("x1", xx);
    this.cursor.setAttribute("x2", xx);
    this.dots.forEach(function (d) { d.node.classList.toggle("is-selected", d.i === i); });
  };

  // =====================================================================
  // Horizontal bar chart (feature importance)
  // =====================================================================
  function HBar(container, opts) {
    this.c = container;
    this.o = opts;
  }

  HBar.prototype.render = function () {
    var o = this.o, c = this.c, rows = o.rows;
    var W = Math.max(260, c.clientWidth);
    var narrow = W < 420;
    var labelW = narrow ? 100 : 176;
    var rowH = 46;
    var H = rows.length * rowH + 6;
    var valW = 52;
    var svg = svgRoot(c, W, H, o.title, o.desc);
    var max = Math.max.apply(null, rows.map(function (r) { return r.value; }));
    var bwMax = W - labelW - valW;

    rows.forEach(function (r, i) {
      var y = i * rowH;
      text(svg, 0, y + (narrow ? 24 : 18), r.feature, "feat-label");
      if (!narrow) text(svg, 0, y + 34, r.note, "feat-note");
      el("rect", { x: labelW, y: y + 10, width: bwMax, height: 18, rx: 3, fill: "var(--paper-3)" }, svg);
      var bar = el("rect", { x: labelW, y: y + 10, width: Math.max(2, (r.value / max) * bwMax), height: 18, rx: 3, class: "hbar" + (i < 3 ? " hbar--top" : "") }, svg);
      bar.style.transitionDelay = (i * 80) + "ms";
      bindTip(bar, function () { return r.feature + "<br><b>attribution " + r.value.toFixed(4) + "</b>"; });
      text(svg, W, y + 24, r.value.toFixed(4), "val-label", { "text-anchor": "end" });
    });
  };

  // =====================================================================
  // Depth-3 decision tree for the surrogate rules
  // =====================================================================
  function RuleTree(container, opts) {
    this.c = container;
    this.o = opts;
    this.active = null;
  }

  RuleTree.prototype.render = function () {
    var o = this.o, c = this.c, t = o.tree;
    var W = Math.max(290, c.clientWidth);
    var narrow = W < 520;
    var H = narrow ? 300 : 340;
    var svg = svgRoot(c, W, H, o.title, o.desc);
    var levels = narrow ? [18, 96, 174, 252] : [24, 110, 196, 284];
    var nodeH = 28;

    function cx(level, idx) {
      var n = Math.pow(2, level);
      return (W / n) * (idx + 0.5);
    }

    var nodeW = [Math.min(130, W / 2 - 16), Math.min(120, W / 2 - 16), Math.min(116, W / 4 - 6), Math.max(30, W / 8 - 5)];
    this.edges = {};
    this.nodes = {};
    var self = this;

    function label(feature, thr, level) {
      if (narrow && level === 2) return feature.short + "≤" + thr;
      return feature.name + " ≤ " + thr;
    }

    // Edges first so nodes draw over them
    for (var lvl = 0; lvl < 3; lvl++) {
      var count = Math.pow(2, lvl);
      for (var i = 0; i < count; i++) {
        [0, 1].forEach(function (side) {
          var child = i * 2 + side;
          var x1 = cx(lvl, i), y1 = levels[lvl] + nodeH;
          var x2 = cx(lvl + 1, child), y2 = levels[lvl + 1];
          var my = (y1 + y2) / 2;
          var p = el("path", { d: "M" + x1 + "," + y1 + " C" + x1 + "," + my + " " + x2 + "," + my + " " + x2 + "," + y2, class: "tree-edge" }, svg);
          self.edges[lvl + "-" + child] = p;
          if (!narrow && lvl < 2) {
            text(svg, (x1 + x2) / 2 + (side ? 12 : -12), my - 2, side ? "no" : "yes", "tree-edge-label");
          }
        });
      }
    }

    // Split nodes
    t.splits.forEach(function (levelNodes, lvl) {
      levelNodes.forEach(function (n, i) {
        var g = el("g", { class: "tree-node" }, svg);
        var w = nodeW[lvl];
        el("rect", { x: cx(lvl, i) - w / 2, y: levels[lvl], width: w, height: nodeH, rx: 6 }, g);
        text(g, cx(lvl, i), levels[lvl] + 18, label(n.feature, n.thr, lvl), "");
        self.nodes[lvl + "-" + i] = g;
      });
    });

    // Leaves
    var lw = nodeW[3];
    t.leaves.forEach(function (leaf, i) {
      var g = el("g", { class: "tree-leaf leaf--" + leaf.level }, svg);
      var x = cx(3, i), y = levels[3];
      el("rect", { x: x - lw / 2, y: y, width: lw, height: 40, rx: 6 }, g);
      text(g, x, y + 17, (leaf.risk < 10 ? leaf.risk.toFixed(1) : Math.round(leaf.risk)) + "%", "");
      text(g, x, y + 32, "R" + leaf.id, "lid");
      g.setAttribute("tabindex", "0");
      g.setAttribute("role", "button");
      g.setAttribute("aria-label", "Rule " + leaf.id + ", risk " + leaf.risk + "%");
      g.style.cursor = "pointer";
      g.addEventListener("click", function () { if (o.onPick) o.onPick(i); });
      g.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (o.onPick) o.onPick(i); }
      });
      self.nodes["3-" + i] = g;
    });

    if (this.active !== null) this.highlight(this.active);
  };

  // leafIndex 0..7 → light up the path root → leaf
  RuleTree.prototype.highlight = function (leafIndex) {
    this.active = leafIndex;
    var self = this;
    Object.keys(this.nodes).forEach(function (k) { self.nodes[k].classList.remove("is-on"); });
    Object.keys(this.edges).forEach(function (k) { self.edges[k].classList.remove("is-on"); });
    var path = [0, leafIndex >> 2, leafIndex >> 1, leafIndex];
    for (var lvl = 0; lvl < 4; lvl++) {
      var n = this.nodes[lvl + "-" + path[lvl]];
      if (n) n.classList.add("is-on");
      if (lvl > 0) {
        var e = this.edges[(lvl - 1) + "-" + path[lvl]];
        if (e) e.classList.add("is-on");
      }
    }
  };

  window.Charts = {
    GroupedBar: GroupedBar,
    DPChart: DPChart,
    HBar: HBar,
    RuleTree: RuleTree,
    srTable: srTable,
    svg: el,
    text: text,
    tooltipHide: tooltipHide
  };
})();
