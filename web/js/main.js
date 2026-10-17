/*
 * Page behaviour: theme, scroll progress, reveals, the federation stepper,
 * and every interactive chart. Data comes from js/data.js (window.PROJECT_DATA).
 */
(function () {
  "use strict";

  var D = window.PROJECT_DATA;
  var C = window.Charts;
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function fmt(n, d) { return Number(n).toFixed(d); }
  function epsLabel(e) { return e === null ? "∞" : String(e); }

  // ---------------------------------------------------------------
  // Theme toggle
  // ---------------------------------------------------------------
  function currentTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  $("#theme-toggle").addEventListener("click", function () {
    var next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
  });

  // ---------------------------------------------------------------
  // Scroll progress + top bar
  // ---------------------------------------------------------------
  var bar = $("#progress-bar"), topbar = $(".topbar");
  var ticking = false;
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    bar.style.transform = "scaleX(" + (max > 0 ? h.scrollTop / max : 0) + ")";
    topbar.classList.toggle("is-scrolled", h.scrollTop > 10);
    ticking = false;
  }
  window.addEventListener("scroll", function () {
    if (!ticking) { ticking = true; requestAnimationFrame(onScroll); }
  }, { passive: true });
  onScroll();

  // ---------------------------------------------------------------
  // Reveal-on-scroll and chart draw-in
  // ---------------------------------------------------------------
  var hasIO = "IntersectionObserver" in window;
  var onVisibleHandlers = new Map();

  var revealIO = hasIO ? new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      en.target.classList.add("is-visible");
      var fn = onVisibleHandlers.get(en.target);
      if (fn) { fn(); onVisibleHandlers.delete(en.target); }
      revealIO.unobserve(en.target);
    });
  }, { rootMargin: "0px 0px -12% 0px", threshold: 0.05 }) : null;

  function whenVisible(node, fn) {
    if (!hasIO) { node.classList.add("is-visible"); fn && fn(); return; }
    if (fn) onVisibleHandlers.set(node, fn);
    revealIO.observe(node);
  }

  $$(".reveal").forEach(function (n) { whenVisible(n); });

  // Charts get .is-drawn when they scroll into view, which triggers grow-in transitions
  function drawOnVisible(container) {
    whenVisible(container, function () {
      requestAnimationFrame(function () { container.classList.add("is-drawn"); });
    });
  }

  // ---------------------------------------------------------------
  // Chapter nav active state
  // ---------------------------------------------------------------
  if (hasIO) {
    var navLinks = $$(".chapter-nav a");
    var navIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        navLinks.forEach(function (a) {
          a.classList.toggle("is-active", a.getAttribute("href") === "#" + en.target.id);
        });
      });
    }, { rootMargin: "-40% 0px -55% 0px" });
    $$("main > section[id]").forEach(function (s) { if (s.id !== "top") navIO.observe(s); });
  }

  // ---------------------------------------------------------------
  // Network diagram (hero + federation stepper)
  // ---------------------------------------------------------------
  function buildNetwork(container, variant) {
    var S = C.svg, T = C.text;
    container.innerHTML = "";
    var svg = S("svg", { viewBox: "0 -10 400 440", "aria-hidden": "true" }, container);
    var cx = 200, cy = 200, R = 150, n = D.dataset.hospitals;
    var pts = [];
    for (var i = 0; i < n; i++) {
      var a = (-90 + i * (360 / n)) * Math.PI / 180;
      pts.push({ x: cx + R * Math.cos(a), y: cy + R * Math.sin(a) });
    }

    pts.forEach(function (p) { S("line", { x1: p.x, y1: p.y, x2: cx, y2: cy, class: "net-link" }, svg); });

    function packets(cls, reverse, delayBase) {
      var g = S("g", { class: cls }, svg);
      pts.forEach(function (p, i) {
        var from = reverse ? cx + "," + cy : p.x + "," + p.y;
        var to = reverse ? p.x + "," + p.y : cx + "," + cy;
        for (var k = 0; k < 2; k++) {
          var c = S("circle", { r: 6, class: "packet" + (reverse ? " packet--down" : "") }, g);
          var anim = S("animateMotion", {
            path: "M" + from + " L" + to, dur: "2.4s", repeatCount: "indefinite",
            begin: -(delayBase + i * 0.35 + k * 1.2 + 0.01) + "s"
          }, c);
          anim.setAttribute("calcMode", "spline");
          anim.setAttribute("keySplines", "0.4 0 0.2 1");
          anim.setAttribute("keyTimes", "0;1");
        }
      });
      return g;
    }

    if (variant === "hero") {
      packets("group-up", false, 0);
    } else {
      packets("group-up", false, 0);
      packets("group-down", true, 0);
      // Ensemble of trees around the server
      var ens = S("g", { class: "ensemble" }, svg);
      for (var t = 0; t < 12; t++) {
        var ang = (t / 12) * Math.PI * 2;
        var tx = cx + 58 * Math.cos(ang), ty = cy + 58 * Math.sin(ang);
        S("path", { d: "M" + tx + "," + (ty - 8) + " l6,11 h-12 z", class: "tree" }, ens);
        S("rect", { x: tx - 1, y: ty + 3, width: 2, height: 4, class: "tree" }, ens);
      }
    }

    // Server
    var sg = S("g", { class: "net-server" }, svg);
    S("circle", { cx: cx, cy: cy, r: 36, class: "halo" }, sg);
    S("circle", { cx: cx, cy: cy, r: 34, class: "core" }, sg);
    T(sg, cx, cy - 2, "SERVER", "");
    var sub = T(sg, cx, cy + 12, variant === "hero" ? "aggregates" : "global model", "");
    sub.style.fontSize = "9px";
    sub.style.fontWeight = "400";

    // Hospitals
    pts.forEach(function (p, i) {
      var g = S("g", { class: "net-hosp" }, svg);
      S("circle", { cx: p.x, cy: p.y, r: 38, class: "ring", "stroke-dasharray": "6 6" }, g);
      S("rect", { x: p.x - 24, y: p.y - 20, width: 48, height: 40, rx: 6, class: "body" }, g);
      S("path", { d: "M" + p.x + "," + (p.y - 9) + " v18 M" + (p.x - 9) + "," + p.y + " h18", class: "cross" }, g);
      // Local database with a lock: the data stays here
      var dx = p.x + 22, dy = p.y + 14;
      S("ellipse", { cx: dx, cy: dy, rx: 9, ry: 3.5, class: "db" }, g);
      S("path", { d: "M" + (dx - 9) + "," + dy + " v10 a9,3.5 0 0 0 18,0 v-10", class: "db" }, g);
      S("path", { d: "M" + (dx - 3) + "," + (dy + 6) + " h6 v5 h-6 z M" + (dx - 2) + "," + (dy + 6) + " v-2 a2,2 0 0 1 4,0 v2", class: "lock" }, g);
      T(g, p.x, p.y + 46, "Hospital " + (i + 1), "");
    });

    if (variant !== "hero") {
      var caps = [
        "train locally · data stays",
        "upload trees / statistics",
        "aggregate: union of 250 trees",
        "broadcast the global model"
      ];
      caps.forEach(function (txt, k) {
        T(svg, cx, 412, txt, "caption-step", { "data-for": String(k + 1) });
      });
    }

    if (reduceMotion && svg.pauseAnimations) svg.pauseAnimations();
    return svg;
  }

  buildNetwork($("#hero-art"), "hero");
  var fedGraphic = $("#fed-diagram");
  buildNetwork(fedGraphic, "federation");

  var steps = $$(".step");
  function setStep(n) {
    fedGraphic.setAttribute("data-step", n);
    steps.forEach(function (s) { s.classList.toggle("is-active", s.getAttribute("data-step") === n); });
  }
  if (hasIO) {
    var stepIO = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) setStep(en.target.getAttribute("data-step"));
      });
    }, { rootMargin: "-45% 0px -45% 0px" });
    steps.forEach(function (s) { stepIO.observe(s); });
  }
  setStep("1");

  // ---------------------------------------------------------------
  // Chapter 1: fact strip
  // ---------------------------------------------------------------
  var ds = D.dataset;
  var facts = [
    [ds.hospitals, "hospitals (clients)"],
    [ds.train.toLocaleString("en-US"), "training patients"],
    [ds.test.toLocaleString("en-US"), "global test patients"],
    [ds.features, "clinical features"],
    ["~" + Math.round(ds.mortalityRate * 100) + "%", "in-hospital mortality"],
    [ds.windowHours + "h", "observation window"]
  ];
  $("#fact-strip").innerHTML = facts.map(function (f) {
    return '<li><span class="v">' + f[0] + '</span><span class="l">' + f[1] + "</span></li>";
  }).join("");

  // ---------------------------------------------------------------
  // Chapter 3: grouped bar chart with metric toggle
  // ---------------------------------------------------------------
  var shortNames = { "Random Forest": "RF", "Decision Tree": "DT", "Gaussian NB": "GNB", "Categorical NB": "CNB", "Logistic Reg.": "LR" };
  var metricNames = { auroc: "AUROC", auprc: "AUPRC", accuracy: "Accuracy" };
  var metric = "auroc";
  var resultsEl = $("#results-chart");

  var resultsChart = new C.GroupedBar(resultsEl, {
    title: "Model performance by training setting",
    desc: "Grouped bars comparing centralised, local-only and federated training for five models.",
    categories: D.models.map(function (m) { return { label: m.name, short: shortNames[m.name] || m.name }; }),
    series: [
      { key: "centralized", label: "Centralised", cls: "bar--central" },
      { key: "local", label: "Local-only", cls: "bar--local" },
      { key: "federated", label: "Federated", cls: "bar--fed" }
    ],
    value: function (ci, key) {
      var row = D.models[ci][key];
      return row ? row[metric] : null;
    },
    metricLabel: function () { return metricNames[metric]; }
  });

  function resultsTable() {
    var old = resultsEl.querySelector(".sr-only");
    if (old) old.remove();
    C.srTable(resultsEl, metricNames[metric] + " by model and setting",
      ["Model", "Centralised", "Local-only", "Federated"],
      D.models.map(function (m) {
        return [m.name].concat(["centralized", "local", "federated"].map(function (k) {
          return m[k] ? fmt(m[k][metric], 4) : "n/a";
        }));
      }));
  }

  resultsChart.render();
  resultsTable();
  drawOnVisible(resultsEl);

  $$(".segmented__btn").forEach(function (btn) {
    btn.addEventListener("click", function () {
      metric = btn.getAttribute("data-metric");
      $$(".segmented__btn").forEach(function (b) {
        var on = b === btn;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
      resultsChart.update();
      resultsTable();
    });
  });

  // Forest weighting table
  var best = Math.max.apply(null, D.forestWeighting.map(function (w) { return w.auroc; }));
  $("#weighting-table").innerHTML =
    '<table><caption>Forest Union tree-weighting strategies</caption>' +
    '<thead><tr><th scope="col">Weighting</th><th scope="col" style="text-align:right">AUROC</th></tr></thead><tbody>' +
    D.forestWeighting.map(function (w) {
      return '<tr class="' + (w.auroc === best ? "is-best" : "") + '"><td>' + w.weighting.replace("_", " ") +
        '</td><td class="num">' + fmt(w.auroc, 4) + "</td></tr>";
    }).join("") +
    '<tr><td>Centralised RF (reference)</td><td class="num">' + fmt(D.models[0].centralized.auroc, 4) + "</td></tr>" +
    "</tbody></table>";

  // Count-up animation for callout numbers
  $$("[data-count]").forEach(function (node) {
    var target = parseFloat(node.getAttribute("data-count"));
    var decimals = parseInt(node.getAttribute("data-decimals") || "0", 10);
    var suffix = node.getAttribute("data-suffix") || "";
    if (reduceMotion) return;
    node.textContent = fmt(0, decimals) + suffix;
    whenVisible(node, function () {
      var start = null, dur = 1200;
      function frame(ts) {
        if (!start) start = ts;
        var p = Math.min(1, (ts - start) / dur);
        var eased = 1 - Math.pow(1 - p, 3);
        node.textContent = fmt(target * eased, decimals) + suffix;
        if (p < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  });

  // ---------------------------------------------------------------
  // Chapter 4: DP chart + ε slider
  // ---------------------------------------------------------------
  var rows = D.dpSweep;
  var slider = $("#eps-slider");
  var dpEl = $("#dp-chart");

  var dpChart = new C.DPChart(dpEl, {
    title: "AUROC versus privacy budget epsilon",
    desc: "Two lines for Gaussian and Categorical Naive Bayes. AUROC rises as epsilon grows; the sweet spot is epsilon 5 to 10.",
    rows: rows,
    selected: parseInt(slider.value, 10),
    onPick: function (i) { slider.value = i; updateEps(); }
  });
  dpChart.render();
  C.srTable(dpEl, "AUROC by epsilon", ["ε", "Gaussian NB", "Categorical NB"], rows.map(function (r) {
    return [epsLabel(r.epsilon), fmt(r.gaussian, 4), fmt(r.categorical, 4)];
  }));
  drawOnVisible(dpEl);

  function sliderFill(input) {
    var min = +input.min, max = +input.max;
    input.style.setProperty("--fill", ((input.value - min) / (max - min)) * 100 + "%");
  }

  function updateEps() {
    var i = parseInt(slider.value, 10);
    var r = rows[i];
    var inf = rows[rows.length - 1];
    sliderFill(slider);
    slider.setAttribute("aria-valuetext", "epsilon " + epsLabel(r.epsilon));
    $("#eps-value").textContent = "ε = " + epsLabel(r.epsilon);
    dpChart.select(i);

    var verdict, cls = "";
    if (r.epsilon === null) {
      verdict = "No noise: the best accuracy, but the shared statistics carry no formal privacy guarantee.";
    } else if (r.epsilon <= 1) {
      verdict = "Strong privacy, but the model is barely better than a coin flip (AUROC ≈ 0.5) and its explanations become unstable.";
      cls = "is-bad";
    } else if (r.epsilon < 5) {
      verdict = "Getting closer. Categorical NB starts to recover, but the feature rankings are still noisy.";
    } else {
      verdict = "Sweet spot. Categorical NB is within " + fmt(inf.categorical - r.categorical, 3) +
        " AUROC of the noise-free model and still beats centralised training, while every statistic is formally private.";
      cls = "is-good";
    }

    var gapText = (r.gap > 0 ? "+" : "") + fmt(r.gap, 4);
    $("#eps-readout").innerHTML =
      cell("Gaussian NB AUROC", fmt(r.gaussian, 4) + (r.gaussianStd ? " <small>± " + fmt(r.gaussianStd, 3) + "</small>" : "")) +
      cell("Categorical NB AUROC", fmt(r.categorical, 4) + (r.categoricalStd ? " <small>± " + fmt(r.categoricalStd, 3) + "</small>" : "")) +
      cell("Utility gap vs centralised", gapText) +
      cell("Explanation stability ρ", fmt(r.rho, 3)) +
      '<p class="readout__verdict ' + cls + '">' + verdict + "</p>";
  }
  function cell(l, v) {
    return '<div class="readout__cell"><span class="l">' + l + '</span><span class="v">' + v + "</span></div>";
  }
  slider.addEventListener("input", updateEps);
  updateEps();

  // ---------------------------------------------------------------
  // Chapter 4b: secure aggregation demo (real arithmetic)
  // ---------------------------------------------------------------
  $("#secagg-dev").textContent = D.secureAgg.maxDeviation;

  var clients = [
    { name: "Hospital A", u: [0.42, 0.17, 0.88] },
    { name: "Hospital B", u: [0.35, 0.61, 0.12] },
    { name: "Hospital C", u: [0.29, 0.44, 0.53] }
  ];
  var dims = 3;

  function randMask() {
    // Masks are whole hundredths so displayed values add up exactly
    return Math.round((Math.random() * 2 - 1) * 900) / 100;
  }

  function computeSecAgg() {
    // Pairwise masks m[i][j] for i < j; client i adds, client j subtracts
    var masks = {};
    for (var i = 0; i < clients.length; i++) {
      for (var j = i + 1; j < clients.length; j++) {
        masks[i + "-" + j] = [];
        for (var d = 0; d < dims; d++) masks[i + "-" + j].push(randMask());
      }
    }
    var masked = clients.map(function (c, i) {
      return c.u.map(function (v, d) {
        var s = v;
        for (var j = 0; j < clients.length; j++) {
          if (j > i) s += masks[i + "-" + j][d];
          if (j < i) s -= masks[j + "-" + i][d];
        }
        return s;
      });
    });
    var sumMasked = [], sumTrue = [];
    for (var d2 = 0; d2 < dims; d2++) {
      sumMasked.push(masked.reduce(function (a, r) { return a + r[d2]; }, 0));
      sumTrue.push(clients.reduce(function (a, c) { return a + c.u[d2]; }, 0));
    }
    var maxDiff = 0;
    for (var d3 = 0; d3 < dims; d3++) maxDiff = Math.max(maxDiff, Math.abs(sumMasked[d3] - sumTrue[d3]));
    return { masked: masked, sum: sumMasked, maxDiff: maxDiff };
  }

  function cells(vec, cls) {
    return '<div class="sa-vec">' + vec.map(function (v) {
      return '<span class="sa-cell ' + (cls || "") + '">' + fmt(v, 2) + "</span>";
    }).join("") + "</div>";
  }

  function renderSecAgg() {
    var r = computeSecAgg();
    var html = '<p class="sa-section">Private update (stays at the hospital)</p>';
    clients.forEach(function (c) {
      html += '<div class="sa-row"><span class="sa-who">' + c.name + "</span>" + cells(c.u) + "</div>";
    });
    html += '<p class="sa-section">Masked upload (all the server receives)</p>';
    clients.forEach(function (c, i) {
      html += '<div class="sa-row"><span class="sa-who">' + c.name + "<small>u + masks</small></span>" + cells(r.masked[i], "sa-cell--masked flash") + "</div>";
    });
    html += '<p class="sa-section">Server adds the uploads</p>';
    html += '<div class="sa-row"><span class="sa-who">Σ uploads<small>= Σ private</small></span>' + cells(r.sum, "sa-cell--sum") + "</div>";
    html += '<div class="sa-foot"><span>Live error vs true sum: <span class="mono">' +
      (r.maxDiff === 0 ? "0" : r.maxDiff.toExponential(2)) + '</span></span>' +
      '<button type="button" class="btn btn--accent" id="sa-shuffle">Shuffle masks</button></div>';
    $("#secagg-demo").innerHTML = html;
    $("#sa-shuffle").addEventListener("click", function () {
      renderSecAgg();
      $("#sa-shuffle").focus();
    });
  }
  renderSecAgg();

  // ---------------------------------------------------------------
  // Chapter 5: feature importance + consistency
  // ---------------------------------------------------------------
  var fiEl = $("#fi-chart");
  var fiChart = new C.HBar(fiEl, {
    title: "Federated feature attribution",
    desc: "Age, SOFA score and APACHE score are the three strongest predictors.",
    rows: D.featureImportance
  });
  fiChart.render();
  C.srTable(fiEl, "Feature attribution", ["Feature", "Attribution"], D.featureImportance.map(function (f) {
    return [f.feature, fmt(f.value, 4)];
  }));
  drawOnVisible(fiEl);

  var cl = $("#consistency-list");
  cl.innerHTML = D.consistency.map(function (s) {
    return "<li><span>" + s.label + '</span><span class="meter"><i data-w="' + Math.max(0, s.mean) * 100 + '"></i></span>' +
      '<span class="v">' + fmt(s.mean, 3) + " ± " + fmt(s.std, 3) + "</span></li>";
  }).join("");
  whenVisible(cl, function () {
    $$("i[data-w]", cl).forEach(function (i) { i.style.width = i.getAttribute("data-w") + "%"; });
  });

  // ---------------------------------------------------------------
  // Chapter 6: rule explorer
  // ---------------------------------------------------------------
  $("#fidelity").textContent = fmt(D.surrogate.fidelity * 100, 1) + "%";
  $("#kappa").textContent = fmt(D.surrogate.kappa, 2);

  function thr(cond) { return parseFloat(cond.replace(/[<>=]/g, "")); }
  function riskLevel(r) { return r < 15 ? "low" : r < 40 ? "mid" : "high"; }
  var levelName = { low: "Low risk", mid: "Moderate risk", high: "High risk" };

  var R = D.rules;
  var F = {
    age: { name: "age", short: "age" },
    sofa: { name: "sofa", short: "SOFA" },
    apache: { name: "apache", short: "AP" }
  };
  // Tree is derived from the rule list: leaves are in left-to-right order
  var tree = {
    splits: [
      [{ feature: F.age, thr: thr(R[0].age) }],
      [{ feature: F.sofa, thr: thr(R[0].sofa) }, { feature: F.sofa, thr: thr(R[4].sofa) }],
      [0, 1, 2, 3].map(function (j) { return { feature: F.apache, thr: thr(R[j * 2].apache) }; })
    ],
    leaves: R.map(function (r) { return { id: r.id, risk: r.risk, level: riskLevel(r.risk) }; })
  };

  var inputs = { age: $("#ex-age"), sofa: $("#ex-sofa"), apache: $("#ex-apache") };
  var treeEl = $("#tree-chart");
  var ruleTree = new C.RuleTree(treeEl, {
    title: "Surrogate decision tree",
    desc: "A depth-3 tree splitting on age, SOFA score and APACHE score, ending in eight risk leaves.",
    tree: tree,
    onPick: function (leaf) {
      // Snap sliders to a representative patient for the clicked leaf
      var path = [0, leaf >> 2, leaf >> 1];
      var goRight = [(leaf >> 2) & 1, (leaf >> 1) & 1, leaf & 1];
      var t = [tree.splits[0][0].thr, tree.splits[1][path[1]].thr, tree.splits[2][path[2]].thr];
      ["age", "sofa", "apache"].forEach(function (k, lvl) {
        inputs[k].value = goRight[lvl] ? Math.min(9, Math.ceil(t[lvl]) + 1) : Math.max(0, Math.floor(t[lvl]) - 1);
      });
      updateExplorer();
    }
  });

  function cond(name, value, threshold) {
    return name + (value > threshold ? " &gt; " : " ≤ ") + threshold;
  }

  function updateExplorer() {
    var v = { age: +inputs.age.value, sofa: +inputs.sofa.value, apache: +inputs.apache.value };
    Object.keys(inputs).forEach(function (k) {
      $("#ex-" + k + "-out").textContent = v[k];
      sliderFill(inputs[k]);
    });
    var t0 = tree.splits[0][0].thr;
    var a = v.age > t0 ? 1 : 0;
    var t1 = tree.splits[1][a].thr;
    var b = v.sofa > t1 ? 1 : 0;
    var t2 = tree.splits[2][a * 2 + b].thr;
    var c = v.apache > t2 ? 1 : 0;
    var leaf = a * 4 + b * 2 + c;
    var rule = R[leaf];
    var lvl = riskLevel(rule.risk);
    ruleTree.highlight(leaf);

    $("#ex-result").innerHTML =
      '<div class="result__risk"><span class="v">' + fmt(rule.risk, 1) + '%</span>' +
      '<span class="badge badge--' + lvl + '">' + levelName[lvl] + "</span></div>" +
      '<div class="gauge"><span class="gauge__pin" style="left:' + rule.risk + '%"></span></div>' +
      '<div class="gauge__scale"><span>0%</span><span>50%</span><span>100%</span></div>' +
      '<p class="result__rule"><b>Rule ' + rule.id + "</b> · IF " + cond("age", v.age, t0) + " AND " +
      cond("sofa", v.sofa, t1) + " AND " + cond("apache", v.apache, t2) +
      " THEN predicted mortality = <b>" + rule.predict + "</b></p>";
  }

  ruleTree.render();
  C.srTable(treeEl, "Surrogate rules", ["Rule", "Age", "SOFA", "APACHE", "Risk"], R.map(function (r) {
    return ["R" + r.id, r.age, r.sofa, r.apache, r.risk + "%"];
  }));
  Object.keys(inputs).forEach(function (k) { inputs[k].addEventListener("input", updateExplorer); });
  $$("[data-preset]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var p = btn.getAttribute("data-preset").split(",");
      inputs.age.value = p[0]; inputs.sofa.value = p[1]; inputs.apache.value = p[2];
      updateExplorer();
    });
  });
  updateExplorer();

  // ---------------------------------------------------------------
  // Chapter 7: scaling bars + k-means facts
  // ---------------------------------------------------------------
  var maxIt = Math.max.apply(null, D.scaling.map(function (s) { return s.iterations; }));
  var sb = $("#scaling-bars");
  sb.innerHTML = D.scaling.map(function (s) {
    return '<div class="hbars__row' + (s.iterations === maxIt ? " hbars__row--slow" : "") + '"><span>' + s.method +
      '</span><span class="hbars__track"><span class="hbars__fill" data-w="' + (s.iterations / maxIt) * 100 +
      '"></span></span><span class="hbars__n">' + s.iterations + "</span></div>";
  }).join("");
  whenVisible(sb, function () {
    $$(".hbars__fill", sb).forEach(function (f) { f.style.width = f.getAttribute("data-w") + "%"; });
  });

  var km = D.kmeans;
  $("#kmeans-kv").innerHTML = [
    ["Subgroups (k)", km.k], ["Federated rounds", km.rounds],
    ["Silhouette", fmt(km.silhouette, 3)], ["ARI vs centralised", fmt(km.ari, 4)]
  ].map(function (p) { return "<div><dt>" + p[0] + "</dt><dd>" + p[1] + "</dd></div>"; }).join("");

  // ---------------------------------------------------------------
  // Appendix: CO table + reproduce commands
  // ---------------------------------------------------------------
  $("#co-table tbody").innerHTML = D.courseOutcomes.map(function (c) {
    return "<tr><td>" + c.co + "</td><td>" + c.focus + "</td><td>" + c.work + "</td><td>" + c.evidence + "</td></tr>";
  }).join("");

  $("#repro-code").innerHTML = D.reproduce.map(function (cmd) {
    return '<span class="p">$ </span>' + cmd;
  }).join("\n");

  var copyBtn = $("#copy-cmds");
  copyBtn.addEventListener("click", function () {
    var txt = D.reproduce.join("\n");
    function done() {
      copyBtn.textContent = "Copied";
      setTimeout(function () { copyBtn.textContent = "Copy commands"; }, 1600);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(txt).then(done, fallback);
    } else {
      fallback();
    }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = txt;
      ta.style.position = "absolute";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand("copy"); done(); } catch (e) {}
      ta.remove();
    }
  });

  // ---------------------------------------------------------------
  // Figure lightbox
  // ---------------------------------------------------------------
  var lb = $("#lightbox");
  $$("[data-zoom]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      $("#lightbox-img").src = btn.getAttribute("data-zoom");
      $("#lightbox-img").alt = btn.querySelector("img").alt;
      $("#lightbox-cap").textContent = btn.getAttribute("data-caption");
      if (lb.showModal) lb.showModal();
    });
  });
  lb.addEventListener("click", function (e) { if (e.target === lb) lb.close(); });

  // ---------------------------------------------------------------
  // Redraw charts on resize (they draw at real pixel width)
  // ---------------------------------------------------------------
  var lastW = window.innerWidth, rt;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth;
      C.tooltipHide();
      resultsChart.render(); resultsTable();
      dpChart.render();
      C.srTable(dpEl, "AUROC by epsilon", ["ε", "Gaussian NB", "Categorical NB"], rows.map(function (r) {
        return [epsLabel(r.epsilon), fmt(r.gaussian, 4), fmt(r.categorical, 4)];
      }));
      fiChart.render();
      ruleTree.render();
    }, 150);
  });
  window.addEventListener("scroll", C.tooltipHide, { passive: true });
})();
