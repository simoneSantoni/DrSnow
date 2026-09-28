// DrSnow GUI frontend.
// Security: all data from the server (column names, cell values, messages) is put in
// the page with textContent / createElement only; no HTML-parsing DOM API is used. Strings
// passed to Plotly (which interprets a small HTML subset) are neutralized by plotText.
"use strict";

(function () {
  let csrfToken = null;
  let columns = [];          // [{name, kind, n_missing, n_unique}]
  let limits = null;

  // ---------------------------------------------------------------- helpers
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "text") node.textContent = String(v);
        else if (k === "className") node.className = v;
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const c of children || []) {
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function byId(id) { return document.getElementById(id); }

  function setStatus(id, msg, kind) {
    const node = byId(id);
    node.textContent = msg || "";
    node.className = "status" + (kind ? " " + kind : "");
  }

  function plotText(s) {
    return String(s).replace(/[<>&]/g, function (c) {
      return c === "<" ? "‹" : c === ">" ? "›" : "＆";
    });
  }

  function fmt(v) {
    if (v === null || v === undefined) return "—";
    if (v === "Inf") return "∞";
    if (v === "-Inf") return "−∞";
    if (typeof v === "boolean") return v ? "true" : "false";
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-4 || a >= 1e6)) return v.toExponential(3);
    return String(Number(v.toPrecision(5)));
  }

  function fmtP(v) {
    if (typeof v === "number" && v < 1e-4) return "< 0.0001";
    return fmt(v);
  }

  let idCounter = 0;
  function uid(prefix) { idCounter += 1; return prefix + "-" + idCounter; }

  // ---------------------------------------------------------------- API
  async function api(method, path, body, headers) {
    const h = Object.assign({ "X-CSRF-Token": csrfToken || "" }, headers || {});
    const resp = await fetch(path, {
      method: method, body: body, headers: h,
      credentials: "same-origin", cache: "no-store"
    });
    let data = null;
    const ctype = resp.headers.get("Content-Type") || "";
    if (ctype.startsWith("application/json")) {
      try { data = await resp.json(); } catch (e) { data = null; }
    }
    if (resp.status === 401) {
      await startSession();
      throw new Error("Your session expired, so the uploaded data were deleted. " +
                      "Please upload the file again.");
    }
    if (!resp.ok) {
      const msg = data && typeof data.error === "string" ? data.error :
        "Request failed (HTTP " + resp.status + ").";
      throw new Error(msg);
    }
    return { data: data, resp: resp };
  }

  async function startSession() {
    const resp = await fetch("/api/session", {
      method: "POST", credentials: "same-origin", cache: "no-store"
    });
    if (!resp.ok) throw new Error("Could not start a session (HTTP " + resp.status + ").");
    const data = await resp.json();
    csrfToken = data.csrf_token;
    limits = data.limits;
    columns = [];
    byId("design-panel").hidden = true;
    byId("preview").replaceChildren();
    const mb = (limits.max_upload_bytes / 1048576).toPrecision(3);
    byId("limits").textContent = "Limits: " + Number(mb) + " MB, " +
      limits.max_rows.toLocaleString() + " rows, " + limits.max_cols + " columns; " +
      "idle sessions expire after " + limits.session_ttl_minutes + " minutes.";
    setStatus("session-status", "Session ready.", "ok");
  }

  // ---------------------------------------------------------------- tables
  function renderTable(tbl, caption) {
    const table = el("table", { className: "data" });
    if (caption) table.appendChild(el("caption", { text: caption }));
    const thead = el("thead");
    const tr = el("tr");
    for (const c of tbl.columns) tr.appendChild(el("th", { scope: "col", text: c }));
    thead.appendChild(tr);
    table.appendChild(thead);
    const tbody = el("tbody");
    for (const row of tbl.rows) {
      const r = el("tr");
      row.forEach(function (v, j) {
        const isNum = typeof v === "number" || v === "Inf" || v === "-Inf";
        const header = String(tbl.columns[j] || "");
        const text = (header.startsWith("Pr(") || header === "p-value" || header === "pvalue") ?
          fmtP(v) : fmt(v);
        r.appendChild(el(j === 0 ? "th" : "td",
          { scope: j === 0 ? "row" : null, className: isNum ? "num" : null, text: text }));
      });
      tbody.appendChild(r);
    }
    table.appendChild(tbody);
    return el("div", { className: "table-wrap", tabindex: "0",
                       role: "region", "aria-label": caption || "table" }, [table]);
  }

  function renderPreview(summary) {
    const box = byId("preview");
    box.replaceChildren();
    const meta = el("p", { className: "hint",
      text: summary.filename + ": " + summary.n_rows.toLocaleString() + " rows, " +
            summary.n_cols + " columns. First " + summary.preview.length + " rows:" });
    const cols = summary.columns.map(function (c) {
      return c.name + " (" + c.kind + (c.n_missing ? ", " + c.n_missing + " missing" : "") + ")";
    });
    const tbl = { columns: summary.columns.map(function (c) { return c.name; }),
                  rows: summary.preview };
    box.appendChild(meta);
    box.appendChild(renderTable(tbl, "Data preview"));
    const details = el("details", null, [el("summary", { text: "Column types" }),
      el("ul", { className: "cols" }, cols.map(function (s) { return el("li", { text: s }); }))]);
    box.appendChild(details);
  }

  // ---------------------------------------------------------------- designs
  const TREATMENT_TYPE = { key: "treatment_type", label: "Treatment column holds",
    type: "choice", options: [["indicator", "a 0/1 treatment indicator"],
      ["first_treated", "the first treated period (0 = never treated)"]] };
  const LEVEL = { key: "level", label: "Confidence level", type: "number",
    value: 0.95, min: 0.5, max: 0.999, step: 0.01 };
  const SEED = { key: "seed", label: "Random seed", type: "number", value: 20260927,
    min: 0, max: 2147483647, step: 1 };

  const DESIGNS = {
    did_twfe: {
      label: "Difference-in-differences: two-way fixed effects",
      help: "Regression of the outcome on unit and period fixed effects and the " +
        "treatment. With staggered adoption the coefficient can weight some effects " +
        "negatively; the Goodman-Bacon decomposition shows the comparisons behind it.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "treatment", label: "Treatment", type: "column", numeric: true },
        TREATMENT_TYPE,
        { key: "unit", label: "Unit identifier", type: "column" },
        { key: "time", label: "Time period", type: "column" },
        { key: "covariates", label: "Time-varying covariates (optional)", type: "columns", numeric: true },
        { key: "cluster", label: "Cluster standard errors by", type: "cluster", defaultLabel: "Unit identifier" },
        { key: "bacon", label: "Goodman-Bacon decomposition (balanced, staggered panels)", type: "checkbox" },
        LEVEL
      ]
    },
    event_study: {
      label: "Event study (dynamic difference-in-differences)",
      help: "Effects by periods relative to first treatment. 'Automatic' uses TWFE with " +
        "a single adoption date and Sun–Abraham with several cohorts.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "treatment", label: "Treatment", type: "column", numeric: true },
        TREATMENT_TYPE,
        { key: "unit", label: "Unit identifier", type: "column" },
        { key: "time", label: "Time period", type: "column" },
        { key: "estimator", label: "Estimator", type: "choice", options: [
          ["auto", "Automatic"], ["twfe", "TWFE dynamic specification"],
          ["sun_abraham", "Sun & Abraham (2021)"],
          ["imputation", "Imputation (Borusyak, Jaravel & Spiess 2024)"],
          ["callaway_santanna", "Callaway & Sant'Anna (2021)"]] },
        { key: "max_pre", label: "Leads to estimate (blank = all)", type: "number", min: 0, max: 1000, step: 1 },
        { key: "max_post", label: "Lags to estimate (blank = all)", type: "number", min: 0, max: 1000, step: 1 },
        { key: "covariates", label: "Covariates (optional)", type: "columns", numeric: true },
        { key: "cluster", label: "Cluster standard errors by", type: "cluster", defaultLabel: "Unit identifier" },
        LEVEL, SEED
      ]
    },
    did_cs: {
      label: "Staggered DiD: Callaway & Sant'Anna",
      help: "Group-time average treatment effects ATT(g, t), aggregated into a " +
        "summary. Requires an absorbing (staggered) treatment.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "treatment", label: "Treatment", type: "column", numeric: true },
        TREATMENT_TYPE,
        { key: "unit", label: "Unit identifier", type: "column" },
        { key: "time", label: "Time period", type: "column" },
        { key: "aggregation", label: "Aggregation", type: "choice", options: [
          ["dynamic", "By event time"], ["simple", "Overall ATT"],
          ["group", "By cohort"], ["calendar", "By calendar period"]] },
        { key: "control_group", label: "Comparison units", type: "choice", options: [
          ["never_treated", "Never treated"], ["not_yet_treated", "Not yet treated"]] },
        { key: "method", label: "Estimation method", type: "choice", options: [
          ["dr", "Doubly robust"], ["dr_improved", "Doubly robust (improved)"],
          ["ipw", "Inverse probability weighting"], ["reg", "Outcome regression"]] },
        { key: "covariates", label: "Covariates (optional)", type: "columns", numeric: true },
        { key: "cluster", label: "Cluster by", type: "cluster", defaultLabel: "Unit identifier" },
        { key: "biters", label: "Bootstrap draws (simultaneous bands)", type: "number", value: 999, min: 99, max: 9999, step: 1 },
        LEVEL, SEED
      ]
    },
    rd: {
      label: "Regression discontinuity",
      help: "Local polynomial estimate at the cutoff with robust bias-corrected " +
        "inference. Give a treatment column for a fuzzy design.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "running", label: "Running variable", type: "column", numeric: true },
        { key: "cutoff", label: "Cutoff", type: "number", value: 0, step: "any" },
        { key: "treatment", label: "Treatment (fuzzy design, optional)", type: "column", numeric: true, optional: true },
        { key: "p", label: "Polynomial order", type: "number", value: 1, min: 0, max: 4, step: 1 },
        { key: "kernel", label: "Kernel", type: "choice", options: [
          ["triangular", "Triangular"], ["epanechnikov", "Epanechnikov"], ["uniform", "Uniform"]] },
        { key: "bwselect", label: "Bandwidth selector", type: "choice", options: [
          ["mserd", "MSE-optimal, common"], ["msetwo", "MSE-optimal, two-sided"],
          ["cerrd", "CER-optimal, common"], ["certwo", "CER-optimal, two-sided"]] },
        { key: "cluster", label: "Cluster by", type: "cluster", defaultLabel: "No clustering" },
        { key: "density_test", label: "Density (manipulation) test", type: "checkbox", value: true },
        LEVEL
      ]
    },
    iv: {
      label: "Instrumental variables (2SLS / LATE)",
      help: "Two-stage least squares with weak-instrument diagnostics and the " +
        "Anderson–Rubin confidence set, which remains valid with weak instruments.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "endogenous", label: "Endogenous treatment", type: "column", numeric: true },
        { key: "instruments", label: "Instrument(s)", type: "columns", numeric: true },
        { key: "covariates", label: "Exogenous covariates (optional)", type: "columns", numeric: true },
        { key: "fe", label: "Fixed effects (optional)", type: "columns" },
        { key: "cluster", label: "Cluster by", type: "cluster", defaultLabel: "No clustering (robust)" },
        LEVEL
      ]
    },
    sdid: {
      label: "Synthetic difference-in-differences",
      help: "Arkhangelsky et al. (2021). Needs a balanced panel with a 0/1 " +
        "absorbing treatment and more never-treated than treated units for placebo " +
        "standard errors.",
      fields: [
        { key: "outcome", label: "Outcome", type: "column", numeric: true },
        { key: "treatment", label: "Treatment (0/1)", type: "column", numeric: true },
        { key: "unit", label: "Unit identifier", type: "column" },
        { key: "time", label: "Time period", type: "column" },
        { key: "method", label: "Estimator", type: "choice", options: [
          ["sdid", "Synthetic DiD"], ["sc", "Synthetic control"], ["did", "Difference-in-differences"]] },
        { key: "se_method", label: "Standard errors", type: "choice", options: [
          ["placebo", "Placebo"], ["bootstrap", "Bootstrap"], ["jackknife", "Jackknife"], ["none", "None"]] },
        { key: "replications", label: "Replications", type: "number", value: 200, min: 10, max: 2000, step: 1 },
        LEVEL, SEED
      ]
    }
  };

  function columnOptions(select, numericOnly, blankLabel) {
    if (blankLabel !== undefined) select.appendChild(el("option", { value: "", text: blankLabel }));
    for (const c of columns) {
      const numeric = c.kind === "numeric" || c.kind === "binary";
      if (numericOnly && !numeric) continue;
      select.appendChild(el("option", { value: c.name, text: c.name }));
    }
  }

  function buildField(f) {
    const id = uid("f-" + f.key);
    const wrap = el("div", { className: "field" });
    let input;
    if (f.type === "column") {
      input = el("select", { id: id, name: f.key, required: !f.optional });
      columnOptions(input, f.numeric, f.optional ? "(none)" : "Choose a column");
    } else if (f.type === "columns") {
      input = el("select", { id: id, name: f.key, multiple: true, size: "4" });
      columnOptions(input, f.numeric);
    } else if (f.type === "cluster") {
      input = el("select", { id: id, name: f.key });
      input.appendChild(el("option", { value: "", text: "Default: " + f.defaultLabel }));
      input.appendChild(el("option", { value: "none", text: "No clustering (robust)" }));
      columnOptions(input, false);
    } else if (f.type === "choice") {
      input = el("select", { id: id, name: f.key });
      for (const [v, label] of f.options) input.appendChild(el("option", { value: v, text: label }));
    } else if (f.type === "checkbox") {
      input = el("input", { id: id, name: f.key, type: "checkbox" });
      input.checked = !!f.value;
      wrap.className = "field check";
      wrap.appendChild(input);
      wrap.appendChild(el("label", { for: id, text: f.label }));
      input.dataset.key = f.key;
      input.dataset.type = f.type;
      return wrap;
    } else {
      input = el("input", { id: id, name: f.key, type: "number", step: f.step,
                            min: f.min, max: f.max });
      if (f.value !== undefined) input.value = String(f.value);
    }
    input.dataset.key = f.key;
    input.dataset.type = f.type;
    wrap.appendChild(el("label", { for: id, text: f.label }));
    wrap.appendChild(input);
    if (f.type === "columns") wrap.appendChild(el("span", { className: "hint small",
      text: "Ctrl/Cmd-click to select several." }));
    return wrap;
  }

  function renderDesignFields() {
    const d = DESIGNS[byId("design").value];
    byId("design-help").textContent = d.help;
    const box = byId("design-fields");
    box.replaceChildren();
    for (const f of d.fields) box.appendChild(buildField(f));
  }

  function collectParams() {
    const params = {};
    for (const input of byId("design-fields").querySelectorAll("[data-key]")) {
      const key = input.dataset.key, type = input.dataset.type;
      if (type === "columns") {
        params[key] = Array.from(input.selectedOptions).map(function (o) { return o.value; });
      } else if (type === "checkbox") {
        params[key] = input.checked;
      } else if (type === "number") {
        if (input.value !== "") params[key] = Number(input.value);
      } else if (input.value !== "") {
        params[key] = input.value;
      }
    }
    return params;
  }

  // ---------------------------------------------------------------- plots
  const COLORS = { series1: "#2a78d6", series2: "#eb6834" };

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function baseLayout(title, xtitle, ytitle) {
    const ink = cssVar("--text-secondary") || "#52514e";
    const grid = cssVar("--grid") || "#e4e3df";
    return {
      title: { text: plotText(title), font: { size: 15 } },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: ink, family: "system-ui, sans-serif" },
      xaxis: { title: { text: plotText(xtitle) }, gridcolor: grid, zerolinecolor: grid },
      yaxis: { title: { text: plotText(ytitle) }, gridcolor: grid, zerolinecolor: ink },
      margin: { t: 48, r: 16, b: 56, l: 64 },
      legend: { orientation: "h", y: -0.2 },
      hovermode: "closest"
    };
  }

  function num(v) {
    if (v === "Inf") return Infinity;
    if (v === "-Inf") return -Infinity;
    return v === null ? NaN : v;
  }

  function eventPlot(p) {
    const lower = p.lower.map(num), upper = p.upper.map(num), y = p.y.map(num);
    const pct = Math.round(p.level * 100);
    const traces = [{
      type: "scatter", mode: "markers", name: "Estimate (" + pct + "% CI)",
      x: p.x, y: y, marker: { color: COLORS.series1, size: 9 },
      error_y: { type: "data", symmetric: false, color: COLORS.series1, thickness: 2, width: 0,
        array: upper.map(function (u, i) { return u - y[i]; }),
        arrayminus: lower.map(function (l, i) { return y[i] - l; }) },
      hovertemplate: "e = %{x}<br>estimate %{y:.4g}<extra></extra>"
    }];
    if (p.reference.length) {
      traces.push({ type: "scatter", mode: "markers", name: "Reference (normalized to 0)",
        x: p.reference, y: p.reference.map(function () { return 0; }),
        marker: { color: COLORS.series1, size: 9, symbol: "circle-open", line: { width: 2 } },
        hovertemplate: "e = %{x} (reference)<extra></extra>" });
    }
    const layout = baseLayout(p.title, "Periods relative to treatment", "Effect");
    layout.shapes = [{ type: "line", xref: "x", yref: "paper", x0: -0.5, x1: -0.5, y0: 0, y1: 1,
      line: { dash: "dot", width: 1, color: cssVar("--text-muted") || "#888" } }];
    return { traces: traces, layout: layout };
  }

  function rdPlot(p) {
    const traces = [
      { type: "scatter", mode: "markers", name: "Bin means", x: p.bins_left.x, y: p.bins_left.y,
        marker: { color: COLORS.series1, size: 8 }, legendgroup: "bins" },
      { type: "scatter", mode: "markers", name: "Bin means (right)", showlegend: false,
        x: p.bins_right.x, y: p.bins_right.y, marker: { color: COLORS.series1, size: 8 },
        legendgroup: "bins" },
      { type: "scatter", mode: "lines", name: "Polynomial fit", x: p.poly_left.x, y: p.poly_left.y,
        line: { color: COLORS.series2, width: 2 }, legendgroup: "fit", hoverinfo: "skip" },
      { type: "scatter", mode: "lines", name: "Polynomial fit (right)", showlegend: false,
        x: p.poly_right.x, y: p.poly_right.y, line: { color: COLORS.series2, width: 2 },
        legendgroup: "fit", hoverinfo: "skip" }
    ];
    const layout = baseLayout(p.title, p.xlabel, p.ylabel);
    layout.shapes = [{ type: "line", xref: "x", yref: "paper", x0: p.cutoff, x1: p.cutoff,
      y0: 0, y1: 1, line: { dash: "dash", width: 1, color: cssVar("--text-muted") || "#888" } }];
    return { traces: traces, layout: layout };
  }

  function sdidPlot(p) {
    const traces = [];
    p.series.forEach(function (s, i) {
      const suffix = p.series.length > 1 ? " (cohort " + plotText(s.cohort) + ")" : "";
      const x = s.time.map(function (t) { return typeof t === "number" ? t : plotText(t); });
      traces.push({ type: "scatter", mode: "lines", name: "Treated" + suffix, x: x,
        y: s.treated.map(num), line: { color: COLORS.series1, width: 2 } });
      traces.push({ type: "scatter", mode: "lines", name: "Synthetic control" + suffix, x: x,
        y: s.synthetic.map(num), line: { color: COLORS.series2, width: 2, dash: "dash" } });
    });
    const layout = baseLayout(p.title, "Time", p.ylabel);
    layout.hovermode = "x unified";
    layout.shapes = p.series.filter(function (s) { return s.adoption !== null; })
      .map(function (s) {
        const x = typeof s.adoption === "number" ? s.adoption : plotText(s.adoption);
        return { type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1,
          line: { dash: "dot", width: 1, color: cssVar("--text-muted") || "#888" } };
      });
    return { traces: traces, layout: layout };
  }

  function drawPlot(container, p) {
    if (typeof window.Plotly === "undefined") {
      container.appendChild(el("p", { className: "hint", text: "Chart library not loaded." }));
      return;
    }
    const spec = p.kind === "event_study" ? eventPlot(p) : p.kind === "rd" ? rdPlot(p) :
      p.kind === "sdid" ? sdidPlot(p) : null;
    if (!spec) return;
    const div = el("div", { className: "plot", role: "img",
      "aria-label": p.title + " (the same numbers are in the tables below)" });
    container.appendChild(div);
    if (p.note) container.appendChild(el("p", { className: "hint small", text: p.note }));
    window.Plotly.newPlot(div, spec.traces, spec.layout,
      { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] });
  }

  // ---------------------------------------------------------------- results
  async function download(res, format, tableIndex) {
    const q = format === "csv" ? "?format=csv&table=" + tableIndex : "?format=json";
    try {
      const { resp } = await api("GET", "/api/results/" + encodeURIComponent(res.id) + q);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: "drsnow_" + res.design + "_" +
        res.id.slice(0, 8) + (format === "csv" ? "_table" + tableIndex + ".csv" : ".json") });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    } catch (e) {
      setStatus("run-status", e.message, "error");
    }
  }

  function diagnosticCard(d) {
    const stat = "Statistic" + (d.dof.length ? " (" + d.dof.map(fmt).join(", ") + ")" : "") +
      " = " + fmt(d.statistic) + ", p-value = " + fmtP(d.pvalue);
    return el("div", { className: "diagnostic" }, [
      el("h4", { text: d.name }),
      el("p", null, [el("strong", { text: "H₀: " }), d.null]),
      d.method ? el("p", { className: "small", text: "Method: " + d.method }) : null,
      el("p", { text: stat }),
      el("p", { className: "verdict", text: d.verdict }),
      d.note ? el("p", { className: "small note", text: d.note }) : null
    ]);
  }

  function renderResult(res) {
    const card = el("article", { className: "result", "aria-labelledby": "h-" + res.id });
    card.appendChild(el("h3", { id: "h-" + res.id, text: res.title }));
    const meta = [];
    if (res.estimand) meta.push("Estimand: " + res.estimand);
    if (res.nobs !== null && res.nobs !== undefined) meta.push("Observations: " + res.nobs);
    card.appendChild(el("p", { className: "meta", text: meta.join(" · ") }));
    const specItems = Object.entries(res.spec).map(function ([k, v]) {
      return k + " = " + (Array.isArray(v) ? "[" + v.join(", ") + "]" : String(v));
    });
    card.appendChild(el("details", null, [el("summary", { text: "Specification" }),
      el("p", { className: "small mono", text: specItems.join("; ") })]));

    if (res.warnings.length) {
      card.appendChild(el("div", { className: "warnings", role: "note" }, [
        el("h4", { text: "Warnings from the estimator" }),
        el("ul", null, res.warnings.map(function (w) { return el("li", { text: w }); }))]));
    }
    if (res.plot) drawPlot(card, res.plot);
    res.tables.forEach(function (t, i) {
      const block = el("div", { className: "table-block" });
      block.appendChild(el("h4", { text: t.title }));
      block.appendChild(renderTable(t, t.title));
      if (t.note) block.appendChild(el("p", { className: "hint small", text: t.note }));
      const btn = el("button", { type: "button", className: "secondary",
                                 text: "Download CSV" });
      btn.addEventListener("click", function () { download(res, "csv", i + 1); });
      block.appendChild(btn);
      card.appendChild(block);
    });
    if (res.diagnostics.length) {
      card.appendChild(el("h4", { text: "Diagnostics" }));
      for (const d of res.diagnostics) card.appendChild(diagnosticCard(d));
    }
    for (const b of res.text_blocks) {
      card.appendChild(el("h4", { text: b.title }));
      card.appendChild(el("pre", { text: b.text }));
    }
    if (res.notes.length) {
      card.appendChild(el("ul", { className: "notes" },
        res.notes.map(function (n) { return el("li", { text: n }); })));
    }
    card.appendChild(el("details", null, [el("summary", { text: "Full text output" }),
      el("pre", { text: res.summary })]));
    const json = el("button", { type: "button", className: "secondary",
                                text: "Download all results (JSON)" });
    json.addEventListener("click", function () { download(res, "json"); });
    card.appendChild(json);
    const box = byId("results");
    box.insertBefore(card, box.firstChild);
    byId("results-panel").hidden = false;
  }

  // ---------------------------------------------------------------- events
  async function onUpload(ev) {
    ev.preventDefault();
    const file = byId("file").files[0];
    if (!file) return;
    if (limits && file.size > limits.max_upload_bytes) {
      setStatus("upload-status", "The file is larger than the upload limit.", "error");
      return;
    }
    setStatus("upload-status", "Uploading…");
    byId("upload-btn").disabled = true;
    try {
      const delim = byId("delim").value;
      const { data } = await api("POST", "/api/upload?delim=" + encodeURIComponent(delim),
        file, { "Content-Type": "text/csv", "X-Filename": file.name.replace(/[^\w .\-()]/g, "_") });
      columns = data.columns;
      renderPreview(data);
      renderDesignFields();
      byId("design-panel").hidden = false;
      setStatus("upload-status", "Loaded " + data.n_rows.toLocaleString() + " rows.", "ok");
    } catch (e) {
      setStatus("upload-status", e.message, "error");
    } finally {
      byId("upload-btn").disabled = false;
    }
  }

  async function onRun(ev) {
    ev.preventDefault();
    const design = byId("design").value;
    const body = JSON.stringify({ design: design, params: collectParams() });
    setStatus("run-status", "Running… (bootstrap and placebo procedures can take a while)");
    byId("run-btn").disabled = true;
    try {
      const { data } = await api("POST", "/api/analyze", body,
                                 { "Content-Type": "application/json" });
      renderResult(data);
      setStatus("run-status", "Done.", "ok");
    } catch (e) {
      setStatus("run-status", e.message, "error");
    } finally {
      byId("run-btn").disabled = false;
    }
  }

  function init() {
    const sel = byId("design");
    for (const [k, d] of Object.entries(DESIGNS)) sel.appendChild(el("option", { value: k, text: d.label }));
    sel.addEventListener("change", renderDesignFields);
    byId("upload-form").addEventListener("submit", onUpload);
    byId("analysis-form").addEventListener("submit", onRun);
    startSession().catch(function (e) { setStatus("session-status", e.message, "error"); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
