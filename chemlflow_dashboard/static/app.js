/* Renders the Agent Mission Control layout from the CheMLFlow artifact
   snapshot. Every value on screen is derived from /api/v1/snapshot; panes
   with no artifact backing (agent decisions) render explicit empty states
   rather than placeholder content. */

const byId = (id) => document.getElementById(id);

const STATUS_ORDER = ["completed", "running", "stale", "failed", "queued"];

const view = {
  snapshot: null,
  filter: "all",
  sortKey: "case_id",
  sortDesc: false,
  selectedCaseId: null,
  traceTab: "decisions",
  lastDetailRefresh: 0,
  elapsedBase: null,
  elapsedAnchor: 0,
  elapsedTicking: false,
};

/* ── helpers ─────────────────────────────────────────────────────────── */

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function show(node, visible) {
  if (node) node.classList.toggle("hidden", !visible);
}

/* CSP blocks inline style attributes, so dynamic geometry goes through
   CSSOM, which the policy does not govern. */
function setWidth(node, percent) {
  node.style.setProperty("width", percent);
}

function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.floor(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${secs}s`;
  return `${secs}s`;
}

function fmtClock(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.floor(Number(seconds)));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function fmtBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function fmtTime(value) {
  const parsed = Date.parse(value || "");
  if (!Number.isFinite(parsed)) return "—";
  return new Date(parsed).toLocaleTimeString([], { hour12: false });
}

function humanize(value) {
  return String(value ?? "—").replaceAll("_", " ");
}

function shortPath(value) {
  const text = String(value || "");
  const parts = text.split("/");
  return parts.length > 3 ? `…/${parts.slice(-3).join("/")}` : text;
}

function inputLabel(value) {
  const key = String(value || "").trim();
  const known = {
    smiles_native: "Native SMILES",
    "featurize.none": "Curated features",
    "featurize.rdkit": "RDKit2D",
    "featurize.rdkit_labeled": "RDKit2D",
    "featurize.morgan": "Morgan",
    "featurize.ecfp4_rdkit": "ECFP4 + RDKit2D",
    "featurize.chemeleon_fp": "CheMeleonFP",
  };
  return known[key] || key || "native";
}

function featureInputLabel(item) {
  const explicit = String(item?.feature_input || "").trim();
  if (explicit) return inputLabel(explicit);
  const nodes = item?.progress?.pipeline?.nodes || [];
  if (nodes.includes("featurize.morgan")) return "Morgan";
  if (nodes.includes("featurize.rdkit") || nodes.includes("featurize.rdkit_labeled")) return "RDKit";
  if (nodes.includes("featurize.ecfp4_rdkit")) return "ECFP4 RDKit";
  if (nodes.includes("featurize.none")) return "native";
  return "native";
}

function statusTag(status) {
  return make("span", `tag tag-${status}`, humanize(status));
}

function isDoe(snapshot) {
  return snapshot.mode === "doe";
}

function primaryCase(snapshot) {
  return (snapshot.cases || [])[0] || null;
}

function modelLabel(value) {
  const key = String(value || "").trim();
  const known = {
    svm: "SVM",
    xgboost: "XGBoost",
    random_forest: "Random forest",
    decision_tree: "Decision tree",
    chemprop: "Chemprop",
    chemeleon: "CheMeleon",
    tabpfn: "TabPFN 2.6",
    ensemble: "Ensemble",
    dl_simple: "DL simple",
    dl_deep: "DL deep",
    dl_gru: "DL GRU",
    dl_resmlp: "DL ResMLP",
    dl_tabtransformer: "DL TabTransformer",
    dl_aereg: "DL AE reg",
  };
  if (known[key]) return known[key];
  return humanize(key).replace(/\b\w/g, (char) => char.toUpperCase());
}

function titleCaseWords(value) {
  return String(value || "")
    .replace(/\bpgp\b/gi, "PGP")
    .replace(/\bsvm\b/gi, "SVM")
    .replace(/\bcv\b/gi, "CV")
    .replace(/\bdoe\b/gi, "DOE")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function inferredDatasetLabel(snapshot) {
  const studyName = String(snapshot.study?.name || "");
  const sourcePath = String(snapshot.study?.source_path || "");
  const firstConfig = String(primaryCase(snapshot)?.config_path || "");
  const haystack = `${studyName} ${sourcePath} ${firstConfig}`.toLowerCase();
  if (haystack.includes("pgp")) return "PGP Broccatelli";
  if (haystack.includes("qm9")) return "QM9";
  if (haystack.includes("urease")) return "Urease";
  if (haystack.includes("ysi")) return "YSI";
  if (haystack.includes("pah")) return "PAH";
  const stem = studyName
    .replace(/[_-]+/g, " ")
    .replace(/\bcv\b.*$/i, "")
    .replace(/\bfold\s*\d+\b/i, "")
    .trim();
  return titleCaseWords(stem || "CheMLFlow");
}

function foldIndexFor(item) {
  if (!item) return null;
  if (item.fold_index !== null && item.fold_index !== undefined) return item.fold_index;
  const metaFold = item.split_meta?.fold_index;
  return metaFold === null || metaFold === undefined ? null : metaFold;
}

function splitLabel(item) {
  if (!item) return "single run";
  const meta = item.split_meta || {};
  const fold = foldIndexFor(item);
  if (meta.mode === "cv") {
    const total = meta.cv?.n_splits;
    if (fold !== null) return `CV fold ${fold}${total ? ` of ${total}` : ""}`;
    return total ? `${total}-fold CV case` : "CV case";
  }
  if (meta.mode === "holdout") return "holdout test split";
  if (meta.mode) return `${humanize(meta.mode)} split`;
  return "single run";
}

function displayName(snapshot) {
  if (!isDoe(snapshot)) {
    const item = primaryCase(snapshot);
    return `${inferredDatasetLabel(snapshot)} · ${modelLabel(item?.model_type)} · ${splitLabel(item)}`;
  }
  const raw = String(snapshot.study?.name || "").replace(/[_-]+/g, " ").trim();
  const base = inferredDatasetLabel(snapshot);
  return raw.toLowerCase().includes("doe") ? titleCaseWords(raw) : `${base} DOE`;
}

function metricScopeLabel(snapshot) {
  const metric = String(snapshot.study?.primary_metric || "metric").toUpperCase();
  if (isDoe(snapshot)) return `Best mean ${metric}`;
  const item = primaryCase(snapshot);
  const mode = item?.split_meta?.mode;
  if (mode === "cv") return `Fold ${metric}`;
  if (mode === "holdout") return `Test ${metric}`;
  return `Run ${metric}`;
}

/* ── sidebar + header ────────────────────────────────────────────────── */

function studyPhase(snapshot) {
  const launcher = snapshot.launcher || {};
  const status = String(launcher.status || "").toLowerCase();
  if (status === "running") return "running";
  if (status === "failed") return "failed";
  if (status === "completed") return snapshot.summary.failed > 0 ? "failed" : "completed";
  if (snapshot.summary.running + snapshot.summary.stale > 0) return "running";
  if (snapshot.summary.settled_cases > 0 && snapshot.summary.settled_cases === snapshot.summary.valid_cases) {
    return snapshot.summary.failed > 0 ? "failed" : "completed";
  }
  return "idle";
}

const PHASE_LABEL = {
  running: "EXECUTING",
  completed: "COMPLETED",
  failed: "FAILED",
  idle: "IDLE",
};

function artifactStudyClock(snapshot, phase) {
  const timeline = snapshot.timeline || {};
  const started = Date.parse(timeline.started_at || "");
  if (!Number.isFinite(started)) return null;

  const ended = Date.parse(timeline.ended_at || "");
  const ticking = phase === "running";
  const endpoint = ticking || !Number.isFinite(ended) ? Date.now() : ended;
  return {
    base: Math.max(0, (endpoint - started) / 1000),
    ticking,
  };
}

function renderSidebar(snapshot) {
  const { study, summary } = snapshot;
  const phase = studyPhase(snapshot);
  const name = displayName(snapshot);
  byId("sideCollectionLabel").textContent = isDoe(snapshot) ? "Studies" : "Runs";
  byId("sideStudyName").textContent = name;
  byId("sideStudyName").title = name;
  const status = byId("sideStudyStatus");
  status.className = `study-status ${phase}`;
  status.textContent = phase === "running" ? "running" : phase;
  const unit = isDoe(snapshot) ? "case" : "run";
  const sub = `${summary.valid_cases} ${unit}${summary.valid_cases === 1 ? "" : "s"} · ${study.primary_metric.toUpperCase()}`;
  byId("sideStudySub").textContent = sub;
  byId("sideStudySub").title = sub;
  byId("sourcePrimary").textContent = snapshot.mode === "doe"
    ? "reads execution_manifest.jsonl"
    : "reads run_status.json";
  byId("sourceSecondary").textContent = snapshot.mode === "doe"
    ? "+ per-case run_status.json"
    : "+ progress.json · metrics";
}

/* Section standfirsts. A DOE sweeps many cases; a single run has exactly one,
   so the same section means something different in each mode. */
function renderSections(snapshot) {
  const doe = isDoe(snapshot);
  const notes = {
    statusNote: doe
      ? "Progress, best result, and health across every case."
      : "Progress, result, and health for this run.",
    resultsNote: doe
      ? "Mean performance and per-fold outcomes."
      : "Metric outcome and per-split detail.",
    executionsNote: doe
      ? "Every valid case — filter by status, sort any column."
      : "The run artifact and where its wall time went.",
    integrityNote: doe
      ? "Split construction and DOE compatibility checks."
      : "How the split was constructed.",
  };
  Object.entries(notes).forEach(([id, text]) => {
    const node = byId(id);
    if (node) node.textContent = text;
  });
}

function renderHeader(snapshot) {
  const { study, launcher } = snapshot;
  const phase = studyPhase(snapshot);
  byId("studyKicker").textContent = `${isDoe(snapshot) ? "DOE study" : "Single config run"} · ${study.task_type} · ${phase}`;
  byId("studyName").textContent = displayName(snapshot);
  byId("configPath").textContent = shortPath(study.source_path);
  byId("configPath").title = study.source_path;

  const pill = byId("statusPill");
  pill.className = `status-pill ${phase}`;
  byId("statusLabel").textContent = PHASE_LABEL[phase] || "IDLE";

  byId("pauseButton").title = "CheMLFlow does not expose an agent control channel; the dashboard is read-only.";

  const started = Date.parse(launcher?.started_at || "");
  const ended = Date.parse(launcher?.ended_at || "");
  if (Number.isFinite(started)) {
    const running = phase === "running" || !Number.isFinite(ended);
    view.elapsedBase = ((Number.isFinite(ended) && !running ? ended : Date.now()) - started) / 1000;
    view.elapsedAnchor = Date.now();
    view.elapsedTicking = running;
  } else {
    const artifactClock = artifactStudyClock(snapshot, phase);
    view.elapsedBase = artifactClock?.base ?? snapshot.summary.observed_wall_seconds;
    view.elapsedAnchor = Date.now();
    view.elapsedTicking = artifactClock?.ticking ?? false;
  }
  paintElapsed();
}

function paintElapsed() {
  if (view.elapsedBase === null || view.elapsedBase === undefined) {
    byId("elapsedLabel").textContent = "—";
    return;
  }
  const drift = view.elapsedTicking ? (Date.now() - view.elapsedAnchor) / 1000 : 0;
  byId("elapsedLabel").textContent = fmtClock(view.elapsedBase + drift);
}

/* ── KPI row ─────────────────────────────────────────────────────────── */

function renderKpis(snapshot) {
  const { study, summary, leaderboard, skips } = snapshot;
  const unit = isDoe(snapshot) ? "case" : "run";

  byId("doneCount").textContent = summary.settled_cases;
  byId("totalCount").textContent = `/ ${summary.valid_cases} ${unit}${summary.valid_cases === 1 ? "" : "s"}`;
  const total = summary.valid_cases || 1;
  setWidth(byId("barDone"), `${(summary.completed / total) * 100}%`);
  setWidth(byId("barFailed"), `${(summary.failed / total) * 100}%`);
  byId("progressSub").textContent = `${summary.running + summary.stale} active · ${summary.queued} queued`;

  const best = leaderboard[0];
  byId("bestKicker").textContent = metricScopeLabel(snapshot);
  byId("bestMetric").textContent = best ? fmtNumber(best.metric_mean) : "—";
  const single = primaryCase(snapshot);
  byId("bestModel").textContent = best
    ? isDoe(snapshot)
      ? `${modelLabel(best.model_type)} · ${inputLabel(best.feature_input)}`
      : `${modelLabel(single?.model_type || best.model_type)} · ${splitLabel(single)}`
    : isDoe(snapshot)
      ? "No parent has completed every fold"
      : "Run has not reported the primary metric";

  const spread = best ? Number(best.metric_std) : Number.NaN;
  byId("bestDelta").textContent = isDoe(snapshot) && Number.isFinite(spread) ? `± ${spread.toFixed(4)} population SD` : "";
  byId("bestSub").textContent = best
    ? isDoe(snapshot)
      ? `${best.total_folds} fold${best.total_folds === 1 ? "" : "s"} · ${best.provisional ? "provisional rank" : "final rank"}`
      : `${humanize(single?.status || best.status)} · single config result`
    : "—";

  byId("failedCount").textContent = summary.failed;
  byId("skippedCount").textContent = `+ ${skips.count} skipped`;
  const topReason = Object.entries(skips.by_reason || {}).sort((a, b) => b[1] - a[1])[0];
  byId("failReason").textContent = summary.failed
    ? `${summary.failed} case${summary.failed === 1 ? "" : "s"} exited non-zero`
    : topReason
      ? `Top skip code ${topReason[0]}`
      : "No failures or skips observed";

  byId("wallValue").textContent = fmtDuration(summary.observed_wall_seconds);
  byId("wallUnit").textContent = isDoe(snapshot) ? "longest case" : "run wall";
  byId("workerLine").textContent = `${summary.active_workers} active local worker${summary.active_workers === 1 ? "" : "s"}`;

  byId("decisionCount").textContent = "0";
  byId("decisionLine").textContent = "no decisions recorded";
}

/* The approval queue has no artifact behind it yet; the card stays hidden
   until CheMLFlow emits one. */
function renderProposal() {
  show(byId("proposalCard"), false);
}

/* ── execution table ─────────────────────────────────────────────────── */

function renderChips(snapshot) {
  const target = byId("filterChips");
  clear(target);
  const counts = { all: snapshot.cases.length };
  STATUS_ORDER.forEach((status) => {
    counts[status] = snapshot.cases.filter((item) => item.status === status).length;
  });
  const labels = {
    all: "All",
    completed: "Done",
    running: "Running",
    failed: "Failed",
    stale: "Stale",
    queued: "Queued",
  };
  ["all", ...STATUS_ORDER].forEach((key) => {
    const chip = make("button", `chip${view.filter === key ? " active" : ""}`);
    chip.type = "button";
    chip.append(document.createTextNode(labels[key]), make("b", "", counts[key]));
    chip.addEventListener("click", () => {
      view.filter = key;
      renderChips(view.snapshot);
      renderTable(view.snapshot);
    });
    target.append(chip);
  });
}

function sortedRows(snapshot) {
  const rows = snapshot.cases.filter((item) => view.filter === "all" || item.status === view.filter);
  const key = view.sortKey;
  return rows.sort((a, b) => {
    let left = a[key];
    let right = b[key];
    if (key === "case_id") {
      const result = String(left).localeCompare(String(right), undefined, { numeric: true });
      return view.sortDesc ? -result : result;
    }
    left = Number.isFinite(Number(left)) ? Number(left) : -Infinity;
    right = Number.isFinite(Number(right)) ? Number(right) : -Infinity;
    return view.sortDesc ? right - left : left - right;
  });
}

function renderSortArrows() {
  document.querySelectorAll("[data-arrow]").forEach((node) => {
    const key = node.dataset.arrow;
    node.textContent = view.sortKey === key ? (view.sortDesc ? "↓" : "↑") : "";
  });
}

function renderTable(snapshot) {
  const body = byId("executionRows");
  clear(body);
  const rows = sortedRows(snapshot);
  show(byId("executionEmpty"), !rows.length);
  byId("casesTitle").textContent = isDoe(snapshot) ? "Execution cases" : "Run artifact";
  byId("caseHeader").textContent = isDoe(snapshot) ? "Case" : "Run";
  byId("metricHeader").textContent = snapshot.study.primary_metric.toUpperCase();
  renderSortArrows();

  const leader = snapshot.leaderboard[0];
  rows.forEach((item) => {
    const row = make("button", "row-grid exec-row");
    row.type = "button";
    row.setAttribute("aria-label", `Inspect ${item.case_id}`);
    row.append(make("div", "c-id", item.case_id));
    row.append(make("div", "c-model", modelLabel(item.model_type)));
    row.append(make("div", "c-feat", featureInputLabel(item)));
    const fold = foldIndexFor(item);
    row.append(make("div", "c-fold", fold === null ? "—" : fold));
    row.append(make("div", "c-rep", item.repeat_index === null || item.repeat_index === undefined ? "—" : item.repeat_index));
    const status = make("div");
    status.append(statusTag(item.status));
    row.append(status);
    const isLeader = leader && item.parent_case_id === leader.parent_case_id;
    row.append(make("div", `c-metric${isLeader ? " lead" : ""}`, fmtNumber(item.metric_value)));
    row.append(make("div", "c-rt", fmtDuration(item.elapsed_seconds)));
    row.append(make("div", "c-chev", "›"));
    row.title = item.status_reason || item.case_id;
    row.addEventListener("click", () => openCase(item.case_id));
    body.append(row);
  });
}

/* ── candidate comparison ────────────────────────────────────────────── */

function renderComparison(snapshot) {
  const target = byId("comparisonRows");
  const legend = byId("comparisonLegend");
  clear(target);
  clear(legend);
  const parents = snapshot.parents || [];
  const metric = snapshot.study.primary_metric.toUpperCase();
  const ranking = new Map((snapshot.leaderboard || []).map((item) => [item.parent_case_id, item]));
  const single = primaryCase(snapshot);
  byId("comparisonKicker").textContent = isDoe(snapshot)
    ? `Candidate comparison — mean ${metric} and fold outcomes`
    : `Run result — ${metric} and split outcomes`;
  byId("comparisonMetricHeader").textContent = `Mean ${metric}`;
  byId("comparisonRef").textContent = isDoe(snapshot)
    ? `${parents.length} candidate${parents.length === 1 ? "" : "s"}`
    : "single config";
  byId("comparisonEmpty").textContent = isDoe(snapshot)
    ? "No parent/fold groups are available."
    : "No split group is available for this run.";
  show(byId("comparisonEmpty"), !parents.length);
  show(target, Boolean(parents.length));
  show(legend, Boolean(parents.length));
  if (!parents.length) return;

  const maximize = snapshot.study.optimize !== "min" && snapshot.study.optimize !== "minimize";
  const values = parents
    .flatMap((parent) => parent.folds.map((fold) => fold.metric_value))
    .filter((value) => Number.isFinite(Number(value)))
    .map(Number);
  const lo = values.length ? Math.min(...values) : 0;
  const hi = values.length ? Math.max(...values) : 1;
  const orderedParents = parents.slice().sort((left, right) => {
    const leftRank = ranking.get(left.parent_case_id)?.rank ?? Number.POSITIVE_INFINITY;
    const rightRank = ranking.get(right.parent_case_id)?.rank ?? Number.POSITIVE_INFINITY;
    if (leftRank !== rightRank) return leftRank - rightRank;
    return String(left.parent_case_id).localeCompare(String(right.parent_case_id));
  });

  orderedParents.forEach((parent) => {
    const row = make("div", "comparison-row");
    const leader = ranking.get(parent.parent_case_id);
    const candidate = make("div", "comparison-candidate");
    const rank = make("span", "comparison-rank", leader ? String(leader.rank).padStart(2, "0") : "—");
    const labelText = isDoe(snapshot)
      ? `${modelLabel(parent.model_type)} · ${inputLabel(parent.feature_input)}`
      : `${modelLabel(parent.model_type)} · ${splitLabel(single)}`;
    const label = make("span", "comparison-label", labelText);
    label.title = parent.parent_case_id;
    const foldDetail = make("span", "comparison-detail", `${parent.completed_folds}/${parent.total_folds} folds`);
    candidate.append(rank, label, foldDetail);

    const cells = make("div", "comparison-folds");
    parent.folds.forEach((fold, index) => {
      const cell = make("button", "cell");
      cell.type = "button";
      const foldLabel = isDoe(snapshot)
        ? (fold.fold_index === null || fold.fold_index === undefined ? index : fold.fold_index)
        : (foldIndexFor(single) ?? index);
      const value = Number(fold.metric_value);
      if (fold.status === "completed" && Number.isFinite(value)) {
        const span = hi - lo;
        let t = span > 0 ? (value - lo) / span : 0.75;
        if (!maximize) t = 1 - t;
        const percent = 18 + t * 72;
        cell.style.setProperty(
          "background",
          `color-mix(in srgb, var(--color-accent) ${percent.toFixed(1)}%, var(--color-accent-100))`
        );
      } else {
        cell.classList.add(fold.status);
      }
      cell.title = `${fold.case_id} · fold ${foldLabel} · ${fold.status}${Number.isFinite(value) ? ` · ${fmtNumber(value)}` : ""}`;
      cell.setAttribute("aria-label", cell.title);
      cell.addEventListener("click", () => openCase(fold.case_id));
      cells.append(cell);
    });

    const score = make("div", "comparison-score");
    score.append(make(
      "strong",
      "comparison-mean",
      parent.metric_mean === null || parent.metric_mean === undefined
        ? `${parent.completed_folds}/${parent.total_folds}`
        : fmtNumber(parent.metric_mean)
    ));
    if (parent.metric_std !== null && parent.metric_std !== undefined) {
      score.append(make("span", "comparison-spread", `± ${fmtNumber(parent.metric_std)}`));
    }
    row.append(candidate, cells, score);
    target.append(row);
  });

  const legendItems = [
    ["strong", "best", "color-mix(in srgb, var(--color-accent) 90%, var(--color-accent-100))"],
    ["weak", "weakest", "color-mix(in srgb, var(--color-accent) 18%, var(--color-accent-100))"],
    ["running", "running", "var(--color-accent-300)"],
    ["failed", "failed", "var(--color-fail)"],
    ["queued", "queued", "var(--color-neutral-200)"],
  ];
  legendItems.forEach(([, label, color]) => {
    const item = make("span");
    const swatch = make("i");
    swatch.style.setProperty("background", color);
    item.append(swatch, document.createTextNode(label));
    legend.append(item);
  });
}

/* ── wall time by model ──────────────────────────────────────────────── */

function renderCompute(snapshot) {
  const target = byId("computeBars");
  clear(target);
  const totals = new Map();
  snapshot.cases.forEach((item) => {
    const seconds = Number(item.elapsed_seconds);
    if (!Number.isFinite(seconds) || seconds <= 0) return;
    const key = item.model_type || "unknown";
    totals.set(key, (totals.get(key) || 0) + seconds);
  });
  // A model that never ran contributes no wall time; listing it as 0s is noise.
  const rows = [...totals.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  show(byId("computeEmpty"), !rows.length);
  if (!rows.length) return;

  const hi = rows[0][1];
  rows.forEach(([key, seconds]) => {
    const row = make("div", "bar-row");
    const label = modelLabel(key);
    const name = make("div", "label", label);
    name.title = label;
    const track = make("div", "track");
    const fill = make("i");
    setWidth(fill, `${hi > 0 ? (seconds / hi) * 100 : 0}%`);
    track.append(fill);
    row.append(name, track, make("div", "value", fmtDuration(seconds)));
    target.append(row);
  });
}

/* ── split integrity ─────────────────────────────────────────────────── */

function pushCheck(target, state, label, value) {
  const row = make("div", "check");
  const marks = { ok: "✓", warn: "!", none: "–" };
  row.append(
    make("span", `mark ${state}`, marks[state]),
    make("span", "label", label),
    make("span", "value", value)
  );
  row.lastChild.title = String(value);
  target.append(row);
}

function renderIntegrity(snapshot) {
  const target = byId("integrityChecks");
  clear(target);
  const withMeta = snapshot.cases.find((item) => item.split_meta && Object.keys(item.split_meta).length);
  const meta = withMeta?.split_meta;
  if (!meta) {
    byId("splitKicker").textContent = "Split integrity";
    pushCheck(target, "none", "No split_meta.json observed yet", "—");
    return;
  }

  byId("splitKicker").textContent = `Split integrity — ${[meta.mode, meta.strategy].filter(Boolean).join(" ")}`;

  const coverage = meta.coverage || {};
  const sizes = meta.sizes || {};
  const assigned = Number(coverage.assigned_fraction);
  pushCheck(
    target,
    assigned === 1 ? "ok" : "warn",
    "Rows assigned to a split",
    `${coverage.assigned_rows ?? "—"} / ${coverage.curated_rows ?? "—"}`
  );
  pushCheck(
    target,
    Number(coverage.dropped_before_split) === 0 ? "ok" : "warn",
    "Dropped before split",
    `${coverage.dropped_before_split ?? "—"} rows`
  );
  pushCheck(
    target,
    meta.require_disjoint ? "ok" : "warn",
    "Disjoint splits required",
    meta.require_disjoint ? "enforced" : "not enforced"
  );
  if (sizes.train !== undefined) {
    pushCheck(target, "ok", "Train / val / test", `${sizes.train} / ${sizes.val ?? "—"} / ${sizes.test ?? "—"}`);
  }
  pushCheck(
    target,
    meta.stratify ? "ok" : "none",
    "Stratified",
    meta.stratify ? String(meta.stratify_column || "on") : "off"
  );
  const dupes = Number(meta.retained_duplicate_feature_label_rows);
  if (Number.isFinite(dupes)) {
    pushCheck(target, dupes === 0 ? "ok" : "warn", "Duplicate feature+label rows", `${dupes} retained`);
  }
  if (meta.cv && meta.cv.n_splits) {
    pushCheck(target, "ok", "CV plan", `k=${meta.cv.n_splits} · r=${meta.cv.repeats ?? 1} · seed ${meta.cv.random_state ?? "—"}`);
  }
}

/* ── compatibility audit ─────────────────────────────────────────────── */

function renderSkips(snapshot) {
  const target = byId("skipReasons");
  clear(target);
  const reasons = Object.entries(snapshot.skips.by_reason || {});
  show(byId("skipEmpty"), !reasons.length);
  reasons.forEach(([code, count]) => {
    const pill = make("div", "skip-pill");
    pill.append(make("b", "", count), make("span", "", code));
    target.append(pill);
  });
}

/* ── reasoning trail ─────────────────────────────────────────────────── */

function activityEvents(snapshot) {
  const events = [];
  const launcher = snapshot.launcher || {};
  if (launcher.started_at) {
    events.push([launcher.started_at, `launcher started · mode ${launcher.mode || "run"}`]);
  }
  snapshot.cases.forEach((item) => {
    if (item.start_time) events.push([item.start_time, `${item.case_id} started · ${modelLabel(item.model_type)}`]);
    const scopes = item.progress?.training?.scopes || {};
    Object.entries(scopes).forEach(([name, scope]) => {
      if (scope.start_time) events.push([scope.start_time, `${item.case_id} ${name} ${scope.phase || "phase"} started`]);
      if (scope.end_time) events.push([scope.end_time, `${item.case_id} ${scope.message || `${name} completed`}`]);
    });
    const nodes = item.progress?.pipeline?.completed_nodes || [];
    if (item.end_time) {
      const value = Number.isFinite(Number(item.metric_value)) ? ` · ${item.metric_name} ${fmtNumber(item.metric_value)}` : "";
      events.push([item.end_time, `${item.case_id} ${item.status}${value} · ${nodes.length} nodes`]);
    }
  });
  if (launcher.ended_at) {
    events.push([launcher.ended_at, launcher.message || `launcher ${launcher.status || "finished"}`]);
  }
  return events
    .filter(([time]) => Number.isFinite(Date.parse(time)))
    .sort((a, b) => Date.parse(a[0]) - Date.parse(b[0]));
}

function renderTrace(snapshot) {
  const decisions = byId("tracePanelDecisions");
  clear(decisions);
  const note = make("div", "empty-state tight");
  note.append(
    make("strong", "", "No agent decisions recorded."),
    make("div", "", "CheMLFlow artifacts do not emit a decision log, so there is nothing to attribute a choice to. This pane populates once a decision record exists.")
  );
  decisions.append(note);

  const log = byId("tracePanelLog");
  clear(log);
  const events = activityEvents(snapshot);
  if (!events.length) {
    log.append(make("div", "empty-state", "No timestamped events in the current artifacts."));
    return;
  }
  events.forEach(([time, text]) => {
    const line = make("div", "log-line");
    line.append(make("span", "t", fmtTime(time)), make("span", "msg", text));
    log.append(line);
  });
}

function setTraceTab(tab) {
  view.traceTab = tab;
  byId("tabDecisions").classList.toggle("active", tab === "decisions");
  byId("tabLog").classList.toggle("active", tab === "log");
  show(byId("tracePanelDecisions"), tab === "decisions");
  show(byId("tracePanelLog"), tab === "log");
}

function openTrace() {
  if (view.snapshot) renderTrace(view.snapshot);
  setTraceTab(view.traceTab);
  show(byId("traceRoot"), true);
}

function closeTrace() {
  show(byId("traceRoot"), false);
}

/* ── case drawer ─────────────────────────────────────────────────────── */

function section(label) {
  const wrap = make("div");
  wrap.append(make("div", "drawer-section-label", label));
  return wrap;
}

async function refreshCase(caseId) {
  view.lastDetailRefresh = Date.now();
  const body = byId("caseBody");
  try {
    const response = await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/detail`, { cache: "no-store" });
    if (!response.ok) throw new Error(`detail ${response.status}`);
    const detail = await response.json();
    if (view.selectedCaseId !== caseId) return;

    byId("caseId").textContent = detail.case_id;
    const statusTagNode = byId("caseStatus");
    statusTagNode.className = `tag tag-${detail.status}`;
    statusTagNode.textContent = humanize(detail.status);
    const snapshotCase = view.snapshot?.cases?.find((item) => item.case_id === caseId);
    byId("caseModel").textContent = view.snapshot && !isDoe(view.snapshot)
      ? `${modelLabel(detail.model_type)} · ${splitLabel(snapshotCase)}`
      : `${modelLabel(detail.model_type)} · ${inputLabel(detail.feature_input)}`;
    byId("casePath").textContent = shortPath(detail.config_path);
    byId("casePath").title = detail.config_path;

    clear(body);

    const numericMetrics = Object.entries(detail.metrics || {})
      .filter(([, value]) => Number.isFinite(Number(value)))
      .slice(0, 6);
    if (numericMetrics.length) {
      const metricSection = section("Metrics");
      const grid = make("div", "metric-grid");
      numericMetrics.forEach(([key, value]) => {
        const cell = make("div");
        cell.append(make("div", "k", key), make("div", "v", fmtNumber(value, 4)));
        grid.append(cell);
      });
      metricSection.append(grid);
      body.append(metricSection);
    }

    const runSection = section("Execution");
    const runList = make("div", "kv");
    const runRows = [
      ["status", humanize(detail.status)],
      ["elapsed", fmtDuration(detail.elapsed_seconds)],
      ["fold / repeat", `${detail.fold_index ?? detail.split_meta?.fold_index ?? "—"} / ${detail.repeat_index ?? detail.split_meta?.repeat_index ?? "—"}`],
      ["heartbeat age", detail.freshness_seconds === null || detail.freshness_seconds === undefined ? "—" : `${Math.round(detail.freshness_seconds)}s`],
      ["pipeline", `${detail.progress?.pipeline?.completed ?? 0}/${detail.progress?.pipeline?.total ?? 0} nodes`],
      ["run dir", shortPath(detail.run_dir)],
    ];
    runRows.forEach(([key, value]) => {
      const row = make("div", "kv-row");
      row.append(make("span", "k", key), make("span", "v", value));
      runList.append(row);
    });
    runSection.append(runList);
    body.append(runSection);

    const meta = detail.split_meta || {};
    if (Object.keys(meta).length) {
      const splitSection = section("split_meta.json");
      const list = make("div", "kv");
      const keys = ["mode", "strategy", "fold_index", "repeat_index", "dataset_rows", "plan_id", "random_state", "stratify"];
      keys.forEach((key) => {
        if (meta[key] === undefined) return;
        const row = make("div", "kv-row");
        row.append(make("span", "k", key), make("span", "v", String(meta[key])));
        list.append(row);
      });
      splitSection.append(list);
      body.append(splitSection);
    }

    const artifacts = detail.artifacts || [];
    const artifactSection = section(`Artifacts · ${artifacts.length}`);
    if (!artifacts.length) {
      artifactSection.append(make("div", "empty-state tight", "No artifacts observed yet."));
    } else {
      const chips = make("div", "artifact-chips");
      artifacts.slice(0, 40).forEach((artifact) => {
        const chip = make("span", "", artifact.name);
        chip.title = `${artifact.name} · ${fmtBytes(artifact.size_bytes)}`;
        chips.append(chip);
      });
      artifactSection.append(chips);
    }
    body.append(artifactSection);

    const logSection = section("Log tail");
    const pre = make("pre", "block", "Loading log…");
    logSection.append(pre);
    body.append(logSection);
    try {
      const logResponse = await fetch(`/api/v1/cases/${encodeURIComponent(caseId)}/log?tail=400`, { cache: "no-store" });
      if (!logResponse.ok) throw new Error("Log is not available yet.");
      const payload = await logResponse.json();
      if (view.selectedCaseId === caseId) pre.textContent = payload.text || "Log is empty.";
    } catch (error) {
      pre.textContent = error.message;
    }
  } catch (error) {
    if (view.selectedCaseId === caseId) {
      clear(body);
      body.append(make("div", "empty-state", `Unable to load case detail: ${error.message}`));
    }
  }
}

function openCase(caseId) {
  view.selectedCaseId = caseId;
  show(byId("caseRoot"), true);
  byId("caseId").textContent = caseId;
  clear(byId("caseBody"));
  byId("caseBody").append(make("div", "empty-state", "Loading artifact detail…"));
  refreshCase(caseId);
}

function closeCase() {
  view.selectedCaseId = null;
  show(byId("caseRoot"), false);
}

/* ── connection ──────────────────────────────────────────────────────── */

function setConnection(connected, label) {
  const node = byId("pollState");
  node.textContent = label;
  node.classList.toggle("offline", !connected);
}

function render(snapshot) {
  view.snapshot = snapshot;
  renderSidebar(snapshot);
  renderSections(snapshot);
  renderHeader(snapshot);
  renderKpis(snapshot);
  renderProposal();
  renderChips(snapshot);
  renderTable(snapshot);
  renderComparison(snapshot);
  renderCompute(snapshot);
  renderIntegrity(snapshot);
  renderSkips(snapshot);
  if (!byId("traceRoot").classList.contains("hidden")) renderTrace(snapshot);
  if (view.selectedCaseId && Date.now() - view.lastDetailRefresh > 3000) refreshCase(view.selectedCaseId);
}

async function loadSnapshot() {
  const response = await fetch("/api/v1/snapshot", { cache: "no-store" });
  if (!response.ok) throw new Error(`snapshot ${response.status}`);
  render(await response.json());
}

function connectEvents() {
  const source = new EventSource("/api/v1/events");
  source.addEventListener("open", () => setConnection(true, "live · artifact stream"));
  source.addEventListener("snapshot", (event) => {
    try {
      render(JSON.parse(event.data));
      setConnection(true, "live · artifact stream");
    } catch (_error) {
      setConnection(false, "invalid snapshot");
    }
  });
  source.addEventListener("error", () => setConnection(false, "reconnecting…"));
}

/* ── wiring ──────────────────────────────────────────────────────────── */

byId("traceButton").addEventListener("click", openTrace);
byId("decisionTile").addEventListener("click", openTrace);
byId("traceClose").addEventListener("click", closeTrace);
byId("traceScrim").addEventListener("click", closeTrace);
byId("tabDecisions").addEventListener("click", () => setTraceTab("decisions"));
byId("tabLog").addEventListener("click", () => setTraceTab("log"));

byId("caseClose").addEventListener("click", closeCase);
byId("caseScrim").addEventListener("click", closeCase);

byId("failedTile").addEventListener("click", () => {
  view.filter = view.snapshot && view.snapshot.summary.failed ? "failed" : "all";
  renderChips(view.snapshot);
  renderTable(view.snapshot);
  byId("cases").scrollIntoView({ behavior: "smooth", block: "start" });
});

document.querySelectorAll("[data-sort]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.sort;
    if (view.sortKey === key) view.sortDesc = !view.sortDesc;
    else {
      view.sortKey = key;
      view.sortDesc = key !== "case_id";
    }
    renderTable(view.snapshot);
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (view.selectedCaseId) closeCase();
  else closeTrace();
});

setInterval(paintElapsed, 1000);

loadSnapshot()
  .then(() => setConnection(true, "snapshot loaded"))
  .catch((error) => setConnection(false, error.message));
connectEvents();
