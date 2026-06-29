"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
const esc = (s) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let mode = "text";

// ── Tabs ────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    mode = tab.dataset.mode;
    $("#tab-text").classList.toggle("hidden", mode !== "text");
    $("#tab-url").classList.toggle("hidden", mode !== "url");
  });
});

// ── Health + sources on load ────────────────────────────────────────
async function loadHealth() {
  try {
    const r = await fetch("/api/health");
    const data = await r.json();
    const online = data.coref?.online;
    $("#coref-dot").className = "dot " + (online ? "online" : "offline");
    $("#coref-label").textContent = online
      ? "coref online"
      : "coref offline";
    $("#coref-status").title = data.coref?.url || "";
  } catch {
    $("#coref-dot").className = "dot offline";
    $("#coref-label").textContent = "api unreachable";
  }
}

async function loadSources() {
  try {
    const r = await fetch("/api/sources");
    const { sources } = await r.json();
    const sel = $("#source_name");
    sources.forEach((s) => {
      const o = el("option");
      o.value = s.name;
      o.textContent = `${s.name} (${s.format || "?"})`;
      sel.appendChild(o);
    });
  } catch {
    /* keep default 'manual' */
  }
}

// ── Run ─────────────────────────────────────────────────────────────
$("#run").addEventListener("click", run);

async function run() {
  const btn = $("#run");
  const hint = $("#hint");
  hint.textContent = "";

  const body = {
    mode,
    source_name: $("#source_name").value,
    use_coref: $("#use_coref").checked,
  };
  if (mode === "text") {
    body.text = $("#text").value;
    if (!body.text.trim()) return (hint.textContent = "Please paste some text first.");
  } else {
    body.url = $("#url").value.trim();
    body.text_field = $("#text_field").value.trim() || null;
    if (!body.url) return (hint.textContent = "Please enter a URL first.");
  }

  btn.disabled = true;
  btn.textContent = "Running…";
  $("#steps").innerHTML = '<p class="placeholder">Running pipeline…</p>';
  $("#chunks-wrap").classList.add("hidden");

  try {
    const r = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "Request failed");
    render(data);
  } catch (e) {
    $("#steps").innerHTML = "";
    hint.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "▶ Run pipeline";
  }
}

// ── Render ──────────────────────────────────────────────────────────
function render(data) {
  $("#run-meta").textContent = `doc ${data.document_id.slice(0, 12)}… · ${data.elapsed_ms} ms`;

  const steps = $("#steps");
  steps.innerHTML = "";
  data.steps.forEach((s, i) => steps.appendChild(stepCard(s, i === 0)));

  const cw = $("#chunks-wrap");
  cw.classList.remove("hidden");
  $("#chunk-count").textContent = `(${data.chunks.length})`;
  const chunks = $("#chunks");
  chunks.innerHTML = "";
  data.chunks.forEach((c) => chunks.appendChild(chunkCard(c)));
}

function stepCard(s, open) {
  const card = el("div", "step" + (open ? " open" : ""));
  const head = el("div", "step-head");
  head.appendChild(el("div", "step-num", String(s.id)));
  head.appendChild(el("div", "step-title", esc(s.name)));
  const right = el("div", "step-summary");
  right.appendChild(el("span", "badge " + s.status, s.status));
  right.appendChild(document.createTextNode(" " + (s.summary || "")));
  head.appendChild(right);
  head.addEventListener("click", () => card.classList.toggle("open"));
  card.appendChild(head);

  const body = el("div", "step-body");
  body.innerHTML = renderDetail(s);
  card.appendChild(body);
  return card;
}

function renderDetail(s) {
  let html = "";
  const d = s.detail || {};

  if (d.rewrites && d.rewrites.length) {
    html += d.rewrites
      .map(
        (r) =>
          `<div class="rewrite"><div class="before">− ${esc(r.before)}</div><div class="after">+ ${esc(r.after)}</div></div>`
      )
      .join("");
  }

  if (d.sections) {
    html += '<div class="kv">Sections:</div>';
    html += d.sections
      .map((sec) => `<span class="section-pill">${esc(sec.section)} · ${sec.chars} chars</span>`)
      .join("");
    html += d.sections
      .map((sec) => `<div class="kv"><b>${esc(sec.section)}</b></div><pre>${esc(sec.preview)}</pre>`)
      .join("");
  }

  // generic key/values (skip the ones we rendered specially)
  const skip = new Set(["rewrites", "sections", "preview"]);
  Object.entries(d).forEach(([k, v]) => {
    if (skip.has(k) || v === null || v === undefined || v === "") return;
    if (typeof v === "object") return;
    html += `<div class="kv"><b>${esc(k)}:</b> ${esc(v)}</div>`;
  });

  if (s.content) html += `<pre>${esc(s.content)}</pre>`;
  return html || '<div class="kv">No further detail.</div>';
}

function chunkCard(c) {
  const card = el("div", "chunk");
  const meta = el("div", "chunk-meta");
  meta.innerHTML =
    `<span>#${c.position}</span>` +
    `<span>${esc(c.section)}</span>` +
    `<span>chunk ${c.chunk_index + 1}/${c.total_chunks}</span>` +
    `<span>${esc(c.source_name)}</span>`;
  card.appendChild(meta);
  card.appendChild(el("div", "chunk-text", esc(c.text)));
  return card;
}

loadHealth();
loadSources();
setInterval(loadHealth, 15000);
