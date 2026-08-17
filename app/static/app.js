// Ask My Docs — frontend logic. No framework, no build step.

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  return n;
};

const input = $("#q");
const askBtn = $("#askBtn");
const resultSection = $("#result");
const errorBox = $("#errorBox");

let sampleCorpusLabel = "";

// ---- corpus status -------------------------------------------------------
async function loadStats() {
  try {
    const r = await fetch("/api/stats");
    const s = await r.json();
    sampleCorpusLabel = `${s.documents} docs · ${s.chunks} chunks`;
    $("#corpusText").textContent = sampleCorpusLabel;
    $("#statusDot").className = "dot " + (s.indexed ? "ok" : "bad");
    $("#footerModels").textContent = `${s.embedding_model} + ${s.reranker_model}`;
  } catch {
    $("#corpusText").textContent = "index offline";
    $("#statusDot").className = "dot bad";
  }
}

// ---- upload --------------------------------------------------------------
const fileInput = $("#fileInput");
const uploadBtn = $("#uploadBtn");

uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  errorBox.hidden = true;
  uploadBtn.disabled = true;
  const original = uploadBtn.innerHTML;
  uploadBtn.textContent = "Indexing…";
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Upload failed");
    $("#uploadName").textContent = `${d.doc_id} · ${d.chunks} chunks`;
    $("#uploadActive").hidden = false;
    $("#corpusText").textContent = "your document";
    input.placeholder = "Ask a question about your uploaded document…";
  } catch (err) {
    errorBox.hidden = false;
    $("#errorBody").textContent = err.message;
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.innerHTML = original;
    fileInput.value = "";
  }
});

$("#clearBtn").addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  $("#uploadActive").hidden = true;
  $("#corpusText").textContent = sampleCorpusLabel;
  input.placeholder = "e.g. How many PTO days do I get, and how many carry over?";
});

// ---- rendering -----------------------------------------------------------

// Turn "[1]" markers in the answer into clickable citation chips.
function renderAnswer(container, text, abstained) {
  container.innerHTML = "";
  container.classList.toggle("abstained", abstained);
  const parts = text.split(/(\[\d+\])/g);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const a = el("a", "cite");
      a.textContent = m[1];
      a.href = `#source-${m[1]}`;
      a.setAttribute("aria-label", `Jump to source ${m[1]}`);
      a.addEventListener("click", (e) => {
        e.preventDefault();
        flashSource(m[1]);
      });
      container.appendChild(a);
    } else if (part) {
      container.appendChild(document.createTextNode(part));
    }
  }
}

function flashSource(idx) {
  const target = document.getElementById(`source-${idx}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.add("flash");
  setTimeout(() => target.classList.remove("flash"), 1400);
}

function renderTrace(stages, totalMs) {
  const wrap = $("#traceStages");
  wrap.innerHTML = "";
  const max = Math.max(...stages.map((s) => s.ms), 1);
  const label = { lexical: "lexical", semantic: "semantic", fuse: "fuse", rerank: "rerank", generate: "generate", verify: "verify" };
  for (const s of stages) {
    const stage = el("div", "trace-stage");
    if (s.name === "generate") stage.classList.add("gen");
    if (s.name === "verify") stage.classList.add("verify");

    const name = el("div", "s-name");
    name.textContent = label[s.name] || s.name;

    const bar = el("div", "s-bar");
    const fill = el("span");
    fill.style.width = `${Math.max(4, (s.ms / max) * 100)}%`;
    bar.appendChild(fill);

    const ms = el("div", "s-ms mono");
    ms.textContent = `${s.ms.toFixed(0)}ms`;

    stage.append(name, bar, ms);
    wrap.appendChild(stage);
  }
  $("#traceTotal").textContent = `${totalMs.toFixed(0)}ms total`;
}

function renderSources(sources) {
  const wrap = $("#sources");
  wrap.innerHTML = "";
  for (const s of sources) {
    const card = el("div", "source");
    card.id = `source-${s.index}`;

    const head = el("div", "source-head");
    const idx = el("span", "source-idx mono");
    idx.textContent = s.index;
    const doc = el("span", "source-doc");
    doc.textContent = s.doc_id;
    const chunk = el("span", "source-chunk");
    chunk.textContent = s.chunk_id;
    const tag = el("span", `tag ${s.retrieval}`);
    tag.textContent = s.retrieval;
    const score = el("span", "source-score");
    score.textContent = `rerank ${s.rerank_score.toFixed(2)}`;

    head.append(idx, doc, chunk, tag, score);

    const body = el("p", "source-text");
    body.textContent = s.text;

    card.append(head, body);
    wrap.appendChild(card);
  }
}

function renderMeta(data) {
  const badge = $("#groundBadge");
  if (data.abstained) {
    badge.className = "badge abstain";
    badge.textContent = "no answer in corpus";
  } else if (data.grounded) {
    badge.className = "badge ok";
    badge.textContent = "grounded";
  } else {
    badge.className = "badge warn";
    badge.textContent = "ungrounded — verify";
  }
  $("#metaLatency").textContent = `${data.trace.total_ms.toFixed(0)}ms`;
  const cost = data.trace.cost_usd;
  $("#metaCost").textContent = cost > 0 ? `$${cost.toFixed(6)}` : "local · $0";
  $("#metaModel").textContent = `${data.provider}/${data.model}`;
}

// ---- ask flow ------------------------------------------------------------
async function ask(question) {
  if (!question || question.trim().length < 3) return;
  errorBox.hidden = true;
  resultSection.hidden = false;
  askBtn.disabled = true;
  askBtn.textContent = "…";
  $("#trace").classList.add("loading");
  $("#traceStages").innerHTML = "";
  $("#traceTotal").textContent = "";
  $("#paper").style.opacity = ".35";

  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);

    $("#trace").classList.remove("loading");
    renderTrace(data.trace.stages, data.trace.total_ms);
    renderMeta(data);
    renderAnswer($("#answer"), data.answer, data.abstained);
    renderSources(data.sources);
    $("#paper").style.opacity = "1";
  } catch (err) {
    resultSection.hidden = true;
    errorBox.hidden = false;
    $("#errorBody").textContent = err.message;
    $("#trace").classList.remove("loading");
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

// ---- events --------------------------------------------------------------
askBtn.addEventListener("click", () => ask(input.value));
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask(input.value);
});
$("#suggestions").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  input.value = chip.textContent;
  ask(chip.textContent);
});

loadStats();
