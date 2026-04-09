/* ============================================================
   script.js — Fake News Detector Frontend Logic
   ============================================================ */

"use strict";

// ── Sample test cases (shown at the bottom of the page) ──────
const TEST_CASES = [
  {
    tag: "REAL",
    text: "NASA scientists have confirmed the discovery of water ice deposits near the Martian south pole, " +
          "according to a study published in the journal Geophysical Research Letters. " +
          "The findings could have significant implications for future crewed missions to the planet.",
  },
  {
    tag: "REAL",
    text: "The Federal Reserve raised its benchmark interest rate by 0.25 percentage points on Wednesday, " +
          "the tenth increase in just over a year, as the central bank continues its effort to bring " +
          "inflation back down to its 2 percent target.",
  },
  {
    tag: "FAKE",
    text: "BREAKING: Doctors REFUSE to talk about this miracle cure that the pharmaceutical industry " +
          "has been suppressing for decades! One simple vegetable DESTROYS cancer cells instantly. " +
          "Big Pharma doesn't want you to see this — SHARE before they DELETE it!",
  },
  {
    tag: "FAKE",
    text: "SHOCKING EXCLUSIVE: Government whistleblower reveals 5G towers are secretly broadcasting " +
          "mind-control frequencies. Elite globalists have been using chemtrails since 1997 to make " +
          "the population more susceptible. The mainstream media will NEVER report this truth.",
  },
];

// ── DOM references ────────────────────────────────────────────
const textarea      = document.getElementById("news-input");
const charCount     = document.getElementById("char-count");
const btnDetect     = document.getElementById("btn-detect");
const btnClear      = document.getElementById("btn-clear");
const btnDemo       = document.getElementById("btn-demo");
const resultPanel   = document.getElementById("result-panel");
const errorBanner   = document.getElementById("error-banner");
const errorMsg      = document.getElementById("error-msg");
const statusBadge   = document.getElementById("status-badge");
const verdictText   = document.getElementById("verdict-text");
const verdictIcon   = document.getElementById("verdict-icon");
const ringFill      = document.getElementById("ring-fill");
const ringLabel     = document.getElementById("ring-label");
const consensusBanner = document.querySelector(".consensus-banner");
const modelBreakdown  = document.getElementById("model-breakdown");
const statChars     = document.getElementById("stat-chars");
const statWords     = document.getElementById("stat-words");
const statModels    = document.getElementById("stat-models");
const testGrid      = document.getElementById("test-grid");
const toggleBtns    = document.querySelectorAll(".toggle-btn");

// ── State ──────────────────────────────────────────────────────
let selectedModel = "both";

// ── Initialise page ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  renderTestCases();
  updateCharCount();
});

// ── Health check ──────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (data.models_loaded) {
      statusBadge.textContent = "✓ Models ready";
      statusBadge.className   = "badge badge-ok";
    } else {
      // Models still loading — retry after 3 s
      statusBadge.textContent = "⚠ Models loading…";
      statusBadge.className   = "badge badge-loading";
      btnDetect.disabled = true;
      setTimeout(checkHealth, 3000);
    }
  } catch (err) {
    console.error("[FakeNewsDetector] Health check failed:", err);
    statusBadge.textContent = "✗ Server offline — run: python app.py";
    statusBadge.className   = "badge badge-error";
    // Retry every 5 s in case the user starts Flask while page is open
    setTimeout(checkHealth, 5000);
  }
}

// ── Model toggle ──────────────────────────────────────────────
toggleBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    toggleBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedModel = btn.dataset.model;
  });
});

// ── Character count & button state ───────────────────────────
textarea.addEventListener("input", updateCharCount);

function updateCharCount() {
  const len = textarea.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000`;
  btnDetect.disabled = len < 20;
}

// ── Clear button ──────────────────────────────────────────────
btnClear.addEventListener("click", () => {
  textarea.value = "";
  updateCharCount();
  hideResult();
  hideError();
  textarea.focus();
});

// ── Demo button ───────────────────────────────────────────────
btnDemo.addEventListener("click", () => {
  const demo = TEST_CASES[Math.floor(Math.random() * TEST_CASES.length)];
  textarea.value = demo.text;
  updateCharCount();
  hideError();
  hideResult();
  textarea.focus();
});

// ── Detect button ─────────────────────────────────────────────
btnDetect.addEventListener("click", runDetection);
textarea.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runDetection();
});

async function runDetection() {
  const text = textarea.value.trim();
  if (!text || text.length < 20) return;

  setLoading(true);
  hideResult();
  hideError();

  try {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, model: selectedModel }),
    });

    // Safely parse JSON — the server might return an HTML error page on crash
    let data;
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      data = await res.json();
    } else {
      const raw = await res.text();
      console.error("[FakeNewsDetector] Non-JSON response:", raw);
      showError(`Server returned an unexpected response (HTTP ${res.status}). ` +
                `Check the terminal for errors.`);
      return;
    }

    if (!res.ok || !data.success) {
      showError(data.error || "An unexpected error occurred. Please try again.");
      return;
    }

    renderResult(data, text);

  } catch (err) {
    // Log the real error to the browser console for debugging
    console.error("[FakeNewsDetector] fetch() failed:", err);
    showError(
      `Could not reach the server (${err.message}). ` +
      `Make sure Flask is running: open a terminal and run  python app.py`
    );
  } finally {
    setLoading(false);
  }
}

// ── Render result ─────────────────────────────────────────────
function labelClass(label) {
  if (label === "REAL")      return "is-real";
  if (label === "FAKE")      return "is-fake";
  return "is-uncertain";
}

function labelIcon(label) {
  if (label === "REAL")      return "✓";
  if (label === "FAKE")      return "✗";
  return "?";
}

function renderResult(data, originalText) {
  const consensus   = data.consensus;                     // "REAL" | "FAKE" | "UNCERTAIN"
  const cls         = labelClass(consensus);
  const avgConf     = avg(data.results.map(r => r.confidence));
  const circumference = 100;

  // Consensus banner
  verdictText.textContent = consensus;
  verdictText.className   = `verdict-text ${cls}`;
  verdictIcon.textContent = labelIcon(consensus);
  consensusBanner.className = `consensus-banner ${cls}`;

  // Show explanatory note for UNCERTAIN
  const existingNote = document.getElementById("uncertain-note");
  if (existingNote) existingNote.remove();
  if (consensus === "UNCERTAIN") {
    const note = document.createElement("p");
    note.id = "uncertain-note";
    note.style.cssText = "font-size:.82rem;color:var(--clr-uncertain);margin-top:8px;";
    note.textContent = "⚠ The models lack enough training data for this topic "
                     + "(e.g. international news, science, sports). "
                     + "Result is inconclusive — verify with a trusted source.";
    consensusBanner.after(note);
  }

  // Confidence ring
  const pct = (avgConf / 100) * circumference;
  ringFill.setAttribute("stroke-dasharray", `${pct.toFixed(1)} ${circumference}`);
  ringFill.setAttribute("class", `ring-fill ${cls}`);
  ringLabel.textContent = `${Math.round(avgConf)}%`;

  // Per-model breakdown
  modelBreakdown.innerHTML = data.results.map(r => `
    <div class="model-row">
      <span class="model-name">${escHtml(r.model)}</span>
      <span class="model-label ${labelClass(r.label)}">${r.label}</span>
      <div class="conf-bar-wrap">
        <div class="conf-bar ${labelClass(r.label)}"
             style="width:${r.confidence}%"></div>
      </div>
      <span class="model-conf">${r.confidence}%</span>
    </div>
  `).join("");

  // Stats
  statChars.textContent  = data.text_length.toLocaleString();
  statWords.textContent  = wordCount(originalText).toLocaleString();
  statModels.textContent = data.results.length;

  resultPanel.classList.remove("hidden");
}

// ── UI helpers ────────────────────────────────────────────────
function setLoading(on) {
  btnDetect.querySelector(".btn-text").style.display    = on ? "none"         : "";
  btnDetect.querySelector(".btn-spinner").style.display = on ? "inline-flex"  : "none";
  btnDetect.disabled = on;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.remove("hidden");
}
function hideError()  { errorBanner.classList.add("hidden"); }
function hideResult() { resultPanel.classList.add("hidden"); }

function avg(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}
function wordCount(str) {
  return str.trim().split(/\s+/).filter(Boolean).length;
}
function escHtml(str) {
  const d = document.createElement("div");
  d.appendChild(document.createTextNode(str));
  return d.innerHTML;
}

// ── Render test cases ─────────────────────────────────────────
function renderTestCases() {
  testGrid.innerHTML = TEST_CASES.map((tc, i) => `
    <div class="test-card" data-idx="${i}" role="button" tabindex="0"
         aria-label="Try ${tc.tag} example ${i+1}">
      <span class="test-card-tag ${tc.tag.toLowerCase()}">${tc.tag}</span>
      <p>${escHtml(tc.text.slice(0, 140))}…</p>
      <p class="try-hint">↑ Click to try this example</p>
    </div>
  `).join("");

  testGrid.querySelectorAll(".test-card").forEach(card => {
    const activate = () => {
      const idx = parseInt(card.dataset.idx, 10);
      textarea.value = TEST_CASES[idx].text;
      updateCharCount();
      hideError();
      hideResult();
      textarea.scrollIntoView({ behavior: "smooth", block: "center" });
      textarea.focus();
    };
    card.addEventListener("click",   activate);
    card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") activate(); });
  });
}
