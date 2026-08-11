// ArchMind Chrome Extension — Popup Logic
// Handles UI interactions, API calls, and result rendering.

const API_BASE = "http://localhost:8765";

// ── DOM refs ─────────────────────────────
const questionEl = document.getElementById("question");
const providerEl = document.getElementById("provider");
const modelEl = document.getElementById("model");
const askBtn = document.getElementById("ask-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");
const errorEl = document.getElementById("error");
const answerSection = document.getElementById("answer-section");
const answerEl = document.getElementById("answer");
const sourcesSection = document.getElementById("sources-section");
const sourcesEl = document.getElementById("sources");
const statusEl = document.getElementById("status");

// ── Init ──────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Check API health & configure defaults
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (resp.ok) {
      const health = await resp.json();
      statusEl.textContent = `🟢 已连接 · ${health.files_count} 份文档`;
      statusEl.className = "status ok";
      // Set default provider based on server
      if (health.provider && providerEl) {
        for (const opt of providerEl.options) {
          if (opt.value === health.provider) {
            opt.selected = true;
            break;
          }
        }
      }
    } else {
      throw new Error("API unhealthy");
    }
  } catch {
    statusEl.textContent = "🔴 未连接 API · 请确认已启动 api.py";
    statusEl.className = "status error";
  }

  // Load stored selection text from context menu
  const stored = await chrome.storage.local.get(["selectedText", "timestamp"]);
  if (stored.selectedText) {
    // Only use if stored within the last 5 minutes
    const age = Date.now() - (stored.timestamp || 0);
    if (age < 5 * 60 * 1000) {
      questionEl.value = stored.selectedText;
      // Clear after reading
      await chrome.storage.local.remove(["selectedText", "timestamp"]);
    }
  }
});

// ── Ask button ────────────────────────────
askBtn.addEventListener("click", async () => {
  const question = questionEl.value.trim();
  if (!question) {
    showError("请输入一个问题。");
    return;
  }

  // UI: loading state
  setLoading(true);
  hideError();
  hideResults();

  try {
    const resp = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        provider: providerEl.value,
        model: modelEl.value.trim() || null,
        top_k: 4,
      }),
    });

    if (!resp.ok) {
      const detail = (await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`;
      throw new Error(detail);
    }

    const data = await resp.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    // Render answer
    answerEl.innerHTML = formatAnswer(data.answer);
    answerSection.classList.remove("hidden");

    // Render sources
    if (data.sources && data.sources.length > 0) {
      sourcesEl.innerHTML = data.sources
        .map(
          (s, i) => `
          <details class="source-item">
            <summary>来源 ${i + 1}：${escapeHtml(s.name)}</summary>
            <div class="source-content">${escapeHtml(s.content)}</div>
          </details>`
        )
        .join("");
      sourcesSection.classList.remove("hidden");
    }
  } catch (err) {
    if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
      showError("无法连接 ArchMind API。请确认已启动：python api.py");
    } else {
      showError(err.message);
    }
  } finally {
    setLoading(false);
  }
});

// ── Keyboard shortcut ─────────────────────
questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    askBtn.click();
  }
});

// ── Helpers ───────────────────────────────
function setLoading(loading) {
  askBtn.disabled = loading;
  btnText.classList.toggle("hidden", loading);
  btnSpinner.classList.toggle("hidden", !loading);
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function hideResults() {
  answerSection.classList.add("hidden");
  sourcesSection.classList.add("hidden");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatAnswer(text) {
  // Simple formatting: preserve line breaks, escape HTML
  let html = escapeHtml(text);
  html = html.replace(/\n\n/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");
  html = `<p>${html}</p>`;
  // Highlight source citations like [来源 1], [来源 2]
  html = html.replace(
    /(\[来源\s*\d+\])/g,
    '<span class="citation">$1</span>'
  );
  return html;
}
