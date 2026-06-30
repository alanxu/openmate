// OpenMate UI — sidebar (new task / history) + main chat pane with a
// collapsed-by-default trajectory view, fed by the SSE event stream at
// /api/chat/stream. One event type per openmate/kernel/events.py class.

const messagesEl = document.getElementById("messages");
const threadListEl = document.getElementById("thread-list");
const inputEl = document.getElementById("input");
const composerEl = document.getElementById("composer");
const sendBtn = document.getElementById("send-btn");
const newTaskBtn = document.getElementById("new-task-btn");
const addBtn = document.getElementById("add-btn");
const addMenu = document.getElementById("add-menu");
const fileInput = document.getElementById("file-input");
const knowledgeListEl = document.getElementById("knowledge-list");
const folderListEl = document.getElementById("folder-list");
const modalOverlay = document.getElementById("modal-overlay");
const modalTitle = document.getElementById("modal-title");
const modalTextarea = document.getElementById("modal-textarea");
const modalInput = document.getElementById("modal-input");
const modalError = document.getElementById("modal-error");
const modalCancelBtn = document.getElementById("modal-cancel");
const modalOkBtn = document.getElementById("modal-ok");

let threadId = null;
let es = null; // current EventSource, if a run is in flight
let liveAssistantBubble = null; // streamed-text target while ModelStreamed deltas arrive
let liveCards = new Map(); // call_id -> trace card element, for the in-flight run

function uuid() {
  return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2) + Date.now();
}

function clearMessages() {
  messagesEl.innerHTML = "";
}

function showEmptyState() {
  clearMessages();
  const div = document.createElement("div");
  div.className = "empty-state";
  div.innerHTML = `<div class="empty-title">OpenMate</div><div class="empty-sub">Start a new task to begin.</div>`;
  messagesEl.appendChild(div);
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function addStatusLine(text) {
  const div = document.createElement("div");
  div.className = "status-line";
  div.textContent = text;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

// Click the head of any trace/thinking card to toggle it open/closed.
function bindTraceToggle(card) {
  const head = card.querySelector(".trace-head");
  head.addEventListener("click", () => card.classList.toggle("open"));
}

// --- tool-call card — collapsed by default, click the head to expand ------------
function addTraceCard(name, args) {
  const card = document.createElement("div");
  card.className = "trace-card";
  card.innerHTML = `
    <div class="trace-head">
      <span class="trace-chevron">▸</span>
      <span class="trace-icon">⚒</span>
      <span class="trace-name">${escapeHtml(name)}</span>
      <span class="trace-meta" data-role="meta">running…</span>
    </div>
    <div class="trace-body">
      <div class="section-label">Arguments</div>
      <div data-role="args">${escapeHtml(JSON.stringify(args, null, 2))}</div>
      <div class="section-label" data-role="result-label" style="display:none">Result</div>
      <div data-role="result"></div>
    </div>`;
  bindTraceToggle(card);
  messagesEl.appendChild(card);
  scrollToBottom();
  return card;
}

// --- "thinking" card — collapsed by default, same shell as a trace card --------
function addThinkingCard(text) {
  const card = document.createElement("div");
  card.className = "trace-card thinking-card";
  card.innerHTML = `
    <div class="trace-head">
      <span class="trace-chevron">▸</span>
      <span class="trace-icon">💭</span>
      <span class="trace-name">Thinking</span>
    </div>
    <div class="trace-body" data-role="thinking-body"></div>`;
  card.querySelector('[data-role="thinking-body"]').textContent = text;
  bindTraceToggle(card);
  messagesEl.appendChild(card);
  scrollToBottom();
  return card;
}

// Walks an assistant message's `content` *in order* — text, thinking, and tool
// calls can all appear in the same model response, and rendering them in their
// original sequence (rather than the old "just msg.text + msg.tool_calls"
// shortcut) is what actually surfaces the thinking/tool trajectory in the UI.
function renderAssistantContent(msg) {
  for (const part of msg.content) {
    if (part.kind === "text") {
      if (!part.text) continue;
      if (liveAssistantBubble) {
        liveAssistantBubble.textContent = part.text;
        liveAssistantBubble = null;
      } else {
        addBubble("assistant", part.text);
      }
    } else if (part.kind === "thinking") {
      if (part.text) addThinkingCard(part.text);
    } else if (part.kind === "tool_call") {
      const card = addTraceCard(part.name, part.args);
      card.dataset.callId = part.id;
      liveCards.set(part.id, card);
    }
  }
}

function fillTraceResult(card, resultText, isError, ms) {
  const meta = card.querySelector('[data-role="meta"]');
  meta.textContent = `${isError ? "error" : "done"} · ${Math.round(ms)}ms`;
  const label = card.querySelector('[data-role="result-label"]');
  label.style.display = "block";
  const result = card.querySelector('[data-role="result"]');
  result.textContent = resultText;
  if (isError) result.classList.add("trace-error");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function toolResultText(resultPart) {
  if (!resultPart || !resultPart.content) return "";
  return resultPart.content
    .map((p) => (p.kind === "text" ? p.text : p.kind === "thinking" ? p.text : JSON.stringify(p)))
    .join("\n");
}

// --- thread list -----------------------------------------------------------------
async function loadThreads() {
  const res = await fetch("/api/threads");
  const threads = await res.json();
  threadListEl.innerHTML = "";
  for (const t of threads) {
    const div = document.createElement("div");
    div.className = "thread-item" + (t.thread_id === threadId ? " active" : "");
    div.textContent = t.title || "Untitled";
    div.title = t.title || "";
    div.addEventListener("click", () => openThread(t.thread_id));
    threadListEl.appendChild(div);
  }
}

// --- URL deep-linking: ?thread_id=<id> so a conversation can be bookmarked/shared ---
function setUrlThreadId(id) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("thread_id", id);
  else url.searchParams.delete("thread_id");
  history.pushState({ threadId: id }, "", url);
}

function getUrlThreadId() {
  return new URLSearchParams(window.location.search).get("thread_id");
}

async function openThread(id) {
  closeStream();
  threadId = id;
  setUrlThreadId(id);
  clearMessages();
  const res = await fetch(`/api/threads/${id}`);
  if (!res.ok) {
    showEmptyState();
    return;
  }
  const data = await res.json();
  for (const m of data.messages) {
    if (m.role === "user") {
      addBubble("user", m.text);
    } else if (m.role === "assistant") {
      renderAssistantContent(m);
    } else if (m.role === "tool") {
      for (const part of m.content) {
        if (part.kind !== "tool_result") continue;
        const card = [...messagesEl.querySelectorAll(".trace-card")].find(
          (c) => c.dataset.callId === part.call_id
        );
        if (card) fillTraceResult(card, toolResultText(part), part.is_error, 0);
      }
    }
  }
  await loadThreads();
  await loadContextChips();
  scrollToBottom();
}

// --- new task ---------------------------------------------------------------------
newTaskBtn.addEventListener("click", () => {
  closeStream();
  threadId = uuid();
  setUrlThreadId(threadId);
  showEmptyState();
  [...threadListEl.children].forEach((el) => el.classList.remove("active"));
  renderContextChips([], []);
});

// back/forward should re-sync the open thread with the URL
window.addEventListener("popstate", () => {
  const id = getUrlThreadId();
  if (id) {
    openThread(id);
  } else {
    closeStream();
    threadId = null;
    showEmptyState();
    [...threadListEl.children].forEach((el) => el.classList.remove("active"));
    renderContextChips([], []);
  }
});

function closeStream() {
  if (es) {
    es.close();
    es = null;
  }
  liveAssistantBubble = null;
  liveCards.clear();
}

// --- sending a message / consuming the SSE event stream --------------------------
composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

function send() {
  const text = inputEl.value.trim();
  if (!text || es) return;
  if (!threadId) {
    threadId = uuid();
    setUrlThreadId(threadId);
  }
  if (messagesEl.querySelector(".empty-state")) clearMessages();

  addBubble("user", text);
  inputEl.value = "";
  sendBtn.disabled = true;

  const url = `/api/chat/stream?thread_id=${encodeURIComponent(threadId)}&message=${encodeURIComponent(text)}`;
  es = new EventSource(url);

  es.addEventListener("RunStarted", () => {});

  es.addEventListener("ModelStreamed", (ev) => {
    const data = JSON.parse(ev.data);
    if (data.delta.kind !== "text" || !data.delta.text) return;
    if (!liveAssistantBubble) liveAssistantBubble = addBubble("assistant", "");
    liveAssistantBubble.textContent += data.delta.text;
    scrollToBottom();
  });

  es.addEventListener("MessageAdded", (ev) => {
    const data = JSON.parse(ev.data);
    const msg = data.message;
    if (msg.role === "assistant") {
      renderAssistantContent(msg);
    }
  });

  es.addEventListener("ToolReturned", (ev) => {
    const data = JSON.parse(ev.data);
    const part = data.result;
    const card = liveCards.get(part.call_id);
    if (card) fillTraceResult(card, toolResultText(part), part.is_error, data.ms);
  });

  es.addEventListener("RunFinished", (ev) => {
    const data = JSON.parse(ev.data);
    if (data.reason && data.reason !== "natural") {
      addStatusLine(`stopped: ${data.reason}`);
    }
    closeStream();
    sendBtn.disabled = false;
    loadThreads();
  });

  es.addEventListener("Error", (ev) => {
    const data = JSON.parse(ev.data);
    addStatusLine(`error: ${data.error}`);
    closeStream();
    sendBtn.disabled = false;
  });

  es.onerror = () => {
    sendBtn.disabled = false;
    closeStream();
  };
}

// --- generic modal (used instead of window.prompt, which some embedded browser
// views — e.g. inside a desktop app's webview — silently block: prompt() returns
// immediately with no dialog shown at all, which looked like a dead "+" button) --
let modalResolve = null;

function openModal({ title, multiline = false, okLabel = "Add" }) {
  modalTitle.textContent = title;
  modalOkBtn.textContent = okLabel;
  modalError.classList.add("hidden");
  modalError.textContent = "";
  if (multiline) {
    modalTextarea.classList.remove("hidden");
    modalInput.classList.add("hidden");
    modalTextarea.value = "";
  } else {
    modalInput.classList.remove("hidden");
    modalTextarea.classList.add("hidden");
    modalInput.value = "";
  }
  modalOverlay.classList.remove("hidden");
  (multiline ? modalTextarea : modalInput).focus();
  return new Promise((resolve) => {
    modalResolve = resolve;
  });
}

function closeModal(value) {
  modalOverlay.classList.add("hidden");
  if (modalResolve) {
    modalResolve(value);
    modalResolve = null;
  }
}

function showModalError(text) {
  modalError.textContent = text;
  modalError.classList.remove("hidden");
}

modalCancelBtn.addEventListener("click", () => closeModal(null));
modalOverlay.addEventListener("click", (e) => {
  if (e.target === modalOverlay) closeModal(null);
});
modalOkBtn.addEventListener("click", () => {
  const visible = modalTextarea.classList.contains("hidden") ? modalInput : modalTextarea;
  const value = visible.value.trim();
  if (!value) {
    showModalError("This field is required.");
    return;
  }
  closeModal(value);
});

// --- '+' menu: add knowledge (paste/upload) or a folder to edit, per thread ------
function ensureThreadId() {
  if (!threadId) {
    threadId = uuid();
    setUrlThreadId(threadId);
  }
  return threadId;
}

addBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  addMenu.classList.toggle("hidden");
});
document.addEventListener("click", (e) => {
  if (!addMenu.contains(e.target) && e.target !== addBtn) addMenu.classList.add("hidden");
});

addMenu.addEventListener("click", async (e) => {
  const action = e.target.dataset && e.target.dataset.action;
  if (!action) return;
  addMenu.classList.add("hidden");

  if (action === "paste-text") {
    const text = await openModal({ title: "Add knowledge — paste text", multiline: true });
    if (text) await addKnowledgeText(text);
  } else if (action === "upload-file") {
    fileInput.value = "";
    fileInput.click();
  } else if (action === "add-folder") {
    const path = await openModal({ title: "Add folder for editing — absolute path", okLabel: "Add folder" });
    if (path) await addFolder(path);
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (file) await addKnowledgeFile(file);
});

async function addKnowledgeText(text) {
  const id = ensureThreadId();
  addStatusLine("adding knowledge…");
  const res = await fetch(`/api/threads/${id}/knowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  if (!res.ok) addStatusLine(`error: ${data.error || "failed to add knowledge"}`);
  await loadContextChips();
}

async function addKnowledgeFile(file) {
  const id = ensureThreadId();
  addStatusLine(`adding knowledge from ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`/api/threads/${id}/knowledge`, { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) addStatusLine(`error: ${data.error || "failed to add knowledge"}`);
  await loadContextChips();
}

async function addFolder(path) {
  const id = ensureThreadId();
  const res = await fetch(`/api/threads/${id}/folders`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const data = await res.json();
  if (!res.ok) addStatusLine(`error: ${data.error || "failed to add folder"}`);
  await loadContextChips();
}

async function removeKnowledge(source) {
  if (!threadId) return;
  await fetch(`/api/threads/${threadId}/knowledge?source=${encodeURIComponent(source)}`, { method: "DELETE" });
  await loadContextChips();
}

async function removeFolder(path) {
  if (!threadId) return;
  await fetch(`/api/threads/${threadId}/folders?path=${encodeURIComponent(path)}`, { method: "DELETE" });
  await loadContextChips();
}

async function loadContextChips() {
  if (!threadId) {
    renderContextChips([], []);
    return;
  }
  const [kRes, fRes] = await Promise.all([
    fetch(`/api/threads/${threadId}/knowledge`),
    fetch(`/api/threads/${threadId}/folders`),
  ]);
  const knowledge = kRes.ok ? await kRes.json() : [];
  const folders = fRes.ok ? await fRes.json() : [];
  renderContextChips(knowledge, folders);
}

function renderContextChips(knowledge, folders) {
  renderContextList(knowledgeListEl, "📄", knowledge.map((k) => k.source), removeKnowledge);
  renderContextList(folderListEl, "📁", folders, removeFolder);
}

function renderContextList(listEl, icon, items, onRemove) {
  listEl.innerHTML = "";
  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "context-empty";
    empty.textContent = "None added yet";
    listEl.appendChild(empty);
    return;
  }
  for (const label of items) {
    listEl.appendChild(makeContextItem(icon, label, () => onRemove(label)));
  }
}

function makeContextItem(icon, label, onRemove) {
  const item = document.createElement("div");
  item.className = "context-item";
  item.innerHTML = `<span class="context-icon">${icon}</span><span class="context-name">${escapeHtml(label)}</span>`;
  const removeBtn = document.createElement("button");
  removeBtn.className = "context-remove";
  removeBtn.textContent = "×";
  removeBtn.title = "Remove";
  removeBtn.addEventListener("click", onRemove);
  item.appendChild(removeBtn);
  return item;
}

// --- boot --------------------------------------------------------------------------
const initialThreadId = getUrlThreadId();
if (initialThreadId) {
  openThread(initialThreadId);
} else {
  loadThreads();
}
