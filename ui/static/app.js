// OpenMate UI — left-panel workspace: Tasks / Libraries / Projects (docs/19).
// A Task is a chat with an SSE trajectory. Libraries are reusable knowledge;
// Projects bundle work directories + goals. A task attaches to ≤1 project and
// 0..N libraries. The center pane shows the selected entity (task/library/project).

const navListEl = document.getElementById("nav-list");
const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const composerEl = document.getElementById("composer");
const sendBtn = document.getElementById("send-btn");
const newBtn = document.getElementById("new-btn");
const taskAttachEl = document.getElementById("task-attach");
const libraryViewEl = document.getElementById("library-view");
const projectViewEl = document.getElementById("project-view");
const sectionTabs = [...document.querySelectorAll(".section-tab")];
const screens = {
  task: document.getElementById("screen-task"),
  library: document.getElementById("screen-library"),
  project: document.getElementById("screen-project"),
};
const modalOverlay = document.getElementById("modal-overlay");
const modalTitle = document.getElementById("modal-title");
const modalTextarea = document.getElementById("modal-textarea");
const modalInput = document.getElementById("modal-input");
const modalList = document.getElementById("modal-list");
const modalForm = document.getElementById("modal-form");
const modalError = document.getElementById("modal-error");
const modalActions = document.querySelector(".modal-actions");
const modalCancelBtn = document.getElementById("modal-cancel");
const modalOkBtn = document.getElementById("modal-ok");

let section = "tasks"; // left panel: tasks | libraries | projects
let threadId = null; // current task
let activeLibraryId = null; // library open in the library screen
let activeProjectId = null; // project open in the project screen
let es = null; // current EventSource, if a run is in flight
let liveAssistantBubble = null;
let liveCards = new Map();

function uuid() {
  return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2) + Date.now();
}

// --- chat rendering (unchanged trajectory view) ---------------------------------
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
function bindTraceToggle(card) {
  card.querySelector(".trace-head").addEventListener("click", () => card.classList.toggle("open"));
}
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
  div.textContent = s == null ? "" : s;
  return div.innerHTML;
}
function toolResultText(resultPart) {
  if (!resultPart || !resultPart.content) return "";
  return resultPart.content
    .map((p) => (p.kind === "text" ? p.text : p.kind === "thinking" ? p.text : JSON.stringify(p)))
    .join("\n");
}

// --- screens + left-panel sections ----------------------------------------------
function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) el.classList.toggle("hidden", k !== name);
}

sectionTabs.forEach((t) => t.addEventListener("click", () => setSection(t.dataset.section)));
function setSection(name) {
  section = name;
  sectionTabs.forEach((t) => t.classList.toggle("active", t.dataset.section === name));
  newBtn.textContent =
    name === "tasks" ? "+ New task" : name === "libraries" ? "+ New library" : "+ New project";
  loadNav();
}

async function loadNav() {
  if (section === "tasks") return loadTasksNav();
  if (section === "libraries") return loadLibrariesNav();
  return loadProjectsNav();
}

async function loadTasksNav() {
  const res = await fetch("/api/threads");
  const threads = res.ok ? await res.json() : [];
  navListEl.innerHTML = "";
  if (!threads.length) return navListEl.appendChild(navEmpty("No tasks yet"));
  for (const t of threads) {
    const active = t.thread_id === threadId && !screens.task.classList.contains("hidden");
    navListEl.appendChild(navItem(t.title || "Untitled", active, () => openThread(t.thread_id)));
  }
}
async function loadLibrariesNav() {
  const res = await fetch("/api/libraries");
  const libs = (res.ok ? await res.json() : []).filter((l) => l.kind === "shared");
  navListEl.innerHTML = "";
  if (!libs.length) return navListEl.appendChild(navEmpty("No libraries yet"));
  for (const l of libs) {
    navListEl.appendChild(
      navItem(l.name || l.library_id, l.library_id === activeLibraryId, () => openLibrary(l.library_id), `${l.n_chunks || 0} chunks`)
    );
  }
}
async function loadProjectsNav() {
  const res = await fetch("/api/projects");
  const projs = res.ok ? await res.json() : [];
  navListEl.innerHTML = "";
  if (!projs.length) return navListEl.appendChild(navEmpty("No projects yet"));
  for (const p of projs) {
    const sub = `${p.n_dirs} dir${p.n_dirs === 1 ? "" : "s"}${p.has_goals ? " · goals" : ""}`;
    navListEl.appendChild(navItem(p.name, p.project_id === activeProjectId, () => openProject(p.project_id), sub));
  }
}
function navItem(label, active, onClick, sub) {
  const div = document.createElement("div");
  div.className = "nav-item" + (active ? " active" : "");
  div.innerHTML = `<div class="nav-name">${escapeHtml(label)}</div>` + (sub ? `<div class="nav-sub">${escapeHtml(sub)}</div>` : "");
  div.title = label;
  div.addEventListener("click", onClick);
  return div;
}
function navEmpty(text) {
  const d = document.createElement("div");
  d.className = "nav-empty";
  d.textContent = text;
  return d;
}

newBtn.addEventListener("click", () => {
  if (section === "tasks") newTask();
  else if (section === "libraries") newLibrary();
  else newProject();
});

// --- tasks ----------------------------------------------------------------------
function newTask() {
  closeStream();
  threadId = uuid();
  setUrlThreadId(threadId);
  showScreen("task");
  showEmptyState();
  renderTaskAttach();
  if (section === "tasks") loadNav();
}

async function openThread(id) {
  closeStream();
  threadId = id;
  setUrlThreadId(id);
  showScreen("task");
  clearMessages();
  const res = await fetch(`/api/threads/${id}`);
  if (res.ok) {
    const data = await res.json();
    for (const m of data.messages) {
      if (m.role === "user") addBubble("user", m.text);
      else if (m.role === "assistant") renderAssistantContent(m);
      else if (m.role === "tool") {
        for (const part of m.content) {
          if (part.kind !== "tool_result") continue;
          const card = [...messagesEl.querySelectorAll(".trace-card")].find((c) => c.dataset.callId === part.call_id);
          if (card) fillTraceResult(card, toolResultText(part), part.is_error, 0);
        }
      }
    }
  } else {
    showEmptyState();
  }
  await renderTaskAttach();
  if (section === "tasks") loadNav();
  scrollToBottom();
}

function setUrlThreadId(id) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("thread_id", id);
  else url.searchParams.delete("thread_id");
  history.pushState({ threadId: id }, "", url);
}
function getUrlThreadId() {
  return new URLSearchParams(window.location.search).get("thread_id");
}
window.addEventListener("popstate", () => {
  const id = getUrlThreadId();
  if (id) openThread(id);
  else newTask();
});

function closeStream() {
  if (es) {
    es.close();
    es = null;
  }
  liveAssistantBubble = null;
  liveCards.clear();
}

function ensureThreadId() {
  if (!threadId) {
    threadId = uuid();
    setUrlThreadId(threadId);
  }
  return threadId;
}

// --- sending a message / consuming the SSE event stream -------------------------
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
  ensureThreadId();
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
    const msg = JSON.parse(ev.data).message;
    if (msg.role === "assistant") renderAssistantContent(msg);
  });
  es.addEventListener("ToolReturned", (ev) => {
    const data = JSON.parse(ev.data);
    const part = data.result;
    const card = liveCards.get(part.call_id);
    if (card) fillTraceResult(card, toolResultText(part), part.is_error, data.ms);
  });
  es.addEventListener("RunFinished", (ev) => {
    const data = JSON.parse(ev.data);
    if (data.reason && data.reason !== "natural") addStatusLine(`stopped: ${data.reason}`);
    closeStream();
    sendBtn.disabled = false;
    if (section === "tasks") loadNav();
  });
  es.addEventListener("Error", (ev) => {
    addStatusLine(`error: ${JSON.parse(ev.data).error}`);
    closeStream();
    sendBtn.disabled = false;
  });
  es.onerror = () => {
    sendBtn.disabled = false;
    closeStream();
  };
}

// --- task attachment bar: project + libraries + private knowledge ---------------
async function renderTaskAttach() {
  if (!threadId) {
    taskAttachEl.classList.add("hidden");
    taskAttachEl.innerHTML = "";
    return;
  }
  const [projRes, libRes] = await Promise.all([
    fetch(`/api/threads/${threadId}/project`),
    fetch(`/api/threads/${threadId}/libraries`),
  ]);
  const project = projRes.ok ? await projRes.json() : null;
  const libs = libRes.ok ? await libRes.json() : [];
  const shared = libs.filter((l) => l.kind === "shared");

  taskAttachEl.classList.remove("hidden");
  taskAttachEl.innerHTML = "";
  taskAttachEl.appendChild(attachLabel("Project"));
  if (project) taskAttachEl.appendChild(attachChip(project.name, () => detachProject(), () => openProject(project.project_id)));
  taskAttachEl.appendChild(attachAction(project ? "change" : "+ attach", changeProjectFlow));
  taskAttachEl.appendChild(attachDivider());
  taskAttachEl.appendChild(attachLabel("Libraries"));
  for (const l of shared) taskAttachEl.appendChild(attachChip(l.name || l.library_id, () => detachLibrary(l.library_id), () => openLibrary(l.library_id)));
  taskAttachEl.appendChild(attachAction("+ attach", attachLibraryFlow));
}
function attachLabel(text) {
  const s = document.createElement("span");
  s.className = "attach-label";
  s.textContent = text;
  return s;
}
function attachDivider() {
  const s = document.createElement("span");
  s.className = "attach-div";
  return s;
}
function attachChip(label, onRemove, onOpen) {
  const s = document.createElement("span");
  s.className = "attach-chip";
  const name = document.createElement("span");
  name.className = "attach-chip-name";
  name.textContent = label;
  if (onOpen) name.addEventListener("click", onOpen);
  s.appendChild(name);
  if (onRemove) {
    const x = document.createElement("button");
    x.className = "attach-chip-x";
    x.textContent = "×";
    x.title = "Detach";
    x.addEventListener("click", (e) => {
      e.stopPropagation();
      onRemove();
    });
    s.appendChild(x);
  }
  return s;
}
function attachAction(label, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "attach-action";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}

async function changeProjectFlow() {
  ensureThreadId();
  const res = await fetch("/api/projects");
  const projs = res.ok ? await res.json() : [];
  const items = projs.map((p) => ({ label: p.name, onPick: () => setThreadProject(p.project_id) }));
  items.push({ label: "＋ New project…", onPick: () => newProject(true) });
  items.push({ label: "✕ Detach project", onPick: () => setThreadProject(null) });
  openPicker({ title: "Attach this task to a project", items, emptyText: "No projects yet" });
}
async function setThreadProject(projectId) {
  await fetch(`/api/threads/${threadId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId }),
  });
  await renderTaskAttach();
}
function detachProject() {
  setThreadProject(null);
}

async function attachLibraryFlow() {
  ensureThreadId();
  const [allRes, attRes] = await Promise.all([fetch("/api/libraries"), fetch(`/api/threads/${threadId}/libraries`)]);
  const all = (allRes.ok ? await allRes.json() : []).filter((l) => l.kind === "shared");
  const attached = new Set((attRes.ok ? await attRes.json() : []).map((l) => l.library_id));
  const available = all.filter((l) => !attached.has(l.library_id));
  const items = available.map((l) => ({ label: `${l.name || l.library_id} (${l.n_chunks || 0})`, onPick: () => attachLibrary(l.library_id) }));
  items.push({ label: "＋ New library…", onPick: () => newLibrary(true) });
  openPicker({ title: "Attach a library to this task", items, emptyText: "No shared libraries yet" });
}
async function attachLibrary(libraryId) {
  const res = await fetch(`/api/threads/${threadId}/libraries`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ library_id: libraryId }),
  });
  if (!res.ok) addStatusLine(`error: ${(await res.json()).error || "attach failed"}`);
  await renderTaskAttach();
}
async function detachLibrary(libraryId) {
  await fetch(`/api/threads/${threadId}/libraries?library_id=${encodeURIComponent(libraryId)}`, { method: "DELETE" });
  await renderTaskAttach();
}

// --- library manager screen -----------------------------------------------------
async function newLibrary(attachToTask) {
  const form = await openForm({
    title: "New library",
    okLabel: "Create",
    fields: [
      { key: "name", label: "Name", required: true, placeholder: "Product docs" },
      { key: "path", label: "Folder or file of docs — absolute path (optional)", placeholder: "/Users/you/docs" },
    ],
  });
  if (!form) return;
  const res = await fetch("/api/libraries", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: form.name }),
  });
  const data = await res.json();
  if (!res.ok) return showError(data.error || "Couldn't create library. Is the server up to date? Restart it.");
  if (attachToTask && threadId) await attachLibrary(data.library_id);
  if (form.path) await ingestPathIntoLibrary(data.library_id, form.path);
  setSection("libraries");
  openLibrary(data.library_id);
}

async function openLibrary(libraryId) {
  activeLibraryId = libraryId;
  showScreen("library");
  const listRes = await fetch("/api/libraries");
  const libs = listRes.ok ? await listRes.json() : [];
  const lib = libs.find((l) => l.library_id === libraryId) || { library_id: libraryId, name: libraryId, n_chunks: 0 };
  const srcRes = await fetch(`/api/libraries/${libraryId}/knowledge`);
  const srcs = srcRes.ok ? await srcRes.json() : [];
  renderLibraryView(lib, srcs);
  if (section === "libraries") loadNav();
}

function renderLibraryView(lib, sources) {
  const title = lib.name || lib.library_id;
  libraryViewEl.innerHTML = "";
  const head = document.createElement("div");
  head.className = "entity-head";
  head.innerHTML = `<div class="entity-title">${escapeHtml(title)}</div>
    <div class="entity-meta">${sources.length} source${sources.length === 1 ? "" : "s"} · ${lib.n_chunks || 0} chunks</div>`;
  const actions = document.createElement("div");
  actions.className = "entity-actions";
  actions.appendChild(btn("Rename", () => renameLibrary(lib.library_id)));
  actions.appendChild(btn("Delete", () => deleteLibrary(lib.library_id), "danger"));
  head.appendChild(actions);
  libraryViewEl.appendChild(head);

  const add = document.createElement("div");
  add.className = "entity-actions";
  add.appendChild(btn("Add folder", () => libraryAddFolder(lib.library_id)));
  add.appendChild(btn("Upload file", () => libraryUploadFile(lib.library_id)));
  add.appendChild(btn("Add text", () => libraryAddText(lib.library_id)));
  libraryViewEl.appendChild(add);

  libraryViewEl.appendChild(sectionHeading("Sources"));
  const list = document.createElement("div");
  list.className = "entity-list";
  if (!sources.length) list.appendChild(navEmpty("No sources yet"));
  for (const s of sources) {
    list.appendChild(rowItem(s.source, `${s.n_chunks} chunks`, () => removeLibrarySource(lib.library_id, s.source)));
  }
  libraryViewEl.appendChild(list);
}

async function libraryAddText(libraryId) {
  const text = await openModal({ title: "Add knowledge — paste text", multiline: true });
  if (!text) return;
  const res = await fetch(`/api/libraries/${libraryId}/knowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) showError((await res.json().catch(() => ({}))).error || "Couldn't add text");
  openLibrary(libraryId);
}
function libraryUploadFile(libraryId) {
  const inp = document.createElement("input");
  inp.type = "file";
  inp.addEventListener("change", async () => {
    if (!inp.files[0]) return;
    const form = new FormData();
    form.append("file", inp.files[0]);
    const res = await fetch(`/api/libraries/${libraryId}/knowledge`, { method: "POST", body: form });
    if (!res.ok) showError((await res.json().catch(() => ({}))).error || "Upload failed");
    openLibrary(libraryId);
  });
  inp.click();
}
async function libraryAddFolder(libraryId) {
  const path = await openModal({ title: "Add a folder or file of docs — absolute path", okLabel: "Ingest" });
  if (!path) return;
  await ingestPathIntoLibrary(libraryId, path);
  openLibrary(libraryId);
}
async function ingestPathIntoLibrary(libraryId, path) {
  const res = await fetch(`/api/libraries/${libraryId}/knowledge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) showError((await res.json().catch(() => ({}))).error || "Couldn't ingest that path");
}
async function removeLibrarySource(libraryId, source) {
  await fetch(`/api/libraries/${libraryId}/knowledge?source=${encodeURIComponent(source)}`, { method: "DELETE" });
  openLibrary(libraryId);
}
async function renameLibrary(libraryId) {
  const name = await openModal({ title: "Rename library", okLabel: "Save" });
  if (!name) return;
  await fetch(`/api/libraries/${libraryId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  openLibrary(libraryId);
  loadNav();
}
async function deleteLibrary(libraryId) {
  if (!(await openConfirm("Delete this library and all its knowledge? Attached tasks lose access."))) return;
  await fetch(`/api/libraries/${libraryId}`, { method: "DELETE" });
  activeLibraryId = null;
  showScreen("task");
  loadNav();
}
// --- project editor screen ------------------------------------------------------
async function newProject(attachToTask) {
  const form = await openForm({
    title: "New project",
    okLabel: "Create",
    fields: [
      { key: "name", label: "Name", required: true, placeholder: "Website revamp" },
      { key: "path", label: "Work folder — absolute path (optional)", placeholder: "/Users/you/project" },
      { key: "goals", label: "Goals (optional)", type: "textarea", placeholder: "What this project is trying to achieve — attached tasks see this as steering context." },
    ],
  });
  if (!form) return;
  const res = await fetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: form.name, goals: form.goals || "" }),
  });
  const data = await res.json();
  if (!res.ok) return showError(data.error || "Couldn't create project. Is the server up to date? Restart it.");
  if (attachToTask && threadId) await setThreadProject(data.project_id);
  if (form.path) {
    const dr = await fetch(`/api/projects/${data.project_id}/directories`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: form.path }),
    });
    if (!dr.ok) showError((await dr.json().catch(() => ({}))).error || "Couldn't add that directory");
  }
  setSection("projects");
  openProject(data.project_id);
}

async function openProject(projectId) {
  activeProjectId = projectId;
  showScreen("project");
  const res = await fetch(`/api/projects/${projectId}`);
  if (!res.ok) {
    projectViewEl.innerHTML = "";
    projectViewEl.appendChild(navEmpty("Project not found"));
    return;
  }
  renderProjectView(await res.json());
  if (section === "projects") loadNav();
}

function renderProjectView(project) {
  projectViewEl.innerHTML = "";
  const head = document.createElement("div");
  head.className = "entity-head";
  head.innerHTML = `<div class="entity-title">${escapeHtml(project.name)}</div>
    <div class="entity-meta">${project.directories.length} director${project.directories.length === 1 ? "y" : "ies"} · ${(project.tasks || []).length} task${(project.tasks || []).length === 1 ? "" : "s"}</div>`;
  const actions = document.createElement("div");
  actions.className = "entity-actions";
  actions.appendChild(btn("Rename", () => renameProject(project.project_id)));
  actions.appendChild(btn("Delete", () => deleteProject(project.project_id), "danger"));
  head.appendChild(actions);
  projectViewEl.appendChild(head);

  // goals
  projectViewEl.appendChild(sectionHeading("Goals"));
  const goals = document.createElement("textarea");
  goals.className = "goals-field";
  goals.rows = 4;
  goals.placeholder = "What is this project trying to achieve? Attached tasks see this as steering context.";
  goals.value = project.goals || "";
  projectViewEl.appendChild(goals);
  const saveRow = document.createElement("div");
  saveRow.className = "entity-actions";
  saveRow.appendChild(btn("Save goals", async () => {
    await fetch(`/api/projects/${project.project_id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goals: goals.value }),
    });
    addStatusLine("goals saved");
    loadNav();
  }));
  projectViewEl.appendChild(saveRow);

  // directories
  projectViewEl.appendChild(sectionHeading("Work directories"));
  const dirActions = document.createElement("div");
  dirActions.className = "entity-actions";
  dirActions.appendChild(btn("Add directory", () => addProjectDir(project.project_id)));
  projectViewEl.appendChild(dirActions);
  const dirList = document.createElement("div");
  dirList.className = "entity-list";
  if (!project.directories.length) dirList.appendChild(navEmpty("No directories yet"));
  for (const d of project.directories) dirList.appendChild(rowItem(d, "", () => removeProjectDir(project.project_id, d)));
  projectViewEl.appendChild(dirList);

  // tasks in project
  projectViewEl.appendChild(sectionHeading("Tasks in this project"));
  const taskList = document.createElement("div");
  taskList.className = "entity-list";
  if (!(project.tasks || []).length) taskList.appendChild(navEmpty("No tasks yet"));
  for (const t of project.tasks || []) taskList.appendChild(rowItem(t.title, "", null, () => openThread(t.thread_id)));
  projectViewEl.appendChild(taskList);
}

async function addProjectDir(projectId) {
  const path = await openModal({ title: "Add work directory — absolute path", okLabel: "Add" });
  if (!path) return;
  const res = await fetch(`/api/projects/${projectId}/directories`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) addStatusLine(`error: ${(await res.json()).error || "failed to add directory"}`);
  openProject(projectId);
}
async function removeProjectDir(projectId, path) {
  await fetch(`/api/projects/${projectId}/directories?path=${encodeURIComponent(path)}`, { method: "DELETE" });
  openProject(projectId);
}
async function renameProject(projectId) {
  const name = await openModal({ title: "Rename project", okLabel: "Save" });
  if (!name) return;
  await fetch(`/api/projects/${projectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  openProject(projectId);
  loadNav();
}
async function deleteProject(projectId) {
  if (!(await openConfirm("Delete this project? Its tasks are detached (not deleted)."))) return;
  await fetch(`/api/projects/${projectId}`, { method: "DELETE" });
  activeProjectId = null;
  showScreen("task");
  loadNav();
  if (threadId) renderTaskAttach();
}

// --- small shared widgets -------------------------------------------------------
function btn(label, onClick, variant) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "ent-btn" + (variant === "danger" ? " ent-btn-danger" : "");
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}
function sectionHeading(text) {
  const d = document.createElement("div");
  d.className = "entity-section-label";
  d.textContent = text;
  return d;
}
function rowItem(name, meta, onRemove, onOpen) {
  const item = document.createElement("div");
  item.className = "row-item";
  const label = document.createElement("span");
  label.className = "row-name" + (onOpen ? " row-link" : "");
  label.textContent = name;
  if (onOpen) label.addEventListener("click", onOpen);
  item.appendChild(label);
  if (meta) {
    const m = document.createElement("span");
    m.className = "row-meta";
    m.textContent = meta;
    item.appendChild(m);
  }
  if (onRemove) {
    const x = document.createElement("button");
    x.className = "row-remove";
    x.textContent = "×";
    x.title = "Remove";
    x.addEventListener("click", onRemove);
    item.appendChild(x);
  }
  return item;
}

// --- modal: single-field prompt / multi-field form / picker / confirm ----------
let modalResolve = null;
let modalMode = "field"; // field | form | picker
let formInputs = null;
let formRequired = [];

function resetModalBody() {
  modalTextarea.classList.add("hidden");
  modalInput.classList.add("hidden");
  modalList.classList.add("hidden");
  modalForm.classList.add("hidden");
  modalError.classList.add("hidden");
  modalError.textContent = "";
}

function openModal({ title, multiline = false, okLabel = "Add" }) {
  modalMode = "field";
  modalTitle.textContent = title;
  modalOkBtn.textContent = okLabel;
  resetModalBody();
  modalActions.classList.remove("hidden");
  const field = multiline ? modalTextarea : modalInput;
  field.classList.remove("hidden");
  field.value = "";
  modalOverlay.classList.remove("hidden");
  field.focus();
  return new Promise((resolve) => {
    modalResolve = resolve;
  });
}

// Multi-field form: fields = [{key, label, type?: "text"|"textarea", required?, placeholder?}].
// Resolves to an object of trimmed values, or null on cancel.
function openForm({ title, fields, okLabel = "Create" }) {
  modalMode = "form";
  modalTitle.textContent = title;
  modalOkBtn.textContent = okLabel;
  resetModalBody();
  modalActions.classList.remove("hidden");
  modalForm.classList.remove("hidden");
  modalForm.innerHTML = "";
  formInputs = {};
  formRequired = [];
  for (const f of fields) {
    const wrap = document.createElement("div");
    wrap.className = "form-field";
    const label = document.createElement("label");
    label.className = "form-label";
    label.textContent = f.label;
    wrap.appendChild(label);
    const el = f.type === "textarea" ? document.createElement("textarea") : document.createElement("input");
    if (f.type === "textarea") el.rows = 3;
    else el.type = "text";
    el.className = "modal-field";
    if (f.placeholder) el.placeholder = f.placeholder;
    wrap.appendChild(el);
    modalForm.appendChild(wrap);
    formInputs[f.key] = el;
    if (f.required) formRequired.push({ key: f.key, label: f.label });
  }
  modalOverlay.classList.remove("hidden");
  formInputs[fields[0].key].focus();
  return new Promise((resolve) => {
    modalResolve = resolve;
  });
}

function closeModal(value) {
  modalOverlay.classList.add("hidden");
  resetModalBody();
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
  if (e.target !== modalOverlay) return;
  if (modalMode === "picker") closePicker();
  else closeModal(null);
});
modalOkBtn.addEventListener("click", () => {
  if (modalMode === "form") {
    for (const { key, label } of formRequired) {
      if (!formInputs[key].value.trim()) return showModalError(`${label} is required.`);
    }
    const out = {};
    for (const k in formInputs) out[k] = formInputs[k].value.trim();
    return closeModal(out);
  }
  const field = modalTextarea.classList.contains("hidden") ? modalInput : modalTextarea;
  const value = field.value.trim();
  if (!value) return showModalError("This field is required.");
  closeModal(value);
});

function openPicker({ title, items, emptyText }) {
  modalMode = "picker";
  modalTitle.textContent = title;
  resetModalBody();
  modalActions.classList.add("hidden");
  modalList.classList.remove("hidden");
  modalList.innerHTML = "";
  if (!items.length) {
    const e = document.createElement("div");
    e.className = "nav-empty";
    e.textContent = emptyText || "Nothing here";
    modalList.appendChild(e);
  }
  for (const it of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "modal-list-item";
    b.textContent = it.label;
    b.addEventListener("click", () => {
      closePicker();
      it.onPick();
    });
    modalList.appendChild(b);
  }
  modalOverlay.classList.remove("hidden");
}
function closePicker() {
  modalOverlay.classList.add("hidden");
  resetModalBody();
  modalActions.classList.remove("hidden");
}

function showError(message) {
  // Visible error (the embedded webview silently blocks window.alert), reusing the picker modal.
  openPicker({ title: message, items: [{ label: "OK", onPick: () => {} }] });
}

function openConfirm(message) {
  return new Promise((resolve) => {
    openPicker({
      title: message,
      items: [
        { label: "Cancel", onPick: () => resolve(false) },
        { label: "Yes, continue", onPick: () => resolve(true) },
      ],
    });
  });
}

// --- boot -----------------------------------------------------------------------
setSection("tasks");
const initialThreadId = getUrlThreadId();
if (initialThreadId) {
  openThread(initialThreadId);
} else {
  newTask();
}
