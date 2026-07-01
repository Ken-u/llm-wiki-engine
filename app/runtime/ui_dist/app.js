const state = {
  abortController: null,
  wikiTreeLoaded: false,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderMarkdown(md) {
  const escaped = escapeHtml(md);
  const lines = escaped.split("\n");
  let html = "";
  let inCode = false;
  for (const line of lines) {
    if (line.startsWith("```")) {
      html += inCode ? "</code></pre>" : "<pre><code>";
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      html += `${line}\n`;
      continue;
    }
    if (line.startsWith("# ")) html += `<h1>${line.slice(2)}</h1>`;
    else if (line.startsWith("## ")) html += `<h2>${line.slice(3)}</h2>`;
    else if (line.startsWith("### ")) html += `<h3>${line.slice(4)}</h3>`;
    else if (line.trim() === "") html += "<br />";
    else html += `<p>${line.replace(/\[\[([^\]]+)\]\]/g, "<code>[[$1]]</code>")}</p>`;
  }
  if (inCode) html += "</code></pre>";
  return html;
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

function setView(name) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (name === "wiki" && !state.wikiTreeLoaded) loadWikiTree();
  if (name === "settings") refreshStatus();
}

function addMessage(role, text = "") {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  $("messages").appendChild(el);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function addTrace(type, payload) {
  if (!$("debugToggle").checked) return;
  const item = document.createElement("div");
  item.className = "trace-item";
  item.textContent = `${type}: ${JSON.stringify(payload)}`;
  $("traceList").prepend(item);
}

function addReference(ref) {
  const item = document.createElement("div");
  item.className = "ref-item";
  item.textContent = ref.type === "case"
    ? `Case ${ref.case_id} ${ref.title || ""}`
    : `${ref.path} ${ref.title || ""}`;
  item.onclick = () => {
    if (ref.type === "case") openCase(ref.case_id);
    else openWiki(ref.path, true);
  };
  $("referenceList").appendChild(item);
}

async function submitChat(event) {
  event.preventDefault();
  const message = $("chatInput").value.trim();
  if (!message || state.abortController) return;
  $("chatInput").value = "";
  $("traceList").innerHTML = "";
  $("referenceList").innerHTML = "";
  addMessage("user", message);
  const assistant = addMessage("assistant", "");
  state.abortController = new AbortController();
  $("stopChat").disabled = false;

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, stream: true, debug: $("debugToggle").checked }),
      signal: state.abortController.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(await resp.text());
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) handleSse(part, assistant);
    }
  } catch (err) {
    if (err.name !== "AbortError") assistant.textContent += `\n\n[error] ${err.message}`;
  } finally {
    state.abortController = null;
    $("stopChat").disabled = true;
  }
}

function handleSse(block, assistant) {
  const eventLine = block.split("\n").find((l) => l.startsWith("event:"));
  const dataLine = block.split("\n").find((l) => l.startsWith("data:"));
  if (!eventLine || !dataLine) return;
  const event = eventLine.slice(6).trim();
  const data = JSON.parse(dataLine.slice(5).trim());
  if (event === "token") {
    assistant.textContent += data.text || "";
    $("messages").scrollTop = $("messages").scrollHeight;
  } else if (event === "tool_call" || event === "tool_result") {
    addTrace(event, data);
  } else if (event === "done") {
    (data.references || []).forEach(addReference);
  } else if (event === "error") {
    assistant.textContent += `\n\n[error] ${data.message}`;
  }
}

function stopChat() {
  state.abortController?.abort();
}

async function loadWikiTree() {
  const tree = await api("/api/wiki");
  $("wikiTree").innerHTML = "";
  $("wikiTree").appendChild(renderTree(tree));
  state.wikiTreeLoaded = true;
}

function renderTree(nodes) {
  const root = document.createElement("div");
  for (const node of nodes) {
    if (node.type === "directory") {
      const label = document.createElement("div");
      label.className = "dir";
      label.textContent = node.name;
      root.appendChild(label);
      const children = document.createElement("div");
      children.className = "children";
      children.appendChild(renderTree(node.children || []));
      root.appendChild(children);
    } else {
      const btn = document.createElement("button");
      btn.textContent = node.title ? `${node.title} (${node.name})` : node.name;
      btn.onclick = () => openWiki(node.path);
      root.appendChild(btn);
    }
  }
  return root;
}

async function openWiki(path, drawer = false) {
  const data = await api(`/api/wiki/${encodeURIComponent(path).replaceAll("%2F", "/")}`);
  const html = renderMarkdown(data.content);
  if (drawer) {
    $("drawerTitle").textContent = data.meta?.title || path;
    $("drawerPath").textContent = path;
    $("drawerContent").innerHTML = html;
    $("drawer").classList.add("open");
  } else {
    setView("wiki");
    $("wikiPath").textContent = path;
    $("wikiPreview").innerHTML = html;
  }
}

async function openCase(caseId) {
  const data = await api(`/api/cases/${encodeURIComponent(caseId)}`);
  $("drawerTitle").textContent = data.title || caseId;
  $("drawerPath").textContent = `case ${caseId}`;
  $("drawerContent").innerHTML = renderMarkdown(data.raw_content || JSON.stringify(data, null, 2));
  $("drawer").classList.add("open");
}

async function runSearch(event) {
  event.preventDefault();
  const query = $("searchInput").value.trim();
  if (!query) return;
  $("wikiResults").innerHTML = "Searching...";
  $("caseResults").innerHTML = "Searching...";
  const [wiki, cases] = await Promise.allSettled([
    api("/api/search", { method: "POST", body: JSON.stringify({ query, mode: $("searchMode").value, top_k: 10 }) }),
    api("/api/cases/search", { method: "POST", body: JSON.stringify({ query, limit: 5 }) }),
  ]);
  renderWikiResults(wiki.status === "fulfilled" ? wiki.value.results : []);
  renderCaseResults(cases.status === "fulfilled" ? cases.value.results : []);
}

function renderWikiResults(results) {
  $("wikiResults").innerHTML = "";
  for (const r of results) {
    const item = document.createElement("div");
    item.className = "result-item";
    item.innerHTML = `<div class="result-title">${escapeHtml(r.title || r.page_id || r.path)}</div>
      <div class="result-meta">${escapeHtml(r.path)} · ${escapeHtml((r.sources || []).join(","))} · ${Number(r.score).toFixed(4)}</div>
      <div class="result-snippet">${escapeHtml(r.snippet || "")}</div>`;
    item.onclick = () => openWiki(r.path, true);
    $("wikiResults").appendChild(item);
  }
}

function renderCaseResults(results) {
  $("caseResults").innerHTML = "";
  for (const r of results) {
    const item = document.createElement("div");
    item.className = "result-item";
    item.innerHTML = `<div class="result-title">${escapeHtml(r.case_id)} ${escapeHtml(r.title || "")}</div>
      <div class="result-meta">${escapeHtml(r.domain || "")} · ${Number(r.score).toFixed(4)}</div>
      <div class="result-snippet">${escapeHtml(r.problem_summary || r.root_cause || r.resolution || "")}</div>`;
    item.onclick = () => openCase(r.case_id);
    $("caseResults").appendChild(item);
  }
}

async function refreshStatus() {
  const status = await api("/api/status");
  $("statusJson").textContent = JSON.stringify(status, null, 2);
  const ok = status.knowledge?.wiki_exists;
  $("statusPill").textContent = ok ? "Ready" : "Needs data";
  $("statusPill").style.color = ok ? "#067647" : "#b42318";
}

async function loadConfig() {
  $("configEditor").value = JSON.stringify(await api("/api/config"), null, 2);
}

async function saveConfig() {
  const body = JSON.parse($("configEditor").value);
  const saved = await api("/api/config", { method: "PUT", body: JSON.stringify(body) });
  $("configEditor").value = JSON.stringify(saved, null, 2);
  await refreshStatus();
}

async function runHooks() {
  await api("/api/hooks/run", { method: "POST", body: "{}" });
  await refreshStatus();
}

document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
$("chatForm").addEventListener("submit", submitChat);
$("stopChat").addEventListener("click", stopChat);
$("clearChat").addEventListener("click", () => { $("messages").innerHTML = ""; $("traceList").innerHTML = ""; $("referenceList").innerHTML = ""; });
$("refreshWiki").addEventListener("click", loadWikiTree);
$("searchForm").addEventListener("submit", runSearch);
$("refreshStatus").addEventListener("click", refreshStatus);
$("reloadConfig").addEventListener("click", loadConfig);
$("saveConfig").addEventListener("click", saveConfig);
$("runHooks").addEventListener("click", runHooks);
$("closeDrawer").addEventListener("click", () => $("drawer").classList.remove("open"));

refreshStatus().catch(() => { $("statusPill").textContent = "Status failed"; });
loadConfig().catch(() => {});

