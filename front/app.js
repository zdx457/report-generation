console.log("[APP] app.js v2 已加载, 时间:", new Date().toISOString());

// 会话管理 - 每个对话独立的 session_id
let SESSION_ID = generateSessionId();

function generateSessionId() {
  return "web_" + crypto.randomUUID();
}

// DOM 元素
const chatContainer = document.getElementById("chatContainer");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const btnClear = document.getElementById("btnClear");
const btnMemory = document.getElementById("btnMemory");
const btnMemoryClose = document.getElementById("btnMemoryClose");
const memoryPanel = document.getElementById("memoryPanel");
const memoryPanelBody = document.getElementById("memoryPanelBody");
const kbTotal = document.getElementById("kbTotal");
const kbMdCount = document.getElementById("kbMdCount");
const kbMeta = document.getElementById("kbMeta");
const btnBuildIncremental = document.getElementById("btnBuildIncremental");
const btnBuildRebuild = document.getElementById("btnBuildRebuild");
const btnExtractMeta = document.getElementById("btnExtractMeta");
const btnUploadXlsx = document.getElementById("btnUploadXlsx");
const xlsxInput = document.getElementById("xlsxInput");
const kbLog = document.getElementById("kbLog");
const kbLogContent = document.getElementById("kbLogContent");
const statusText = document.getElementById("statusText");

// 状态
let isProcessing = false;
let currentReader = null; // 当前 SSE 读取器
let chatAbortController = null; // 当前请求的 AbortController

// API 基础地址
const API_BASE = "/api";

// 发送按钮
btnSend.addEventListener("click", () => sendMessage());
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 清空按钮
btnClear.addEventListener("click", async () => {
  if (isProcessing) {
    // 强制取消当前请求
    cancelCurrentRequest();
    updateStatus("⚠️ 已中断当前请求，请稍后重试清空");
    return;
  }
  await doClearSession();
});

// 强制执行清空
async function doClearSession() {
  try {
    await fetch(`${API_BASE}/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
  } catch (e) {
    /* ignore */
  }
  chatContainer.innerHTML = "";
  chatContainer.appendChild(createEmptyState());
  updateStatus("✅ 会话已清空");
  addSystemMessage("🗑️ 会话已清空，记忆已重置");
}

// 短期记忆按钮
btnMemory.addEventListener("click", async () => {
  await loadMemoryPanel();
});

// 关闭短期记忆面板
btnMemoryClose.addEventListener("click", () => {
  closeMemoryPanel();
});

function closeMemoryPanel() {
  memoryPanel.style.display = "none";
  const overlay = document.querySelector(".memory-overlay");
  if (overlay) overlay.remove();
}

async function loadMemoryPanel() {
  memoryPanel.style.display = "block";

  let overlay = document.querySelector(".memory-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "memory-overlay";
    overlay.addEventListener("click", closeMemoryPanel);
    document.body.appendChild(overlay);
  }

  memoryPanelBody.innerHTML = '<div class="memory-loading">加载中...</div>';

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const resp = await fetch(
      `${API_BASE}/memory?session_id=${SESSION_ID}&_t=${Date.now()}`,
      {
        method: "GET",
        keepalive: false,
        cache: "no-store",
        signal: controller.signal,
      },
    );
    clearTimeout(timeout);

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    console.log("[短期记忆] 获取成功:", data);
    renderMemoryPanel(data);
  } catch (e) {
    clearTimeout(timeout);
    console.error("[短期记忆] 获取失败:", e);
    memoryPanelBody.innerHTML = `<div class="memory-error">⚠️ 获取短期记忆失败: ${escapeHtml(e.message)}</div>`;
  }
}

function renderMemoryPanel(data) {
  const turns = data.turns || [];
  const entities = data.entities || {};
  const summaries = data.summaries || [];
  const currentTurns = data.current_turns || 0;
  const totalTurns = data.total_turns || 0;
  const maxRounds = data.max_rounds || 5;

  let html = "";

  html += `<div class="memory-stats">
    <span class="memory-stat">当前轮数: <strong>${currentTurns}</strong> / ${maxRounds}</span>
    <span class="memory-stat">累计轮数: <strong>${totalTurns}</strong></span>
    <span class="memory-stat">实体数: <strong>${Object.keys(entities).length}</strong></span>
    <span class="memory-stat">摘要数: <strong>${summaries.length}</strong></span>
  </div>`;

  if (turns.length > 0) {
    html += '<div class="memory-section"><h4>对话历史</h4>';
    turns.forEach((turn) => {
      html += `
        <div class="memory-turn">
          <div class="memory-turn-header">第 ${turn.round} 轮</div>
          <div class="memory-turn-user"><span class="memory-role-label">👤 用户:</span> ${escapeHtml(turn.user)}</div>
          <div class="memory-turn-assistant"><span class="memory-role-label">🤖 助手:</span> ${escapeHtml(turn.assistant)}</div>
        </div>
      `;
    });
    html += "</div>";
  } else {
    html += '<div class="memory-empty">暂无对话历史</div>';
  }

  if (Object.keys(entities).length > 0) {
    html += '<div class="memory-section"><h4>实体追踪</h4><div class="memory-entities">';
    for (const [key, val] of Object.entries(entities)) {
      const displayVal = Array.isArray(val) ? val.join(", ") : val;
      html += `<div class="memory-entity"><span class="memory-entity-key">${escapeHtml(key)}:</span> ${escapeHtml(String(displayVal))}</div>`;
    }
    html += "</div></div>";
  }

  if (summaries.length > 0) {
    html += '<div class="memory-section"><h4>历史摘要（淘汰轮次压缩）</h4>';
    summaries.forEach((s, i) => {
      html += `<div class="memory-summary">${i + 1}. ${escapeHtml(s)}</div>`;
    });
    html += "</div>";
  }

  memoryPanelBody.innerHTML = html;
}

// ==================== 页面切换 ====================

const sidebarBtns = document.querySelectorAll(".sidebar-btn");
const pages = document.querySelectorAll(".page");

sidebarBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    // 关闭弹窗
    const panel = document.getElementById("llmConfigPanel");
    if (panel) panel.style.display = "none";

    const page = btn.dataset.page;
    sidebarBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    pages.forEach(p => p.classList.remove("active"));
    const pageMap = { agent: "pageAgent", kb: "pageKB", config: "pageConfig" };
    const target = document.getElementById(pageMap[page] || "pageAgent");
    if (target) target.classList.add("active");
    if (page === "kb") {
      refreshKBStatus();
    }
    if (page === "config") {
      loadConfig();
    }
  });
});

// ==================== 知识库管理 ====================

let kbAbortController = null;

async function refreshKBStatus() {
  try {
    const resp = await fetch(`${API_BASE}/kb/status?t=${Date.now()}`);
    const data = await resp.json();
    
    if (data.rebuilding) {
      kbTotal.textContent = "重建中...";
      kbTotal.style.color = "#ff9800";
    } else {
      kbTotal.textContent = data.total !== undefined ? data.total : "-";
      kbTotal.style.color = "";
    }
    
    kbMdCount.textContent = data.md_count !== undefined ? data.md_count : "-";
    kbMeta.textContent = data.metadata_exists ? "✅ 已生成" : "❌ 未生成";
    kbMeta.style.color = data.metadata_exists ? "#2e7d32" : "#e53935";
  } catch (e) {
    console.error("[KB] 获取状态失败:", e);
  }
  refreshKBFiles();
}

async function refreshKBFiles() {
  try {
    const resp = await fetch(`${API_BASE}/kb/files?t=${Date.now()}`);
    const data = await resp.json();
    const files = data.files || [];
    const kbFiles = document.getElementById("kbFiles");
    const kbFilesList = document.getElementById("kbFilesList");
    kbFiles.style.display = "block";
    if (files.length === 0) {
      kbFilesList.innerHTML = `<div class="kb-file-item kb-file-empty">暂无文件，请上传 xlsx</div>`;
      return;
    }
    kbFilesList.innerHTML = files.map(f => {
      const date = new Date(f.mtime * 1000).toLocaleString("zh-CN");
      const sizeKB = (f.size / 1024).toFixed(1);
      return `<div class="kb-file-item">
        <span class="kb-file-name">📄 ${escapeHtml(f.name)}</span>
        <span class="kb-file-info">${f.slice_count} 切片 · ${sizeKB} KB · ${date}</span>
      </div>`;
    }).join("");
  } catch (e) {
    console.error("[KB] 获取文件列表失败:", e);
  }
}

function setKBBtnsDisabled(disabled) {
  btnUploadXlsx.disabled = disabled;
  btnBuildIncremental.disabled = disabled;
  btnBuildRebuild.disabled = disabled;
  btnExtractMeta.disabled = disabled;
  if (disabled) {
    btnUploadXlsx.classList.add("btn-disabled");
    btnBuildIncremental.classList.add("btn-disabled");
    btnBuildRebuild.classList.add("btn-disabled");
    btnExtractMeta.classList.add("btn-disabled");
  } else {
    btnUploadXlsx.classList.remove("btn-disabled");
    btnBuildIncremental.classList.remove("btn-disabled");
    btnBuildRebuild.classList.remove("btn-disabled");
    btnExtractMeta.classList.remove("btn-disabled");
  }
}

async function runKBBuild(rebuild) {
  if (kbAbortController) {
    kbAbortController.abort();
  }
  kbAbortController = new AbortController();

  setKBBtnsDisabled(true);
  kbLog.style.display = "block";
  kbLogContent.innerHTML = "";

  const mode = rebuild ? "全量重建" : "增量构建";
  appendKBLog(`[${mode}] 开始执行...`, "info");

  try {
    const resp = await fetch(`${API_BASE}/kb/build`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rebuild, batch_size: 16 }),
      signal: kbAbortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") {
            appendKBLog("✅ 构建完成", "done");
            await refreshKBStatus();
            return;
          }
          try {
            const event = JSON.parse(payload);
            const level = event.level || "info";
            appendKBLog(event.msg || "", level);
          } catch (e) {
            appendKBLog(payload, "info");
          }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      appendKBLog("⚠️ 操作已取消", "error");
    } else {
      appendKBLog(`❌ 错误: ${e.message}`, "error");
    }
  } finally {
    setKBBtnsDisabled(false);
    kbAbortController = null;
  }
}

async function runExtractMetadata() {
  if (kbAbortController) {
    kbAbortController.abort();
  }
  kbAbortController = new AbortController();

  setKBBtnsDisabled(true);
  kbLog.style.display = "block";
  kbLogContent.innerHTML = "";
  appendKBLog("[提取元数据] 开始执行...", "info");

  try {
    const resp = await fetch(`${API_BASE}/kb/extract-metadata`, {
      method: "POST",
      signal: kbAbortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") {
            appendKBLog("✅ 提取完成", "done");
            await refreshKBStatus();
            return;
          }
          try {
            const event = JSON.parse(payload);
            const level = event.level || "info";
            appendKBLog(event.msg || "", level);
          } catch (e) {
            appendKBLog(payload, "info");
          }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      appendKBLog("⚠️ 操作已取消", "error");
    } else {
      appendKBLog(`❌ 错误: ${e.message}`, "error");
    }
  } finally {
    setKBBtnsDisabled(false);
    kbAbortController = null;
  }
}

function appendKBLog(msg, level) {
  // 进度类消息：覆盖上一条进度，而不是追加
  if (msg.includes("向量化进度")) {
    const lastProgress = kbLogContent.querySelector(".kb-log-progress");
    if (lastProgress) {
      lastProgress.textContent = msg;
      lastProgress.className = `kb-log-line kb-log-${level} kb-log-progress`;
      kbLogContent.scrollTop = kbLogContent.scrollHeight;
      return;
    }
  }
  
  const div = document.createElement("div");
  div.className = `kb-log-line kb-log-${level}`;
  if (msg.includes("向量化进度")) {
    div.classList.add("kb-log-progress");
  }
  div.textContent = msg;
  kbLogContent.appendChild(div);
  kbLogContent.scrollTop = kbLogContent.scrollHeight;
}

btnBuildIncremental.addEventListener("click", () => runKBBuild(false));
btnBuildRebuild.addEventListener("click", () => runKBBuild(true));
btnExtractMeta.addEventListener("click", () => runExtractMetadata());

// 上传 xlsx 并切片
btnUploadXlsx.addEventListener("click", () => xlsxInput.click());
xlsxInput.addEventListener("change", () => {
  if (xlsxInput.files.length > 0) {
    runUploadXlsx(xlsxInput.files[0]);
  }
});

async function runUploadXlsx(file) {
  if (kbAbortController) {
    kbAbortController.abort();
  }
  kbAbortController = new AbortController();

  setKBBtnsDisabled(true);
  kbLog.style.display = "block";
  kbLogContent.innerHTML = "";
  appendKBLog(`[上传切片] ${file.name}`, "info");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const resp = await fetch(`${API_BASE}/kb/upload`, {
      method: "POST",
      body: formData,
      signal: kbAbortController.signal,
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const payload = line.slice(6);
          if (payload === "[DONE]") {
            appendKBLog("✅ 上传切片完成", "done");
            xlsxInput.value = "";
            await refreshKBStatus();
            return;
          }
          try {
            const event = JSON.parse(payload);
            const level = event.level || "info";
            appendKBLog(event.msg || "", level);
          } catch (e) {
            appendKBLog(payload, "info");
          }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      appendKBLog("⚠️ 操作已取消", "error");
    } else {
      appendKBLog(`❌ 错误: ${e.message}`, "error");
    }
  } finally {
    setKBBtnsDisabled(false);
    kbAbortController = null;
    xlsxInput.value = "";
  }
}

// 取消当前 SSE 请求
function cancelCurrentRequest() {
  if (currentReader) {
    try {
      currentReader.cancel();
    } catch (e) {
      /* ignore */
    }
    currentReader = null;
  }
  if (chatAbortController) {
    try {
      chatAbortController.abort();
    } catch (e) {
      /* ignore */
    }
    chatAbortController = null;
  }
  setProcessing(false);
}

function createEmptyState() {
  const div = document.createElement("div");
  div.className = "empty-state";
  div.innerHTML = `
    <div class="empty-icon">📋</div>
    <p>输入CT/MRI等检查关键词，生成结构化影像报告</p>
    <p class="hint">例如：CT脑出血、MRI膝关节、脑梗</p>
  `;
  return div;
}

async function sendMessage(selectedDiagnosis) {
  const query = userInput.value.trim();
  if (!query || isProcessing) return;

  console.log("[DEBUG] sendMessage 开始, query:", query);

  // 先取消上一个未清理的请求
  cancelCurrentRequest();

  const es = chatContainer.querySelector(".empty-state");
  if (es) es.remove();

  addUserMessage(query);
  userInput.value = "";
  setProcessing(true);

  // 思考过程容器
  const thinking = addThinkingContainer();
  addThinkingStep(thinking, "👤", "用户提问", query);
  console.log("[DEBUG] thinking container 已创建");

  // 创建 AbortController（60秒超时）
  chatAbortController = new AbortController();
  const timeoutId = setTimeout(() => {
    chatAbortController.abort();
  }, 60000);
  let streamDone = false;

  try {
    console.log("[DEBUG] 开始 fetch POST /api/chat");
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      keepalive: false,
      cache: "no-store",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        session_id: SESSION_ID,
        selected_diagnosis: selectedDiagnosis || null,
      }),
      signal: chatAbortController.signal,
    });

    clearTimeout(timeoutId);
    console.log("[DEBUG] fetch 响应状态:", response.status);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    currentReader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let eventCount = 0;

    console.log("[DEBUG] 开始读取 SSE 流...");
    while (true) {
      const { done, value } = await currentReader.read();
      if (done) {
        console.log("[DEBUG] 流读取完成 (done=true)");
        streamDone = true;
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") {
            console.log("[DEBUG] 收到 [DONE], 共处理事件数:", eventCount);
            streamDone = true;
            break;
          }
          try {
            const event = JSON.parse(data);
            eventCount++;
            console.log("[DEBUG] 事件 #" + eventCount + " type=" + event.type, event);
            handleStreamEvent(event, thinking);
          } catch (e) {
            console.warn("[DEBUG] JSON 解析失败:", data.substring(0, 100), e);
          }
        }
      }
    }
  } catch (e) {
    console.error("[DEBUG] 请求异常:", e.name, e.message);
    if (e.name === "AbortError") {
      addSystemMessage("⏱️ 请求超时，已自动取消");
    } else {
      addSystemMessage(`❌ 请求失败: ${e.message}`);
    }
    if (thinking && thinking.parentNode) {
      thinking.remove();
    }
  } finally {
    clearTimeout(timeoutId);

    // 关键修复：仅在流未自然完成时 cancel。
    // 对已 done 的流调用 cancel 会破坏浏览器连接池回收。
    if (currentReader && !streamDone) {
      console.log("[DEBUG] finally: 取消 reader (streamDone=" + streamDone + ")");
      try {
        await currentReader.cancel();
      } catch (e) {
        /* ignore */
      }
    }

    currentReader = null;
  }

  chatAbortController = null;
  setProcessing(false);
  updateStatus("就绪");

  // 自动保存对话到历史记录
  if (!currentConversationId) {
    currentConversationId = generateConvId();
    const history = getChatHistory();
    history.unshift({
      id: currentConversationId,
      session_id: SESSION_ID,
      title: getConversationTitleFromChat(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    });
    saveChatHistory(history);
  }
  console.log(">>> 准备调用 saveCurrentConversation <<<");
  saveCurrentConversation();
  console.log(">>> saveCurrentConversation 调用完成 <<<");
  renderChatHistoryList();

  console.log("[DEBUG] sendMessage 结束");
}

function handleStreamEvent(event, thinking) {
  switch (event.type) {
    case "status":
      updateStatusText(event.message);
      break;

    case "intent":
      addThinkingStep(thinking, "🎯", "意图识别", event.intent);
      break;

    case "query_rewrite":
      addThinkingStep(
        thinking,
        "🔄",
        "查询改写",
        `${event.original} → ${event.rewritten}`,
      );
      break;

    case "context_resolve":
      addThinkingStep(
        thinking,
        "🔗",
        "上下文消解",
        `${event.original} → ${event.resolved}`,
      );
      break;

    case "search":
      addThinkingStep(thinking, "🔍", "检索", `正在检索: ${event.query}`);
      break;

    case "recall":
      addRecallDetail(thinking, event);
      break;

    case "rerank":
      addRerankDetail(thinking, event);
      break;

    case "search_result":
      addSearchResultDetail(thinking, event);
      break;

    case "token":
      addTokenStream(thinking, event.content);
      break;

    case "reasoning":
      addReasoningStep(thinking, event.text);
      break;

    case "report":
      finishThinking(thinking);
      addReportMessage(event.content);
      break;

    case "message":
      finishThinking(thinking);
      addAssistantMessage(event.content);
      break;

    case "error":
      finishThinking(thinking);
      addSystemMessage(`❌ ${event.message}`);
      break;

    case "entity_update":
      addThinkingStep(thinking, "📋", "实体更新", JSON.stringify(event.slots));
      break;

    case "intent_switch":
      addThinkingStep(thinking, "🔄", "切换意图", event.message);
      break;

    case "memory_retrieval":
      addMemoryRetrievalDetail(thinking, event);
      break;

    case "tool_executed":
      addToolExecutedDetail(thinking, event);
      break;

    case "ambiguous":
      console.log("[DEBUG] ambiguous 事件触发, options数量=", (event.options || []).length);
      finishThinking(thinking);
      addAmbiguousOptions(event);
      break;
  }
}

// ========== 思考过程容器 ==========

function addThinkingContainer() {
  const container = document.createElement("div");
  container.className = "thinking-container thinking-collapsed";
  container.innerHTML = `
    <div class="thinking-header" onclick="this.parentElement.classList.toggle('thinking-collapsed')">
      <span class="thinking-toggle">▼</span>
      <span>🧠 Agent 思考过程</span>
      <span class="thinking-status">进行中...</span>
    </div>
    <div class="thinking-body" id="thinkingBody"></div>
  `;
  chatContainer.appendChild(container);
  scrollToBottom();
  return container;
}

function addThinkingStep(container, icon, label, detail) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step";
  step.innerHTML = `
    <span class="thinking-step-icon">${icon}</span>
    <span class="thinking-step-label">${label}:</span>
    <span class="thinking-step-detail">${escapeHtml(detail)}</span>
  `;
  body.appendChild(step);
  scrollToBottom();
}

function addRecallDetail(container, data) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-recall";
  step.innerHTML = `
    <span class="thinking-step-icon">📊</span>
    <span class="thinking-step-label">多路召回详情</span>
    <div class="thinking-detail-block">
      <div>路径1（向量检索）: ${data.vector_count} 条</div>
      <div>路径2（元数据过滤）: ${data.metadata_count} 条</div>
      <div>路径3（关键词检索）: ${data.keyword_count} 条</div>
      <div class="thinking-highlight">合并去重: ${data.total_before} 条 → ${data.total_after} 条（去重 ${data.dedup} 条）</div>
    </div>
  `;
  body.appendChild(step);
  scrollToBottom();
}

function addRerankDetail(container, data) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-rerank";
  let items = "";
  data.results.forEach((r, i) => {
    const textId = "rerank_text_" + Math.random().toString(36).slice(2, 8);
    items += `
      <div class="rerank-item">
        <span class="rerank-rank">[${i + 1}]</span>
        <span class="rerank-score">${(r.score * 100).toFixed(1)}%</span>
        <span class="rerank-source">${escapeHtml(r.source)}</span>
        <span class="rerank-diagnosis">${escapeHtml(r.diagnosis)}</span>
      </div>
      <div class="rerank-text-toggle" onclick="
        var el=document.getElementById('${textId}');
        var btn=document.getElementById('${textId}_btn');
        if(el.classList.toggle('hidden')){
          btn.textContent='展开内容 ▸';
        }else{
          btn.textContent='收起内容 ▾';
        }
      " id="${textId}_btn">展开内容 ▸</div>
      <div class="rerank-text-content hidden" id="${textId}">
        <pre>${escapeHtml(r.text)}</pre>
      </div>`;
  });
  step.innerHTML = `
    <span class="thinking-step-icon">🎯</span>
    <span class="thinking-step-label">Rerank 重排序（top-${data.results.length}）</span>
    <div class="thinking-detail-block rerank-list">${items}</div>
  `;
  body.appendChild(step);
  scrollToBottom();
}

function addSearchResultDetail(container, event) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-search-result";
  const id = "sr_" + Math.random().toString(36).slice(2, 8);
  step.innerHTML = `
    <span class="thinking-step-icon">📋</span>
    <span class="thinking-step-label">检索结果内容</span>
    <span class="thinking-collapse-btn" onclick="document.getElementById('${id}').classList.toggle('hidden')">
      展开/收起
    </span>
    <div class="thinking-detail-block search-result-content hidden" id="${id}">
      <pre>${escapeHtml(event.result)}</pre>
    </div>
  `;
  body.appendChild(step);
  scrollToBottom();
}

function addReasoningStep(container, text) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-reasoning";
  const id = "reasoning_" + Math.random().toString(36).slice(2, 8);

  if (text.length > 300) {
    const preview = text.slice(0, 300) + "...";
    step.innerHTML = `
      <span class="thinking-step-icon">💭</span>
      <span class="thinking-step-label">模型推理</span>
      <span class="thinking-collapse-btn" onclick="
        var f=document.getElementById('${id}');
        var p=document.getElementById('${id}_preview');
        var b=document.getElementById('${id}_btn');
        if(f.classList.toggle('hidden')){
          p.classList.remove('hidden');
          b.textContent='展开全文';
        }else{
          p.classList.add('hidden');
          b.textContent='收起';
        }
      " id="${id}_btn">收起</span>
      <div class="thinking-detail-block reasoning-preview hidden" id="${id}_preview">${escapeHtml(preview)}</div>
      <div class="thinking-detail-block reasoning-full" id="${id}">${escapeHtml(text)}</div>
    `;
  } else {
    step.innerHTML = `
      <span class="thinking-step-icon">💭</span>
      <span class="thinking-step-label">模型推理</span>
      <div class="thinking-detail-block reasoning-full">${escapeHtml(text)}</div>
    `;
  }

  body.appendChild(step);
  scrollToBottom();
}

function finalizeThinking(container) {
  const status = container.querySelector(".thinking-status");
  if (status) status.textContent = "✅ 完成";
  container.classList.add("thinking-done");
  scrollToBottom();
}

// ========== 消息气泡 ==========

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.innerHTML = `
    <div class="message-role">你</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function addReportMessage(content) {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  msg.innerHTML = `
    <div class="message-role">📝 结构化报告</div>
    <div class="bubble">${renderMarkdown(content)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function addSystemMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message system";
  msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

// ========== 记忆检索详情 ==========

function addMemoryRetrievalDetail(container, data) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-memory";

  let html = `<span class="thinking-step-icon">🧠</span>
    <span class="thinking-step-label">记忆检索 <span style="color:#888;font-size:11px">(基于 "${escapeHtml(data.query || '')}")</span></span>
    <div class="thinking-detail-block">`;

  if (data.ltm && data.ltm.length > 0) {
    html += `<div class="thinking-highlight">📌 相关用户偏好 (LTM): ${data.ltm.length} 条</div>`;
    data.ltm.forEach(function (item) {
      html += `<div style="margin-left:12px;color:#7c3aed;">• ${escapeHtml(item)}</div>`;
    });
  } else {
    html += `<div style="color:#888;">📌 相关用户偏好 (LTM): 无</div>`;
  }

  if (data.stm && data.stm.length > 0) {
    html += `<div class="thinking-highlight" style="margin-top:8px;">💬 相关历史对话 (STM): ${data.stm.length} 条</div>`;
    data.stm.forEach(function (item) {
      var short = item.length > 80 ? item.slice(0, 80) + "..." : item;
      html += `<div style="margin-left:12px;color:#059669;">• ${escapeHtml(short)}</div>`;
    });
  } else {
    html += `<div style="color:#888;margin-top:8px;">💬 相关历史对话 (STM): 无</div>`;
  }

  html += `</div>`;
  step.innerHTML = html;
  body.appendChild(step);
  scrollToBottom();
}

// ========== 工具执行详情 ==========

function addToolExecutedDetail(container, data) {
  const body = container.querySelector(".thinking-body");
  const step = document.createElement("div");
  step.className = "thinking-step thinking-tool";

  var toolIcon = "🔧";
  var toolLabel = data.tool;
  if (data.tool === "rag_search") {
    toolIcon = "🔍";
    toolLabel = "RAG 检索";
  } else if (data.tool === "edit_report") {
    toolIcon = "✏️";
    toolLabel = "编辑报告";
  } else if (data.tool === "refine_report") {
    toolIcon = "🔄";
    toolLabel = "精炼报告";
  }

  var paramsHtml = "";
  if (data.params) {
    var keys = Object.keys(data.params);
    keys.forEach(function (k) {
      var v = data.params[k];
      if (typeof v === "string" && v.length > 100) {
        v = v.slice(0, 100) + "...";
      }
      paramsHtml += `<div style="margin-left:12px;color:#666;">• ${escapeHtml(k)}: ${escapeHtml(String(v))}</div>`;
    });
  }

  step.innerHTML = `
    <span class="thinking-step-icon">${toolIcon}</span>
    <span class="thinking-step-label">工具执行: ${toolLabel}</span>
    <div class="thinking-detail-block">
      ${paramsHtml}
      <div class="thinking-highlight">结果长度: ${data.result_length} 字符 ${data.is_final ? "✅ 最终结果" : "⏳ 继续处理"}</div>
    </div>
  `;
  body.appendChild(step);
  scrollToBottom();
}

// ========== 歧义选项 ==========

function addAmbiguousOptions(data) {
  console.log("[DEBUG] addAmbiguousOptions 被调用, options=", data.options);
  var question = data.question || "请选择：";
  var options = data.options || [];
  var scores = data.scores || [];

  var bubble = document.createElement("div");
  bubble.className = "message assistant";

  var optionsHtml = (options || []).map(function (opt, i) {
    var scoreText = scores[i] ? ' <span style="color:#999;font-size:0.8em;">(' + (scores[i] * 100).toFixed(1) + '%)</span>' : '';
    return '<button class="ambiguity-option" onclick="selectAmbiguityOption(\'' + escapeHtml(opt) + '\')" style="display:block;margin:4px 0;padding:8px 16px;border:1px solid #667eea;border-radius:8px;background:#f0f0ff;cursor:pointer;text-align:left;width:100%;">' + (i + 1) + '. ' + escapeHtml(opt) + scoreText + '</button>';
  }).join("");

  bubble.innerHTML = `
    <div class="bubble">
      <p>${escapeHtml(question)}</p>
      <div>${optionsHtml}</div>
      <p style="color:#999;font-size:0.8em;margin-top:8px;">点击选择，或直接输入完整描述</p>
    </div>
  `;
  chatContainer.appendChild(bubble);
  scrollToBottom();
}

window.selectAmbiguityOption = function (option) {
  var input = document.getElementById("userInput");
  if (input) {
    cancelCurrentRequest();
    setProcessing(false);
    input.value = option;
    sendMessage(option);
  }
};

// ========== 工具函数 ==========

function setProcessing(processing) {
  isProcessing = processing;
  btnSend.disabled = processing;
  btnClear.disabled = processing;
  const btnText = btnSend.querySelector(".btn-text");
  const btnLoading = btnSend.querySelector(".btn-loading");
  if (processing) {
    if (btnText) btnText.style.display = "none";
    if (btnLoading) btnLoading.style.display = "inline";
    updateStatus("⏳ 处理中...");
  } else {
    if (btnText) btnText.style.display = "inline";
    if (btnLoading) btnLoading.style.display = "none";
  }
}

function updateStatus(text) {
  statusText.textContent = text;
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);

  // 标题
  html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");

  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // 段落
  html = html.replace(/\n\n/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");

  if (!html.startsWith("<h")) {
    html = "<p>" + html + "</p>";
  }

  return html;
}

function updateStatusText(text) {
  statusText.textContent = text;
}

function addTokenStream(container, content) {
  const body = container.querySelector(".thinking-body");
  let tokenEl = body.querySelector(".token-stream");
  if (!tokenEl) {
    tokenEl = document.createElement("div");
    tokenEl.className = "thinking-detail-block token-stream";
    tokenEl.style.cssText =
      "white-space:pre-wrap;font-family:monospace;font-size:13px;color:#333;background:#fff;border:1px solid #e8e8e8;border-radius:6px;max-height:300px;overflow-y:auto;";
    body.appendChild(tokenEl);
  }
  tokenEl.textContent += content;
  tokenEl.scrollTop = tokenEl.scrollHeight;
}

function finishThinking(container) {
  const status = container.querySelector(".thinking-status");
  if (status) status.textContent = "✅ 完成";
  container.classList.add("thinking-collapsed", "thinking-done");
}

// ==================== 对话历史管理 ====================

const CHAT_HISTORY_KEY = "chat_history_list";
const CHAT_MESSAGES_PREFIX = "chat_messages_";

let currentConversationId = null;

function getChatHistory() {
  try {
    return JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || "[]");
  } catch (e) {
    return [];
  }
}

function saveChatHistory(list) {
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(list));
}

function getChatMessages(convId) {
  try {
    return JSON.parse(localStorage.getItem(CHAT_MESSAGES_PREFIX + convId) || "[]");
  } catch (e) {
    return [];
  }
}

function saveChatMessages(convId, messages) {
  localStorage.setItem(CHAT_MESSAGES_PREFIX + convId, JSON.stringify(messages));
}

function deleteChatMessages(convId) {
  localStorage.removeItem(CHAT_MESSAGES_PREFIX + convId);
}

function generateConvId() {
  return "conv_" + crypto.randomUUID();
}

function saveCurrentConversation() {
  console.log("[DEBUG] saveCurrentConversation 被调用, currentConversationId=", currentConversationId);
  const messages = [];
  chatContainer.querySelectorAll(".message").forEach(msg => {
    const isUser = msg.classList.contains("user");
    const isAssistant = msg.classList.contains("assistant");
    const isSystem = msg.classList.contains("system");
    const bubble = msg.querySelector(".bubble");
    if (bubble) {
      const entry = {
        role: isUser ? "user" : isAssistant ? "assistant" : "system",
        content: bubble.innerHTML,
      };
      if (isAssistant) {
        // 思考过程是 assistant 消息的前一个兄弟节点
        const prev = msg.previousElementSibling;
        console.log("[DEBUG] assistant msg, prev=", prev ? prev.className : "null");
        if (prev && prev.classList.contains("thinking-container")) {
          entry.thinking = prev.outerHTML;
          console.log("[DEBUG] ✅ 保存思考过程:", prev.outerHTML.length, "字符");
        } else {
          console.log("[DEBUG] ❌ 未找到思考过程");
        }
      }
      messages.push(entry);
    }
  });

  console.log("[DEBUG] 共保存", messages.length, "条消息");
  for (let i = 0; i < messages.length; i++) {
    console.log("[DEBUG] 消息[" + i + "] role=" + messages[i].role + " thinking=" + (messages[i].thinking ? messages[i].thinking.length + "字符" : "无"));
  }

  if (messages.length === 0) return;

  const title = getConversationTitle(messages);

  if (currentConversationId) {
    const history = getChatHistory();
    const idx = history.findIndex(h => h.id === currentConversationId);
    if (idx !== -1) {
      history[idx].title = title;
      history[idx].updatedAt = new Date().toISOString();
      saveChatHistory(history);
    }
    saveChatMessages(currentConversationId, messages);
  }
}

function getConversationTitle(messages) {
  const firstUserMsg = messages.find(m => m.role === "user");
  if (firstUserMsg) {
    const text = firstUserMsg.content.replace(/<[^>]*>/g, "").trim();
    return text.length > 30 ? text.slice(0, 30) + "..." : text;
  }
  return "新对话";
}

function getConversationTitleFromChat() {
  const userMsgs = chatContainer.querySelectorAll(".message.user .bubble");
  if (userMsgs.length > 0) {
    const text = userMsgs[0].textContent.trim();
    return text.length > 30 ? text.slice(0, 30) + "..." : text;
  }
  return "新对话";
}

function startNewConversation() {
  saveCurrentConversation();

  currentConversationId = generateConvId();
  SESSION_ID = generateSessionId();

  const history = getChatHistory();
  history.unshift({
    id: currentConversationId,
    session_id: SESSION_ID,
    title: "新对话",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });
  saveChatHistory(history);

  doClearSession();
  renderChatHistoryList();
}

function loadConversation(convId) {
  console.log("[DEBUG] loadConversation: convId=", convId, "currentConversationId=", currentConversationId);
  if (convId === currentConversationId) return;

  saveCurrentConversation();

  currentConversationId = convId;
  const history = getChatHistory();
  const entry = history.find(h => h.id === convId);
  if (entry && entry.session_id) {
    SESSION_ID = entry.session_id;
  }

  const messages = getChatMessages(convId);
  console.log("[DEBUG] loadConversation: 加载了", messages.length, "条消息");

  chatContainer.innerHTML = "";
  if (messages.length === 0) {
    chatContainer.appendChild(createEmptyState());
  } else {
    messages.forEach((msg, idx) => {
      console.log("[DEBUG] 渲染消息[" + idx + "] role=" + msg.role + " thinking=" + (msg.thinking ? msg.thinking.length + "字符" : "无"));
      if (msg.role === "assistant" && msg.thinking) {
        // 思考过程作为独立节点插入在 assistant 消息之前
        const thinkingContainer = document.createElement("div");
        thinkingContainer.innerHTML = msg.thinking;
        chatContainer.appendChild(thinkingContainer.firstElementChild);
        console.log("[DEBUG] ✅ 恢复思考过程:", msg.thinking.length, "字符");
      }
      const el = document.createElement("div");
      el.className = "message " + msg.role;
      if (msg.role === "user") {
        el.innerHTML = `<div class="message-role">你</div><div class="bubble">${msg.content}</div>`;
      } else if (msg.role === "assistant") {
        el.innerHTML = `<div class="message-role">📝 结构化报告</div><div class="bubble">${msg.content}</div>`;
      } else {
        el.innerHTML = `<div class="bubble">${msg.content}</div>`;
      }
      chatContainer.appendChild(el);
    });
    scrollToBottom();
  }

  renderChatHistoryList();
  updateStatus("✅ 已加载历史对话");
}

function deleteConversation(convId, e) {
  e.stopPropagation();

  const history = getChatHistory();
  const filtered = history.filter(h => h.id !== convId);
  saveChatHistory(filtered);
  deleteChatMessages(convId);

  if (convId === currentConversationId) {
    currentConversationId = null;
    SESSION_ID = generateSessionId();
    doClearSession();
    chatContainer.innerHTML = "";
    chatContainer.appendChild(createEmptyState());
  }

  renderChatHistoryList();
}

function renderChatHistoryList() {
  const list = document.getElementById("chatHistoryList");
  if (!list) return;

  const history = getChatHistory();
  if (history.length === 0) {
    list.innerHTML = '<div class="chat-history-empty">暂无历史对话</div>';
    return;
  }

  list.innerHTML = history.map(h => {
    const isActive = h.id === currentConversationId;
    const date = new Date(h.updatedAt || h.createdAt);
    const dateStr = formatDate(date);
    return `
      <div class="chat-history-item ${isActive ? "active" : ""}"
           onclick="loadConversation('${h.id}')">
        <div class="chat-history-item-title" title="${escapeHtml(h.title)}">${escapeHtml(h.title)}</div>
        <div class="chat-history-item-date">${dateStr}</div>
        <div class="chat-history-item-delete" onclick="deleteConversation('${h.id}', event)">🗑</div>
      </div>
    `;
  }).join("");
}

function formatDate(date) {
  const now = new Date();
  const diff = now - date;
  if (diff < 60000) return "刚刚";
  if (diff < 3600000) return Math.floor(diff / 60000) + "分钟前";
  if (diff < 86400000) return Math.floor(diff / 3600000) + "小时前";
  const month = date.getMonth() + 1;
  const day = date.getDate();
  return month + "/" + day;
}

// 初始化：加载历史列表
function initChatHistory() {
  const history = getChatHistory();
  console.log("[DEBUG] initChatHistory: 历史对话数=", history.length);
  if (history.length > 0) {
    // 刷新后自动加载最近一次对话
    console.log("[DEBUG] initChatHistory: 自动加载最近对话", history[0].id);
    loadConversation(history[0].id);
  } else {
    console.log("[DEBUG] initChatHistory: 无历史对话，新建会话");
    currentConversationId = null;
    SESSION_ID = generateSessionId();
    renderChatHistoryList();
  }
}

// 监听清空会话按钮，同时保存当前对话
const originalClearHandler = btnClear.onclick;
btnClear.addEventListener("click", () => {
  saveCurrentConversation();
  currentConversationId = null;
  SESSION_ID = generateSessionId();
  renderChatHistoryList();
});

// 新对话按钮
document.getElementById("btnNewChat").addEventListener("click", startNewConversation);

// 页面加载时初始化
initChatHistory();

// 页面刷新/关闭前自动保存当前对话
window.addEventListener("beforeunload", () => {
  saveCurrentConversation();
});

// ==================== 配置管理 ====================

// 模型列表配置：每个模型类型有哪些字段
const MODEL_DEFS = {
  llms: {
    listId: "modelListLlms",
    btnAddId: "btnAddLlm",
    fields: [
      { key: "name", label: "名称", type: "text", placeholder: "标识名称", sm: false },
      { key: "base_url", label: "API 地址", type: "text", placeholder: "http://...", sm: false },
      { key: "model", label: "模型名", type: "text", placeholder: "qwen36_27b", sm: false },
      { key: "api_key", label: "API Key", type: "password", placeholder: "可选", sm: false },
      { key: "max_tokens", label: "最大 Token", type: "number", sm: true },
      { key: "temperature", label: "温度", type: "number", step: "0.1", sm: true },
    ],
  },
  embeddings: {
    listId: "modelListEmbeddings",
    btnAddId: "btnAddEmbedding",
    fields: [
      { key: "name", label: "名称", type: "text", placeholder: "标识名称", sm: false },
      { key: "base_url", label: "API 地址", type: "text", placeholder: "http://...", sm: false },
      { key: "model", label: "模型名", type: "text", placeholder: "bge-m3", sm: false },
      { key: "api_key", label: "API Key", type: "password", placeholder: "可选", sm: false },
      { key: "dimension", label: "维度", type: "number", sm: true },
    ],
  },
  reranks: {
    listId: "modelListReranks",
    btnAddId: "btnAddRerank",
    fields: [
      { key: "name", label: "名称", type: "text", placeholder: "标识名称", sm: false },
      { key: "base_url", label: "API 地址", type: "text", placeholder: "https://...", sm: false },
      { key: "model", label: "模型名", type: "text", placeholder: "Qwen/Qwen3-VL-Reranker-8B", sm: false },
      { key: "api_key", label: "API Key", type: "password", placeholder: "可选", sm: false },
    ],
  },
};

// 非模型列表的简单字段映射
const SIMPLE_FIELDS = {
  "retrieval.rag_top_k": "cfg_retrieval_rag_top_k",
  "retrieval.rerank_top_k": "cfg_retrieval_rerank_top_k",
  "retrieval.collection_name": "cfg_retrieval_collection_name",
  "retrieval.db_path": "cfg_retrieval_db_path",
  "short_term_memory.max_rounds": "cfg_stm_max_rounds",
  "short_term_memory.decay_factor": "cfg_stm_decay_factor",
};

// 渲染一个模型列表
function renderModelList(sectionKey) {
  const def = MODEL_DEFS[sectionKey];
  const container = document.getElementById(def.listId);
  if (!container) return;
  container.innerHTML = "";
  const items = _configData[sectionKey] || [];
  items.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.index = idx;
    card.dataset.section = sectionKey;

    card.innerHTML = `
      <div class="model-card-header">
        <div class="model-card-title">${escapeHtml(item.name || `模型 ${idx + 1}`)}</div>
        <div class="model-card-actions">
          <span class="model-status model-status-clickable" title="点击测试 API 连通性">验证</span>
          <button type="button" class="btn btn-xs btn-danger model-remove-btn" title="删除此模型">✕</button>
        </div>
      </div>
      <div class="model-card-body">
        ${def.fields.map(f => {
          const val = item[f.key] !== undefined ? item[f.key] : "";
          return `
            <div class="model-field">
              <label class="model-field-label">${f.label}</label>
              <input type="${f.type}" ${f.step ? `step="${f.step}"` : ""}
                class="config-input ${f.sm ? "config-input-sm" : ""}"
                data-field="${f.key}" value="${escapeHtml(String(val))}"
                placeholder="${f.placeholder || ""}">
            </div>`;
        }).join("")}
      </div>
      <div class="model-card-error" style="display: none"></div>
    `;
    container.appendChild(card);
  });

  // 绑定状态点击（点击重新测试）
  container.querySelectorAll(".model-status").forEach(statusEl => {
    statusEl.addEventListener("click", async (e) => {
      const card = e.target.closest(".model-card");
      if (!card) return;
      await testModelConnection(sectionKey, card);
    });
  });

  // 绑定删除按钮
  container.querySelectorAll(".model-remove-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const card = e.target.closest(".model-card");
      if (!card) return;
      card.remove();
      reindexModelCards(sectionKey);
    });
  });
}

function reindexModelCards(sectionKey) {
  const def = MODEL_DEFS[sectionKey];
  const container = document.getElementById(def.listId);
  if (!container) return;
  const cards = container.querySelectorAll(".model-card");
  cards.forEach((card, idx) => {
    card.dataset.index = idx;
  });
}

function addModelItem(sectionKey) {
  const def = MODEL_DEFS[sectionKey];
  const newItem = {};
  def.fields.forEach(f => { newItem[f.key] = ""; });
  newItem.name = "new";
  if (!_configData[sectionKey]) _configData[sectionKey] = [];
  _configData[sectionKey].push(newItem);
  renderModelList(sectionKey);
}

// 收集模型列表数据
function collectModelData(sectionKey) {
  const def = MODEL_DEFS[sectionKey];
  const container = document.getElementById(def.listId);
  if (!container) return [];
  const cards = container.querySelectorAll(".model-card");
  const items = [];
  cards.forEach((card, idx) => {
    const item = {};
    def.fields.forEach(f => {
      const input = card.querySelector(`[data-field="${f.key}"]`);
      if (!input) return;
      let val = input.value.trim();
      if (input.type === "number" || input.step) {
        val = val === "" ? "" : Number(val);
        if (input.step === "0.1" && val !== "") val = parseFloat(val);
      }
      if (val !== "" || input.type === "password") {
        item[f.key] = val;
      }
    });
    items.push(item);
  });
  return items;
}

// 测试模型 API 连接
async function testModelConnection(sectionKey, card) {
  const def = MODEL_DEFS[sectionKey];
  const model = {};
  def.fields.forEach(f => {
    const input = card.querySelector(`[data-field="${f.key}"]`);
    if (!input) return;
    let val = input.value.trim();
    if (input.type === "number" || input.step) {
      val = val === "" ? "" : Number(val);
      if (input.step === "0.1" && val !== "") val = parseFloat(val);
    }
    if (val !== "" || input.type === "password") {
      model[f.key] = val;
    }
  });

  const errorEl = card.querySelector(".model-card-error");
  errorEl.style.display = "none";
  errorEl.textContent = "";

  if (!model.base_url) {
    errorEl.textContent = "❌ API 地址不能为空";
    errorEl.style.display = "block";
    return;
  }
  if (!model.model) {
    errorEl.textContent = "❌ 模型名不能为空";
    errorEl.style.display = "block";
    return;
  }

  const statusEl = card.querySelector(".model-status");
  statusEl.textContent = "验证中";
  statusEl.className = "model-status model-status-clickable model-status-verifying";

  try {
    const resp = await fetch(`${API_BASE}/test-model`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_config: model, model_type: sectionKey }),
    });
    const data = await resp.json();
    if (data.success) {
      statusEl.textContent = "✅可用";
      statusEl.className = "model-status model-status-clickable model-status-ok";
      errorEl.style.display = "none";
      errorEl.textContent = "";
    } else {
      statusEl.textContent = "❌不可用";
      statusEl.className = "model-status model-status-clickable model-status-fail";
      const modelName = model.name || '未命名';
      const errorMsg = data.error || data.message || '未知错误';
      console.error(`[模型验证] ${sectionKey} - ${modelName} 验证失败:`, errorMsg);
      errorEl.textContent = `❌ 验证失败: ${errorMsg}`;
      errorEl.style.display = "block";
    }
  } catch (e) {
    statusEl.textContent = "❌不可用";
    statusEl.className = "model-status model-status-clickable model-status-fail";
    const modelName = model.name || '未命名';
    console.error(`[模型验证] ${sectionKey} - ${modelName} 验证失败:`, e);
    errorEl.textContent = `❌ 验证失败: ${e.message}`;
    errorEl.style.display = "block";
  }
}

// 自动验证所有模型
async function testAllModels() {
  for (const sectionKey of Object.keys(MODEL_DEFS)) {
    const def = MODEL_DEFS[sectionKey];
    const container = document.getElementById(def.listId);
    if (!container) continue;
    const cards = container.querySelectorAll(".model-card");
    for (const card of cards) {
      await testModelConnection(sectionKey, card);
    }
  }
}

// 加载配置
let _configData = {};

async function loadConfig() {
  try {
    const resp = await fetch(`${API_BASE}/config?t=${Date.now()}`);
    const data = await resp.json();
    if (data.error) {
      showConfigHint("❌ " + data.error);
      return;
    }
    _configData = data.config || {};

    // 渲染模型列表
    Object.keys(MODEL_DEFS).forEach(sectionKey => {
      renderModelList(sectionKey);
    });

    // 填充简单字段
    for (const [key, elId] of Object.entries(SIMPLE_FIELDS)) {
      const el = document.getElementById(elId);
      if (!el) continue;
      const keys = key.split(".");
      let val = _configData;
      for (const k of keys) {
        if (val && typeof val === "object") val = val[k];
        else { val = undefined; break; }
      }
      if (val !== undefined && val !== null) {
        el.value = val;
      }
    }
    showConfigHint("✅ 配置已加载");
    // 刷新 Agent 页面的模型下拉框
    refreshModelSelect();
    // 自动验证所有模型
    testAllModels();
  } catch (e) {
    console.error("[Config] 加载配置失败:", e);
    showConfigHint("❌ 加载配置失败: " + e.message);
  }
}

function showConfigHint(msg) {
  const hint = document.getElementById("configHint");
  if (hint) hint.textContent = msg;
}

// 保存配置
async function saveConfig() {
  const btn = document.getElementById("btnSaveConfig");
  btn.disabled = true;
  btn.textContent = "⏳ 保存中...";
  try {
    const cfg = {};

    // 收集模型列表
    Object.keys(MODEL_DEFS).forEach(sectionKey => {
      cfg[sectionKey] = collectModelData(sectionKey);
    });

    // 收集简单字段
    for (const [key, elId] of Object.entries(SIMPLE_FIELDS)) {
      const el = document.getElementById(elId);
      if (!el) continue;
      const keys = key.split(".");
      let node = cfg;
      for (let i = 0; i < keys.length - 1; i++) {
        if (!node[keys[i]]) node[keys[i]] = {};
        node = node[keys[i]];
      }
      const lastKey = keys[keys.length - 1];
      let val = el.value.trim();
      if (el.type === "number" || el.step) {
        val = val === "" ? "" : Number(val);
        if (el.step === "0.1" && val !== "") val = parseFloat(val);
      }
      if (val !== "") {
        node[lastKey] = val;
      }
    }

    // 保留 active_models
    if (_configData.active_models) {
      cfg.active_models = _configData.active_models;
    }

    const resp = await fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfg }),
    });
    const data = await resp.json();
    if (data.status === "ok") {
      showConfigHint("✅ " + data.message);
    } else {
      showConfigHint("❌ " + (data.error || "保存失败"));
    }
  } catch (e) {
    console.error("[Config] 保存配置失败:", e);
    showConfigHint("❌ 保存失败: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "💾 保存配置";
  }
}

// 当前选中的 LLM 名称
let _currentLlmName = "";

// 刷新当前激活的 LLM 名称
function refreshModelSelect() {
  const llms = _configData.llms || [];
  const activeModels = _configData.active_models || {};
  const savedName = activeModels.chat_llm || "";
  if (savedName && llms.some(m => m.name === savedName)) {
    _currentLlmName = savedName;
  } else if (llms.length > 0) {
    _currentLlmName = llms[0].name || "";
  }
}
// 打开模型选择弹窗
function openLlmConfigPanel() {
  const panel = document.getElementById("llmConfigPanel");
  if (!panel) return;
  panel.style.display = "flex";

  const activeModels = _configData.active_models || {};

  // 填充 LLM 下拉框
  const llmSelect = document.getElementById("activeChatLlm");
  const llms = _configData.llms || [];
  llmSelect.innerHTML = llms.map((m, i) =>
    `<option value="${i}" ${m.name === activeModels.chat_llm ? "selected" : ""}>${escapeHtml(m.name || `模型 ${i + 1}`)}</option>`
  ).join("");

  // 填充 Embedding 下拉框
  const embSelect = document.getElementById("activeEmbedding");
  const embeddings = _configData.embeddings || [];
  embSelect.innerHTML = embeddings.map((m, i) =>
    `<option value="${i}" ${m.name === activeModels.embedding ? "selected" : ""}>${escapeHtml(m.name || `模型 ${i + 1}`)}</option>`
  ).join("");

  // 填充 Rerank 下拉框
  const rerankSelect = document.getElementById("activeRerank");
  const reranks = _configData.reranks || [];
  rerankSelect.innerHTML = reranks.map((m, i) =>
    `<option value="${i}" ${m.name === activeModels.rerank ? "selected" : ""}>${escapeHtml(m.name || `模型 ${i + 1}`)}</option>`
  ).join("");

  document.getElementById("llmConfigHint").textContent = "";
}

// 弹窗保存激活模型
async function saveLlmConfigFromPanel() {
  const hint = document.getElementById("llmConfigHint");
  const llms = _configData.llms || [];
  const embeddings = _configData.embeddings || [];
  const reranks = _configData.reranks || [];

  const llmIdx = parseInt(document.getElementById("activeChatLlm").value);
  const embIdx = parseInt(document.getElementById("activeEmbedding").value);
  const rerankIdx = parseInt(document.getElementById("activeRerank").value);

  const chatLlm = llms[llmIdx];
  const embedding = embeddings[embIdx];
  const rerank = reranks[rerankIdx];

  if (!chatLlm || !chatLlm.name) { hint.textContent = "❌ 请选择对话大语言模型"; return; }
  if (!embedding || !embedding.name) { hint.textContent = "❌ 请选择向量化模型"; return; }
  if (!rerank || !rerank.name) { hint.textContent = "❌ 请选择重排模型"; return; }

  if (!_configData.active_models) _configData.active_models = {};
  _configData.active_models.chat_llm = chatLlm.name;
  _configData.active_models.embedding = embedding.name;
  _configData.active_models.rerank = rerank.name;
  _currentLlmName = chatLlm.name;

  const btn = document.getElementById("btnSaveLlmConfig");
  btn.disabled = true;
  btn.textContent = "⏳ 保存中...";

  try {
    const cfg = { ..._configData };
    cfg.active_models = _configData.active_models;

    const resp = await fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: cfg }),
    });
    const data = await resp.json();
    if (data.status === "ok") {
      hint.textContent = "✅ " + data.message;
      // 保存后关闭弹窗
      setTimeout(() => {
        document.getElementById("llmConfigPanel").style.display = "none";
      }, 800);
    } else {
      hint.textContent = "❌ " + (data.error || "保存失败");
    }
  } catch (e) {
    hint.textContent = "❌ 保存失败: " + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "保存";
  }
}

  document.addEventListener("DOMContentLoaded", () => {
  const btnSaveConfig = document.getElementById("btnSaveConfig");
  if (btnSaveConfig) {
    btnSaveConfig.addEventListener("click", saveConfig);
  }
  // 绑定各模型列表的添加按钮
  Object.values(MODEL_DEFS).forEach(def => {
    const btn = document.getElementById(def.btnAddId);
    if (btn) {
      const sectionKey = Object.keys(MODEL_DEFS).find(k => MODEL_DEFS[k] === def);
      btn.addEventListener("click", () => addModelItem(sectionKey));
    }
  });
  // 绑定配置模型按钮 - 打开弹窗
  const btnConfigModel = document.getElementById("btnConfigModel");
  if (btnConfigModel) {
    btnConfigModel.addEventListener("click", () => {
      openLlmConfigPanel();
    });
  }
  // 绑定弹窗关闭
  const btnLlmConfigClose = document.getElementById("btnLlmConfigClose");
  if (btnLlmConfigClose) {
    btnLlmConfigClose.addEventListener("click", () => {
      document.getElementById("llmConfigPanel").style.display = "none";
    });
  }
  // 绑定弹窗保存按钮
  const btnSaveLlmConfig = document.getElementById("btnSaveLlmConfig");
  if (btnSaveLlmConfig) {
    btnSaveLlmConfig.addEventListener("click", () => {
      saveLlmConfigFromPanel();
    });
  }
  // 绑定配置导航切换
  const navItems = document.querySelectorAll(".config-nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const targetKey = item.dataset.config;
      navItems.forEach(i => i.classList.remove("active"));
      document.querySelectorAll(".config-group").forEach(g => g.style.display = "none");
      item.classList.add("active");
      const targetGroup = document.getElementById(`configGroup${targetKey.charAt(0).toUpperCase() + targetKey.slice(1)}`);
      if (targetGroup) {
        targetGroup.style.display = "block";
      }
    });
  });
  // 页面加载后立即加载配置，确保 Agent 页面打开时模型下拉框就有内容
  loadConfig();
});

function addAssistantMessage(content) {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  msg.innerHTML = `
    <div class="message-role">💬 助手</div>
    <div class="bubble">${renderMarkdown(content)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function addUserMessage(text) {
  const msg = document.createElement("div");
  msg.className = "message user";
  msg.innerHTML = `
    <div class="message-role">👤 用户</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}