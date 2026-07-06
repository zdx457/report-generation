console.log("[APP] app.js v2 已加载, 时间:", new Date().toISOString());

// 会话管理
const SESSION_ID = "web_" + Math.random().toString(36).slice(2, 10);

// DOM 元素
const chatContainer = document.getElementById("chatContainer");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const btnClear = document.getElementById("btnClear");
const btnMemory = document.getElementById("btnMemory");
const btnMemoryClose = document.getElementById("btnMemoryClose");
const memoryPanel = document.getElementById("memoryPanel");
const memoryPanelBody = document.getElementById("memoryPanelBody");
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

async function sendMessage() {
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
  let tokenEl = container.querySelector(".token-stream");
  if (!tokenEl) {
    tokenEl = document.createElement("div");
    tokenEl.className = "thinking-detail-block token-stream";
    tokenEl.style.cssText =
      "white-space:pre-wrap;font-family:monospace;font-size:13px;color:#ccc;padding:8px;background:#1a1a2e;border-radius:4px;max-height:300px;overflow-y:auto;";
    container.appendChild(tokenEl);
  }
  tokenEl.textContent += content;
  tokenEl.scrollTop = tokenEl.scrollHeight;
}

function finishThinking(container) {
  const status = container.querySelector(".thinking-status");
  if (status) status.textContent = "✅ 完成";
  container.classList.add("thinking-collapsed", "thinking-done");
}

function addAssistantMessage(content) {
  const msg = document.createElement("div");
  msg.className = "message assistant";
  msg.innerHTML = `
    <div class="message-role">💬 助手</div>
    <div class="bubble">${escapeHtml(content).replace(/\n/g, "<br>")}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}