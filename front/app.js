// 会话管理
const SESSION_ID = 'web_' + Math.random().toString(36).slice(2, 10);

// DOM 元素
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const btnSend = document.getElementById('btnSend');
const btnClear = document.getElementById('btnClear');
const btnInfo = document.getElementById('btnInfo');
const statusText = document.getElementById('statusText');

// 状态
let isProcessing = false;
let currentReader = null;           // 当前 SSE 读取器
let chatAbortController = null;     // 当前请求的 AbortController

// API 基础地址
const API_BASE = '/api';

// 发送按钮
btnSend.addEventListener('click', () => sendMessage());
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 清空按钮
btnClear.addEventListener('click', async () => {
  if (isProcessing) {
    // 强制取消当前请求
    cancelCurrentRequest();
    updateStatus('⚠️ 已中断当前请求，请稍后重试清空');
    return;
  }
  await doClearSession();
});

// 强制执行清空
async function doClearSession() {
  try {
    await fetch(`${API_BASE}/clear`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
  } catch (e) { /* ignore */ }
  chatContainer.innerHTML = '';
  chatContainer.appendChild(createEmptyState());
  updateStatus('✅ 会话已清空');
  addSystemMessage('🗑️ 会话已清空，记忆已重置');
}

// 状态按钮
btnInfo.addEventListener('click', async () => {
  console.log('[会话状态] 开始获取...');
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    console.log('[会话状态] 请求超时，取消');
    controller.abort();
  }, 15000);

  try {
    const resp = await fetch(`${API_BASE}/info?session_id=${SESSION_ID}&_t=${Date.now()}`, {
      method: 'GET',
      keepalive: false,
      cache: 'no-store',
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = await resp.json();
    console.log('[会话状态] 获取成功:', data);
    const turns = data.current_turns || 0;
    const entities = data.entity_count || 0;
    const accumulated = data.accumulated_searches || 0;
    addSystemMessage(
      `📊 会话状态 | 对话轮数: ${turns} | 实体数: ${entities} | 累积检索: ${accumulated}`
    );
    updateStatus(`会话ID: ${SESSION_ID.slice(0, 12)}... | ${turns}轮对话`);
  } catch (e) {
    clearTimeout(timeout);
    console.error('[会话状态] 获取失败:', e);
    if (e.name === 'AbortError') {
      addSystemMessage('⚠️ 获取会话状态超时，请稍后重试');
    } else {
      addSystemMessage('⚠️ 无法获取会话状态，请确认后端服务正在运行');
    }
    updateStatus('⚠️ 获取状态失败');
  }
});

// 取消当前 SSE 请求
function cancelCurrentRequest() {
  if (currentReader) {
    try { currentReader.cancel(); } catch (e) { /* ignore */ }
    currentReader = null;
  }
  if (chatAbortController) {
    try { chatAbortController.abort(); } catch (e) { /* ignore */ }
    chatAbortController = null;
  }
  setProcessing(false);
}

function createEmptyState() {
  const div = document.createElement('div');
  div.className = 'empty-state';
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

  // 先取消上一个未清理的请求
  cancelCurrentRequest();

  const es = chatContainer.querySelector('.empty-state');
  if (es) es.remove();

  addUserMessage(query);
  userInput.value = '';
  setProcessing(true);

  // 思考过程容器
  const thinking = addThinkingContainer();

  // 创建 AbortController（60秒超时）
  chatAbortController = new AbortController();
  const timeoutId = setTimeout(() => {
    chatAbortController.abort();
  }, 60000);
  let streamDone = false;

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      keepalive: false,
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        session_id: SESSION_ID,
      }),
      signal: chatAbortController.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    currentReader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let streamDone = false;

    while (true) {
      const { done, value } = await currentReader.read();
      if (done) {
        streamDone = true;
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;
          try {
            const event = JSON.parse(data);
            handleStreamEvent(event, thinking);
          } catch (e) { /* ignore */ }
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      addSystemMessage('⏱️ 请求超时，已自动取消');
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
      try { await currentReader.cancel(); } catch (e) { /* ignore */ }
    }

    currentReader = null;
    }

    chatAbortController = null;
    setProcessing(false);
    updateStatus('就绪');
  }

function handleStreamEvent(event, thinking) {
  switch (event.type) {
    case 'query_rewrite':
      addThinkingStep(thinking, '🔄', '查询改写', `${event.original} → ${event.rewritten}`);
      break;

    case 'context_resolve':
      addThinkingStep(thinking, '🔗', '上下文消解', `${event.original} → ${event.resolved}`);
      break;

    case 'search':
      addThinkingStep(thinking, '🔍', '检索', `正在检索: ${event.query}`);
      break;

    case 'recall':
      addRecallDetail(thinking, event);
      break;

    case 'rerank':
      addRerankDetail(thinking, event);
      break;

    case 'search_result':
      addSearchResultDetail(thinking, event);
      break;

    case 'reasoning':
      addReasoningStep(thinking, event.text);
      break;

    case 'report':
      // 收拢思考过程
      finalizeThinking(thinking);
      // 添加最终报告
      addReportMessage(event.content);
      break;

    case 'error':
      addSystemMessage(`❌ ${event.message}`);
      break;

    default:
      break;
  }
}

// ========== 思考过程容器 ==========

function addThinkingContainer() {
  const container = document.createElement('div');
  container.className = 'thinking-container';
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
  const body = container.querySelector('.thinking-body');
  const step = document.createElement('div');
  step.className = 'thinking-step';
  step.innerHTML = `
    <span class="thinking-step-icon">${icon}</span>
    <span class="thinking-step-label">${label}:</span>
    <span class="thinking-step-detail">${escapeHtml(detail)}</span>
  `;
  body.appendChild(step);
  scrollToBottom();
}

function addRecallDetail(container, data) {
  const body = container.querySelector('.thinking-body');
  const step = document.createElement('div');
  step.className = 'thinking-step thinking-recall';
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
  const body = container.querySelector('.thinking-body');
  const step = document.createElement('div');
  step.className = 'thinking-step thinking-rerank';
  let items = '';
  data.results.forEach((r, i) => {
    items += `
      <div class="rerank-item">
        <span class="rerank-rank">[${i + 1}]</span>
        <span class="rerank-score">${(r.score * 100).toFixed(1)}%</span>
        <span class="rerank-source">${escapeHtml(r.source)}</span>
        <span class="rerank-diagnosis">${escapeHtml(r.diagnosis)}</span>
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
  const body = container.querySelector('.thinking-body');
  const step = document.createElement('div');
  step.className = 'thinking-step thinking-search-result';
  const id = 'sr_' + Math.random().toString(36).slice(2, 8);
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
  const body = container.querySelector('.thinking-body');
  const step = document.createElement('div');
  step.className = 'thinking-step thinking-reasoning';
  const id = 'reasoning_' + Math.random().toString(36).slice(2, 8);

  if (text.length > 300) {
    const preview = text.slice(0, 300) + '...';
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
  const status = container.querySelector('.thinking-status');
  if (status) status.textContent = '✅ 完成';
  container.classList.add('thinking-done');
  scrollToBottom();
}

// ========== 消息气泡 ==========

function addUserMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message user';
  msg.innerHTML = `
    <div class="message-role">你</div>
    <div class="bubble">${escapeHtml(text)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function addReportMessage(content) {
  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.innerHTML = `
    <div class="message-role">📝 结构化报告</div>
    <div class="bubble">${renderMarkdown(content)}</div>
  `;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

function addSystemMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message system';
  msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatContainer.appendChild(msg);
  scrollToBottom();
}

// ========== 工具函数 ==========

function setProcessing(processing) {
  isProcessing = processing;
  btnSend.disabled = processing;
  btnClear.disabled = processing;
  btnInfo.disabled = processing;
  const btnText = btnSend.querySelector('.btn-text');
  const btnLoading = btnSend.querySelector('.btn-loading');
  if (processing) {
    if (btnText) btnText.style.display = 'none';
    if (btnLoading) btnLoading.style.display = 'inline';
    updateStatus('⏳ 处理中...');
  } else {
    if (btnText) btnText.style.display = 'inline';
    if (btnLoading) btnLoading.style.display = 'none';
  }
}

function updateStatus(text) {
  statusText.textContent = text;
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  let html = escapeHtml(text);

  // 标题
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

  // 粗体
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // 段落
  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  if (!html.startsWith('<h')) {
    html = '<p>' + html + '</p>';
  }

  return html;
}