"""ReAct 多轮推理对话终端（带 RAG 检索工具）

用法：
  python rag_chat.py              # 命令行交互模式
  python rag_chat.py --debug      # 显示调试信息
  python rag_chat.py --web        # 启动 Web 服务（默认端口 8000）
  python rag_chat.py --web --port 8080  # 指定端口
"""

import json
import os
import re
import sys
import time
import uuid
import asyncio
import queue
from functools import wraps
from typing import Optional, Callable, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv
from pymilvus import MilvusClient

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from rag.rerank import rerank_documents, get_rerank_config
from rag.retrieval import multi_recall
from rag.query_rewrite import (
    parse_query_keywords,
    is_too_vague,
    get_clarification,
    standardize_query,
    needs_rewrite,
    rewrite_query,
)

# ── Web 模式依赖（可选） ──
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# ── Web 会话存储 ──
_web_sessions: dict[str, dict[str, Any]] = {}

SESSION_ID = f"rag_react_{uuid.uuid4().hex[:8]}"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(ENV_PATH)

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36-27b")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_pipeline", "milvus_lite.db")
COLLECTION_NAME = "report_slices"
PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "prompt.md")

MAX_STEPS = 5
RAG_TOP_K = 5
RERANK_TOP_K = 3
MAX_CONTEXT_STEPS = 2


def retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,)):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"  ⚠️ {e}，{delay}s 后重试 ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
        return wrapper
    return decorator


def parse_react_output(text: str):
    text = text.strip()
    if "[FINAL]" in text:
        return "final", text.split("[FINAL]", 1)[1].strip()
    action_match = re.search(r'\[ACTION:\s*(\w+)\]\s*\n?(.*)', text, re.IGNORECASE | re.DOTALL)
    if action_match:
        action_type = action_match.group(1).strip().lower()
        raw = action_match.group(2).strip()
        action_input = raw.split("\n")[0].strip() if raw else ""
        return "action", (action_type, action_input)
    if "[CONTINUE]" in text:
        return "continue", text.split("[CONTINUE]", 1)[1].strip()
    return "continue", text


def load_system_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一个医疗影像报告分析助手。请根据检索到的参考信息回答用户问题。如果参考信息不足以回答问题，请如实说明。"


@retry()
def get_embedding(text):
    payload = {"model": EMBED_MODEL, "input": [text]}
    r = requests.post(EMBED_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def search_reports(query, top_k=RAG_TOP_K, rerank_top_k=RERANK_TOP_K, client=None, _emit=None):
    """RAG 检索：向量检索 + 多路召回 + Rerank，返回格式化文本"""
    if rerank_top_k > top_k:
        rerank_top_k = top_k

    query_vec = get_embedding(query)
    keywords = parse_query_keywords(query)

    candidates, recall_details = multi_recall(query_vec, keywords, top_k=top_k, client=client, return_details=True)

    if not candidates:
        return "未检索到相关报告。"

    vec_results = recall_details.get("vector", [])
    meta_results = recall_details.get("metadata", [])
    kw_results = recall_details.get("keyword", [])

    total_before = len(vec_results) + len(meta_results) + len(kw_results)
    dedup_count = total_before - len(candidates)

    if _emit:
        _emit("recall", {
            "vector_count": len(vec_results),
            "metadata_count": len(meta_results),
            "keyword_count": len(kw_results),
            "total_before": total_before,
            "total_after": len(candidates),
            "dedup": dedup_count,
        })

    details = []
    details.append(f"=== 多路召回详情 ===")
    details.append(f"路径1（向量检索）: {len(vec_results)} 条")
    details.append(f"路径2（元数据过滤）: {len(meta_results)} 条")
    details.append(f"路径3（关键词检索）: {len(kw_results)} 条")
    details.append(f"合并去重: {total_before} 条 → {len(candidates)} 条（去重 {dedup_count} 条）")
    details.append("")

    documents = [e["text"] for e in candidates]

    reranked_entities = []
    try:
        rerank_results = rerank_with_retry(query, documents, top_n=rerank_top_k)
        for rr in rerank_results:
            idx = rr.get("index", 0)
            if idx < len(candidates):
                rerank_score = rr.get("relevance_score", 0)
                entity = candidates[idx]
                entity["_rerank_score"] = rerank_score
                reranked_entities.append(entity)
    except Exception:
        reranked_entities = candidates[:rerank_top_k]

    if _emit:
        _emit("rerank", {
            "results": [
                {
                    "score": e.get("_rerank_score", 0),
                    "source": e.get("source", "未知"),
                    "diagnosis": e.get("诊断结论", ""),
                }
                for e in reranked_entities
            ]
        })

    contexts = []
    for i, entity in enumerate(reranked_entities, 1):
        rerank_score = entity.get("_rerank_score", 0)
        source = entity.get("source", "未知")
        diagnosis = entity.get("诊断结论", "")
        text = entity.get("text", "")
        parts = [f"【参考{i}】(Rerank分数: {rerank_score:.4f}, 来源: {source})"]
        if diagnosis:
            parts.append(f"诊断结论: {diagnosis}")
        parts.append(text)
        contexts.append("\n".join(parts))

    return "\n".join(details) + "\n" + "\n\n".join(contexts)


@retry()
def chat_stream(messages, max_tokens=2048, temperature=0.3, debug=False):
    """流式调用 LLM，返回完整文本。debug=True 时边生成边打印。"""
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120, stream=True)
    r.raise_for_status()

    full_text = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if debug:
                        print(content, end="", flush=True)
                    full_text += content
            except json.JSONDecodeError:
                continue
    if debug:
        print()
    return full_text.strip()


@retry()
def summarize_fn(messages: list[dict]) -> str:
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": 80,
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=15)
        r.raise_for_status()
        obj = r.json()
        return obj["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


REACT_SYSTEM_PROMPT = """你是一个具备多步推理能力的 AI 助手，你有访问报告数据库的检索工具。

## 输出格式

每轮你必须输出以下三种格式之一：

### 继续推理
```
[CONTINUE]
你对当前问题的推理分析（可以是一段话，也可以是多点分析）
```

### 调用检索
```
[ACTION: search]
检索查询词（简洁的搜索关键词）
```

### 最终回答
```
[FINAL]
你的最终回答（Markdown 格式，基于检索结果和推理给出准确回答）
```

## 工作方式

1. 收到问题后，先判断是否需要检索：
   - 需要检索 → 输出 [ACTION: search] 进行检索
   - 不需要检索 → 输出 [CONTINUE] 进行推理
2. 检索结果会以"观察"形式返回给你
3. 综合分析检索结果和推理，判断是否需要继续：
   - 信息不足 → 继续 [ACTION: search] 或 [CONTINUE]
   - 信息充分 → 输出 [FINAL] 给出最终回答

## 修改/编辑场景（重要）

当用户要求修改、调整、变更、替换之前生成的内容时，**绝对不要检索**：
- 用户说的是"修改CT值为70"、"把诊断改成XX"、"调整一下格式"等 → 这是编辑请求
- 编辑请求不需要检索知识库，知识库里没有用户想要的具体数值
- 直接基于对话历史中的上一轮报告，找到要修改的部分，用 [CONTINUE] 推理修改方案，然后 [FINAL] 输出修改后的完整报告
- 修改时只改变用户指定的内容，其余部分保持原样

## 多病变追加规则

当用户输入的是与上一轮不同的新病变时，需要将新旧病变合并到同一份报告中：
- 检索新病变的信息
- 将新病变的「影像学表现」追加到上一轮报告的对应模块中，两个病变之间空一行分隔
- 将新病变的「诊断意见」追加到上一轮报告的对应模块中，用分号分隔
- 如果用户输入的是修改请求（见上条规则），则不要追加，只修改

## 重要规则

- 第一轮不要直接输出 [FINAL]，至少先检索或推理一步
- 检索时使用简洁的关键词，不要用完整句子
- [FINAL] 回答要基于检索结果，注明信息来源（如"参考1显示..."）
- 如果检索结果不足以回答问题，如实说明"""

MULTI_DISEASE_PROMPT = """
## 多病变合并输出规则

当收到多份不同病变的检索结果时，需要将所有病变合并到同一份报告中：

1. 「影像学表现」模块：按病变顺序依次描述每个病变，每个病变之间空一行分隔，可在每个病变前标注病变名称（如"CT脑出血："）
2. 「诊断意见」模块：按病变顺序依次列出每个病变的诊断意见，用分号分隔
3. 每个病变的书写规则与单病变规则一致（忠实原文、分行排版、病灶独立拆分等）
4. 输出格式仍为 Markdown，仅包含「一、影像学表现」和「二、诊断意见」两个模块，不得出现多个独立的影像学表现或诊断意见模块
"""


@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)


def generate_merged_report(accumulated_searches, base_sys_prompt):
    """将累积的多份检索结果合并，生成一份包含所有病变的结构化报告"""
    merged_prompt = base_sys_prompt + "\n\n" + MULTI_DISEASE_PROMPT

    parts = []
    for i, entry in enumerate(accumulated_searches, 1):
        parts.append(f"### 病变{i}：{entry['query']}\n{entry['result']}")

    user_content = (
        "以下是多份不同病变的检索结果，请将它们合并为一份完整的结构化报告，"
        "在同一份报告的「影像学表现」和「诊断意见」中按顺序描述所有病变：\n\n"
        + "\n\n".join(parts)
    )

    messages = [
        {"role": "system", "content": merged_prompt},
        {"role": "user", "content": user_content},
    ]

    print("\n📋 正在生成合并报告...")
    return chat_stream(messages, max_tokens=4096, temperature=0.3, debug=True)


def run_react_with_events(query, session_id, stm, ltm, client, accumulated_searches, last_report, _emit):
    """Web 模式的 ReAct 管道，通过 _emit 发送 SSE 事件"""

    accumulated_searches.clear()

    if is_too_vague(query):
        clarification = get_clarification(query)
        _emit("error", {"message": f"查询过于模糊: {clarification}"})
        return

    original_query = query
    query = standardize_query(query)

    enhanced = stm.resolve_context(session_id, query)
    if enhanced != query:
        _emit("context_resolve", {"original": query, "resolved": enhanced})
    query = enhanced

    if needs_rewrite(query):
        rewritten = rewrite_query(query)
        if rewritten != query:
            _emit("query_rewrite", {"original": original_query, "rewritten": rewritten})
            query = rewritten

    _emit("search", {"query": query})

    base_sys_prompt = load_system_prompt()
    sys_prompt = base_sys_prompt + "\n\n" + REACT_SYSTEM_PROMPT
    pref_prompt = ltm.get_preference_prompt()
    history = stm.get_history(session_id)

    reasoning_steps = []
    search_results = []
    final_answer = None
    step = 0

    messages = [{"role": "system", "content": sys_prompt}]
    if pref_prompt:
        messages.append({"role": "system", "content": pref_prompt})
    # if last_report and last_report[0]:
    #     messages.append({"role": "system", "content": f"上一轮生成的报告：\n{last_report[0]}"})
    for msg in history:
        if msg.get("content", "").strip():
            messages.append(msg)
    messages.append({"role": "user", "content": f"用户问题：{enhanced}"})

    try:
        while step < MAX_STEPS:
            step += 1
            try:
                output = chat_stream(messages, max_tokens=2048, temperature=0.3, debug=True)
            except Exception as e:
                _emit("error", {"message": f"LLM 调用失败: {e}"})
                final_answer = f"抱歉，调用模型时出错: {e}"
                break

            action_type, payload = parse_react_output(output)

            if action_type == "final":
                final_answer = payload
                messages.append({"role": "assistant", "content": output})
                break

            elif action_type == "action":
                action_name, action_input = payload
                if not action_input:
                    action_input = enhanced

                # 过滤无效检索词
                if not action_input.strip() or action_input.strip() in ("无", "none", "None", "无相关", "不需要"):
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": "无需检索，请直接输出 [FINAL] 回答。"})
                    continue

                _emit("search", {"query": action_input})
                messages.append({"role": "assistant", "content": output})
                try:
                    search_result = search_reports(action_input, client=client, _emit=_emit)
                    _emit("search_result", {"query": action_input, "result": search_result})
                except Exception as e:
                    search_result = f"检索失败: {e}"
                    _emit("error", {"message": search_result})

                search_results.append((action_input, search_result))
                reasoning_steps.append(f"[检索] {action_input}\n→ 返回 {len(search_result)} 字符")
                messages.append({"role": "user", "content": f"观察（检索结果）：{search_result}\n请判断下一步。"})

            else:
                reasoning_text = payload
                _emit("reasoning", {"text": reasoning_text})
                reasoning_steps.append(reasoning_text)
                messages.append({"role": "assistant", "content": reasoning_text})

        if final_answer is None:
            messages.append({"role": "user", "content": "请基于以上推理和检索结果，输出最终回答。只输出 [FINAL] 和你的回答。\n\n[FINAL]"})
            try:
                force_output = chat_stream(messages, max_tokens=2048, temperature=0.3, debug=True)
                force_output = force_output.strip()
                if "[FINAL]" in force_output:
                    idx = force_output.find("[FINAL]")
                    final_answer = force_output[idx + len("[FINAL]"):].strip()
                else:
                    final_answer = force_output
            except Exception as e:
                final_answer = f"抱歉，调用模型时出错: {e}"

        # 收集本轮检索结果到累积池
        for action_input, search_result in search_results:
            accumulated_searches.append({
                "query": action_input,
                "result": search_result,
            })

        # 生成报告：只有当累积了多个不同病变时才合并输出
        if last_report and last_report[0]:
    # 跨轮：新检索 + 上一轮报告 → 合并
            final_answer = generate_merged_report(accumulated_searches, base_sys_prompt, last_report[0])

        elif len(accumulated_searches) >= 2:
            try:
                final_answer = generate_merged_report(accumulated_searches, base_sys_prompt)
            except Exception as e:
                final_answer = f"抱歉，生成合并报告时出错: {e}"
                _emit("error", {"message": final_answer})
        elif not final_answer:
            final_answer = "未检索到相关信息，无法生成报告。"

        _emit("report", {"content": final_answer})

    finally:
        if final_answer:
            last_report[0] = final_answer
            stm.add_turn(session_id, query, final_answer)


def web_main(port=8000):
    """启动 Web 服务"""
    if not WEB_AVAILABLE:
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print("向量数据库不存在，请先运行 build_vector_db.py", file=sys.stderr)
        sys.exit(1)

    app = FastAPI(title="影像报告生成系统")

    front_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "front")
    if os.path.isdir(front_dir):
        app.mount("/static", StaticFiles(directory=front_dir), name="static")

    @app.get("/")
    async def index():
        index_path = os.path.join(front_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "影像报告生成系统 API", "docs": "/docs"}

    def _get_or_create_session(session_id):
        if session_id not in _web_sessions:
            client = MilvusClient(uri=DB_PATH)
            client.load_collection(COLLECTION_NAME)
            stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
            ltm = LongTermMemory(user_id=session_id)
            _web_sessions[session_id] = {
                "stm": stm,
                "ltm": ltm,
                "client": client,
                "accumulated_searches": [],
                "last_report": [""],
            }
        return _web_sessions[session_id]

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        query = body.get("query", "").strip()
        session_id = body.get("session_id", f"web_{uuid.uuid4().hex[:8]}")

        if not query:
            return StreamingResponse(
                _sse_error("查询内容不能为空"),
                media_type="text/event-stream",
            )

        session = _get_or_create_session(session_id)

        async def event_generator():
            q = asyncio.Queue()

            def _emit_sse(event_type, data):
                try:
                    q.put_nowait(json.dumps({"type": event_type, **data}))
                except Exception:
                    pass

            def _run_pipeline():
                try:
                    run_react_with_events(
                        query, session_id,
                        session["stm"], session["ltm"], session["client"],
                        session["accumulated_searches"],
                        session["last_report"],
                        _emit_sse,
                    )
                except Exception as e:
                    try:
                        q.put_nowait(json.dumps({"type": "error", "message": str(e)}))
                    except Exception:
                        pass
                finally:
                    try:
                        q.put_nowait("[DONE]")
                    except Exception:
                        pass

            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _run_pipeline)

            start_time = time.time()
            OVERALL_TIMEOUT = 300  # 5 分钟整体超时

            while True:
                elapsed = time.time() - start_time
                if elapsed > OVERALL_TIMEOUT:
                    yield "data: {\"type\":\"error\",\"message\":\"整体请求超时\"}\n\n"
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30)
                except asyncio.TimeoutError:
                    continue
                if event == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {event}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "close",
            },
        )

    @app.post("/api/clear")
    async def clear(request: Request):
        body = await request.json()
        session_id = body.get("session_id", "")
        if session_id in _web_sessions:
            session = _web_sessions[session_id]
            session["stm"].clear(session_id)
            session["accumulated_searches"].clear()
        return {"status": "ok"}

    @app.get("/api/info")
    async def info(session_id: str = ""):
        if session_id not in _web_sessions:
            return {"current_turns": 0}
        session = _web_sessions[session_id]
        session_info = session["stm"].session_info(session_id)
        return {
            "current_turns": session_info.get("current_turns", 0),
            "entity_count": session_info.get("entity_count", 0),
            "accumulated_searches": len(session["accumulated_searches"]),
        }

    print(f"\n{'='*60}")
    print(f"  影像报告生成系统 Web 服务")
    print(f"  {'='*60}")
    print(f"  地址: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  生成模型: {CHAT_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}")
    print(f"  Rerank: {get_rerank_config()['rerank_model']}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


async def _sse_error(message):
    yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
    yield "data: [DONE]\n\n"


def main():
    debug = "--debug" in sys.argv

    if "--web" in sys.argv:
        port = 8000
        try:
            port_idx = sys.argv.index("--port")
            port = int(sys.argv[port_idx + 1])
        except (ValueError, IndexError):
            pass
        web_main(port=port)
        return

    if not os.path.exists(DB_PATH):
        print("向量数据库不存在，请先运行 build_vector_db.py", file=sys.stderr)
        sys.exit(1)

    stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
    ltm = LongTermMemory(user_id=SESSION_ID)

    client = MilvusClient(uri=DB_PATH)
    client.load_collection(COLLECTION_NAME)

    print("=" * 60)
    print("=== ReAct 多轮推理对话（带 RAG 检索） ===")
    print("=" * 60)
    print(f"生成模型: {CHAT_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Rerank: {get_rerank_config()['rerank_model']}")
    print(f"检索返回: top-{RAG_TOP_K} → rerank top-{RERANK_TOP_K}")
    print(f"最大推理步数: {MAX_STEPS}")
    print(f"用户ID: {ltm.user_id}")
    print()
    print("命令:")
    print("  exit/quit - 退出")
    print("  clear     - 清空会话")
    print("  info      - 查看记忆状态")
    print("  直接输入   - 进入 ReAct 推理循环（[CONTINUE]→[ACTION]→[FINAL]）")
    print()

    accumulated_searches = []  # 跨轮次累积检索结果池

    try:
        while True:
            try:
                try:
                    user_input = input("你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见！")
                    break

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("再见！")
                    break

                if user_input.lower() == "clear":
                    stm.clear(SESSION_ID)
                    accumulated_searches.clear()
                    print("🧹 会话已清空\n")
                    continue

                if user_input.lower() == "info":
                    print(f"📊 记忆状态:")
                    session_info = stm.session_info(SESSION_ID)
                    print(f"   短期记忆: {session_info['current_turns']} 轮, {session_info['entity_count']} 个实体, {session_info['summary_count']} 条摘要")
                    ltm_info = ltm.get_stats()
                    print(f"   长期记忆: {ltm_info['total_sessions']} 次会话, {ltm_info['total_turns']} 轮")
                    entities = stm.get_entities(SESSION_ID)
                    if entities:
                        print(f"   当前实体: {entities}")
                    summaries = stm.get_summaries(SESSION_ID)
                    if summaries:
                        print(f"   历史摘要:")
                        for i, s in enumerate(summaries, 1):
                            print(f"     {i}. {s}")
                    print()
                    continue

                query = user_input

                # ── 模糊查询检测 ──
                if is_too_vague(query):
                    clarification = get_clarification(query)
                    print(f"⚠️ 查询过于模糊: {clarification}")
                    print()
                    continue

                # ── 查询标准化 + 上下文消解 + 改写 ──
                original_query = query
                query = standardize_query(query)

                enhanced = stm.resolve_context(SESSION_ID, query)
                if debug and enhanced != query:
                    print(f"🔗 上下文消解: '{query}' → '{enhanced}'")
                query = enhanced

                if needs_rewrite(query):
                    rewritten = rewrite_query(query)
                    if rewritten != query:
                        print(f"🔄 查询改写: '{original_query}' → '{rewritten}'")
                        query = rewritten
                elif query != original_query:
                    print(f"📝 查询标准化: '{original_query}' → '{query}'")

                base_sys_prompt = load_system_prompt()
                sys_prompt = base_sys_prompt + "\n\n" + REACT_SYSTEM_PROMPT
                pref_prompt = ltm.get_preference_prompt()
                history = stm.get_history(SESSION_ID)

                reasoning_steps = []
                search_results = []
                final_answer = None
                step = 0

                messages = [{"role": "system", "content": sys_prompt}]
                if pref_prompt:
                    messages.append({"role": "system", "content": pref_prompt})
                for msg in history:
                    if msg.get("content", "").strip():
                        messages.append(msg)
                messages.append({"role": "user", "content": f"用户问题：{enhanced}"})

                try:
                    while step < MAX_STEPS:
                        step += 1

                        if debug:
                            print(f"\n--- 第 {step} 轮 ---")
                            for i, msg in enumerate(messages):
                                preview = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
                                print(f"  [{i}] {msg['role']}: {preview}")

                        try:
                            output = chat_stream(messages, max_tokens=2048, temperature=0.3, debug=True)
                        except Exception as e:
                            print(f"\nLLM 调用失败: {e}")
                            final_answer = f"抱歉，调用模型时出错: {e}"
                            break

                        if debug:
                            print(f"[完整输出]:\n{output}")

                        action_type, payload = parse_react_output(output)

                        if action_type == "final":
                            final_answer = payload
                            if "[CONTINUE]" in output:
                                cont_match = re.search(r'\[CONTINUE\](.*?)\[FINAL\]', output, re.DOTALL)
                                if cont_match:
                                    reasoning_steps.append(cont_match.group(1).strip())
                            messages.append({"role": "assistant", "content": output})
                            if debug:
                                print(f"✅ 推理完成（{step} 步）")
                            break

                        elif action_type == "action":
                            action_name, action_input = payload
                            if not action_input:
                                action_input = enhanced

                            print(f"  🔍 检索: {action_input}")
                            messages.append({"role": "assistant", "content": output})
                            try:
                                search_result = search_reports(action_input, client=client)
                                print(f"  📋 检索结果长度: {len(search_result)} 字符")
                                print(f"  📋 完整检索结果:\n{search_result}")
                                print("  ---")
                            except Exception as e:
                                search_result = f"检索失败: {e}"
                                print(f"  ❌ {search_result}")

                            search_results.append((action_input, search_result))
                            reasoning_steps.append(f"[检索] {action_input}\n→ 返回 {len(search_result)} 字符")
                            messages.append({"role": "user", "content": f"观察（检索结果）：{search_result}\n请判断下一步。"})

                        else:
                            reasoning_text = payload
                            print(f"  💭 推理: {reasoning_text[:50]}{'...' if len(reasoning_text) > 50 else ''}")
                            reasoning_steps.append(reasoning_text)
                            messages.append({"role": "assistant", "content": reasoning_text})

                    if final_answer is None:
                        if debug:
                            print("⚠️ 达到最大步数，基于已有推理和检索结果生成最终回答")
                        messages.append({"role": "user", "content": "请基于以上推理和检索结果，输出最终回答。只输出 [FINAL] 和你的回答。\n\n[FINAL]"})
                        try:
                            force_output = chat_stream(messages, max_tokens=2048, temperature=0.3, debug=True)
                            force_output = force_output.strip()
                            if "[FINAL]" in force_output:
                                idx = force_output.find("[FINAL]")
                                final_answer = force_output[idx + len("[FINAL]"):].strip()
                            else:
                                final_answer = force_output
                        except Exception as e:
                            final_answer = f"抱歉，调用模型时出错: {e}"

                    # ── 收集本轮检索结果到累积池 ──
                    for action_input, search_result in search_results:
                        accumulated_searches.append({
                            "query": action_input,
                            "result": search_result,
                        })

                    # ── 生成合并报告 ──
                    if accumulated_searches:
                        try:
                            final_answer = generate_merged_report(accumulated_searches, base_sys_prompt)
                            print(f"\n✅ 回答: \n{final_answer}")
                        except Exception as e:
                            final_answer = f"抱歉，生成合并报告时出错: {e}"
                            print(f"\nAI: {final_answer}")
                    elif final_answer:
                        print(f"\n✅ 回答: \n{final_answer}")
                    else:
                        final_answer = "未检索到相关信息，无法生成报告。"
                        print(f"\nAI: {final_answer}")

                finally:
                    if final_answer:
                        stm.add_turn(SESSION_ID, query, final_answer)
                        ltm.sync_from_short_term(stm, SESSION_ID)

                print()

            except KeyboardInterrupt:
                print("\n\n⚠️ 推理被中断")
                break

    finally:
        ltm.on_session_end(stm, SESSION_ID)
        ltm.close()
        client.close()


if __name__ == "__main__":
    main()