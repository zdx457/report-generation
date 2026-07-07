"""
rag_chat_v2.py - 三段式工作流版本
Stage 1: 意图识别（LLM 分类器） → SEARCH / EDIT / CHAT
Stage 2: 意图分叉（硬编码路由）
Stage 3: 执行器（结构化提取 / 精准编辑 / 直接回复）
"""

import json
import os
import re
import sys
import time
import uuid
import shutil
import asyncio
import queue
from functools import wraps
from typing import Optional, Callable, Any

import yaml
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
from config import (
    get_embed_base_url, get_embed_model, get_embed_api_key,
    get_llm_base_url, get_llm_model, get_llm_api_key,
    get_llm_max_tokens, get_llm_temperature,
    get_db_path, get_collection_name,
    get_rag_top_k, get_rerank_top_k,
    get_rerank_api_key,
    reload_config,
)

from data_pipeline.build_vector_db import build_db
from data_pipeline.extract_metadata import extract_metadata
from data_pipeline.xlsx_slicer import process_file

# ── Web 模式依赖（可选） ──
try:
    from fastapi import FastAPI, Request, UploadFile, File
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# =============================================================================
# 配置
# =============================================================================
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

EMBED_URL = get_embed_base_url()
EMBED_MODEL = get_embed_model()
EMBED_API_KEY = get_embed_api_key()
CHAT_URL = get_llm_base_url()
CHAT_MODEL = get_llm_model()
CHAT_API_KEY = get_llm_api_key()
DB_PATH = get_db_path()
COLLECTION_NAME = get_collection_name()
PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag", "prompt.md")

RAG_TOP_K = get_rag_top_k()
RERANK_TOP_K = get_rerank_top_k()

# =============================================================================
# 工具函数
# =============================================================================
def retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"  ⚠️ {e}，{delay}s 后重试 ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def load_system_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一个医疗影像报告分析助手。"


def get_embedding(text):
    payload = {"model": EMBED_MODEL, "input": text}
    headers = {"Content-Type": "application/json"}
    if EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
    r = requests.post(EMBED_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def _estimate_tokens(messages):
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    return total_chars, total_chars // 2


@retry()
def chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=False, caller="chat_stream"):
    total_chars, est_tokens = _estimate_tokens(messages)
    if debug:
        print(f"📤 [{caller}] 发送请求: {len(messages)} messages, {total_chars} chars, 估算 ~{est_tokens} tokens")

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if CHAT_API_KEY:
        headers["Authorization"] = f"Bearer {CHAT_API_KEY}"

    try:
        r = requests.post(CHAT_URL, headers=headers, json=payload, timeout=120, stream=True)
    except requests.RequestException as e:
        print(f"❌ [{caller}] 请求失败: {e}")
        raise

    if not r.ok:
        try:
            error_body = r.text[:500]
        except Exception:
            error_body = "(无法读取响应体)"
        print(f"❌ [{caller}] HTTP {r.status_code}: {error_body}")
        r.raise_for_status()

    full_text = ""
    token_count = 0
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
                    token_count += 1
                    if _emit:
                        _emit("token", {"content": content})
                    full_text += content
            except json.JSONDecodeError:
                continue

    if debug:
        print(f"📥 [{caller}] 完成: 收到 {token_count} tokens, {len(full_text)} chars")
    return full_text.strip()

@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)


def search_reports(query, top_k=RAG_TOP_K, rerank_top_k=RERANK_TOP_K, client=None, _emit=None):
    """RAG 检索：多路召回 + Rerank，返回格式化文本"""
    if rerank_top_k > top_k:
        rerank_top_k = top_k

    query_vec = get_embedding(query)
    keywords = parse_query_keywords(query)

    candidates, recall_details = multi_recall(query_vec, keywords, top_k=top_k, client=client, return_details=True)

    if not candidates:
        return "未检索到相关报告。"

    if _emit:
        vec_results = recall_details.get("vector", [])
        meta_results = recall_details.get("metadata", [])
        kw_results = recall_details.get("keyword", [])
        total_before = len(vec_results) + len(meta_results) + len(kw_results)
        _emit("recall", {
            "vector_count": len(vec_results),
            "metadata_count": len(meta_results),
            "keyword_count": len(kw_results),
            "total_before": total_before,
            "total_after": len(candidates),
            "dedup": total_before - len(candidates),
        })

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
                    "index": i,
                    "score": e.get("_rerank_score", 0),
                    "source": e.get("source", ""),
                    "diagnosis": e.get("诊断结论", ""),
                    "text": e.get("text", ""),
                }
                for i, e in enumerate(reranked_entities)
            ]
        })

    parts = []
    for i, entity in enumerate(reranked_entities, 1):
        score = entity.get("_rerank_score", 0)
        parts.append(f"### 参考{i}（Rerank分数: {score:.4f}）\n{entity['text']}\n")

    return "\n".join(parts)


# =============================================================================
# Stage 1: 意图识别 Prompt
# =============================================================================
INTENT_PROMPT = """你是一个医疗影像报告助手的意图分类器。根据用户输入，判断用户意图。

## 分类规则

- **SEARCH**：用户要生成、查询、检索影像报告。例如：
  - "CT 脑出血"
  - "生成一份胸部CT报告"
  - "头颅MRI平扫"
  - "脑梗"

- **EDIT**：用户要修改之前已经生成的内容。例如：
  - "修改CT值为80"
  - "把诊断改成脑梗死"
  - "调整一下格式"
  - "换成左侧基底节区"

- **CHAT**：闲聊、问候、与报告生成无关的问题。例如：
  - "你好"
  - "今天天气怎么样"
  - "你是什么模型"

## 输出格式

只输出一个单词，不要任何其他内容：
SEARCH
或
EDIT
或
CHAT"""


# =============================================================================
# Stage 3A: 结构化提取 Prompt
# =============================================================================
STRUCTURE_PROMPT = """你是一个医疗影像报告结构化提取器。根据检索到的参考报告，提取关键信息并生成结构化报告。

## 核心规则

1. **忠实原文**：直接使用参考报告中的原文描述，不修改数值，不脑补缺失信息
2. **选最高分**：当多条参考的 Rerank 分数接近时，优先使用分数最高的参考
3. **不补全**：参考中缺少的字段（如左右侧），保留原文表述，不自行补全
4. **格式规范**：影像学表现中每个句号结尾的句子独占一段
5. **不重复已有病变**：如果上一轮报告中已包含某病变，请勿重复生成，只提取本次检索结果中新增的病变

## 输出格式

严格输出 JSON，不要输出任何其他内容：

```json
{
  "reasoning": "你的推理过程：选择了哪条参考、为什么选择、如何处理",
  "影像学表现": {
    "病变名称": "描述内容（每句一段，用换行符\\n分隔）"
  },
  "诊断意见": {
    "病变名称": "诊断意见"
  }
}
```

## 示例

输入参考：
### 参考1（Rerank分数: 0.6562）
CT平扫显示右侧基底节区团块状高密度灶，CT值65HU，范围约3.0×1.6cm。中线结构轻度向对侧移位。脑干及小脑未见异常。

输出：
```json
{
  "reasoning": "参考1的Rerank分数最高(0.6562)，描述完整，包含CT值、尺寸、继发改变，选择参考1作为基准。",
  "影像学表现": {
    "右侧基底节区脑出血": "CT平扫显示右侧基底节区团块状高密度灶，CT值65HU，范围约3.0×1.6cm。\\n中线结构轻度向对侧移位。\\n脑干及小脑未见异常。"
  },
  "诊断意见": {
    "右侧基底节区脑出血": "右侧基底节区脑出血"
  }
}
```"""


# =============================================================================
# Stage 3B: 精准编辑 Prompt
# =============================================================================
EDIT_PROMPT = """你是一个医疗影像报告编辑器。根据用户指令，修改已有报告。

## 规则

1. **只改指定内容**：只修改用户明确要求修改的部分，其余内容保持不变
2. **Key 不变**：影像学表现和诊断意见的病变名称（Key）不能增删
3. **不检索**：不需要检索知识库，直接基于已有报告修改
4. **不添加新病变**：不要添加用户没有要求的病变

## 输出格式

严格输出 JSON，不要输出任何其他内容：

```json
{
  "影像学表现": {
    "病变名称": "修改后的描述"
  },
  "诊断意见": {
    "病变名称": "修改后的诊断意见"
  }
}
```"""


# =============================================================================
# Stage 1: 意图识别
# =============================================================================
def classify_intent(query, history, _emit=None):
    """调用 LLM 进行意图分类，返回 SEARCH / EDIT / CHAT"""
    messages = [{"role": "system", "content": INTENT_PROMPT}]

    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)

    messages.append({"role": "user", "content": query})

    if _emit:
        _emit("status", {"message": "正在识别意图..."})

    output = chat_stream(messages, max_tokens=32, temperature=0.0, _emit=None, debug=True, caller="classify_intent")
    intent = output.strip().upper()

    if intent not in ("SEARCH", "EDIT", "CHAT"):
        print(f"  ⚠️ 意图识别结果异常: {intent}，默认当作 SEARCH")
        intent = "SEARCH"

    if _emit:
        _emit("intent", {"intent": intent})

    return intent


# =============================================================================
# Stage 3A: 结构化提取
# =============================================================================
def structure_report(search_result, history, last_report, _emit=None):
    """将检索结果 + 上一轮报告(如有) 结构化输出为 JSON"""
    sys_prompt = STRUCTURE_PROMPT

    if last_report and last_report[0]:
        try:
            last_obj = json.loads(last_report[0])
            last_obj.pop("reasoning", None)
            last_text = json.dumps(last_obj, ensure_ascii=False)
        except Exception:
            last_text = last_report[0]
        sys_prompt += f"\n\n## 已生成的报告（仅参考，请勿重复其中的病变）\n{last_text}"

    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history[-4:]:
        content = msg.get("content", "").strip()
        if not content:
            continue
        role = msg.get("role", "")
        if role == "assistant" and len(content) > 200:
            content = "已生成报告（内容较长，此处省略完整文本）"
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": f"检索结果：\n{search_result}\n\n请按 JSON 格式输出结构化报告。"})

    if _emit:
        _emit("status", {"message": "正在生成结构化报告..."})

    output = chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=True, caller="structure_report")

    report_json = _extract_json(output)

    if _emit:
        reasoning = report_json.get("reasoning", "") if isinstance(report_json, dict) else ""
        if reasoning:
            _emit("reasoning", {"text": reasoning})

    return report_json

# =============================================================================
# Stage 3B: 精准编辑
# =============================================================================
def edit_report(query, last_report, history, _emit=None):
    """根据用户指令修改已有报告"""
    if not last_report or not last_report[0]:
        return {"error": "没有可修改的报告，请先生成一份报告。"}

    old_json_str = last_report[0]

    try:
        old_json = json.loads(old_json_str)
    except json.JSONDecodeError:
        old_json = {"raw": old_json_str}

    sys_prompt = EDIT_PROMPT

    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)

    messages.append({"role": "user", "content": f"当前报告：\n{old_json_str}\n\n修改指令：{query}\n\n请按 JSON 格式输出修改后的完整报告。"})

    if _emit:
        _emit("status", {"message": "正在修改报告..."})

    output = chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=True, caller="edit_report")

    new_json = _extract_json(output)

    if isinstance(new_json, dict) and isinstance(old_json, dict):
        if "影像学表现" in new_json and "影像学表现" in old_json:
            old_keys = set(old_json["影像学表现"].keys())
            new_keys = set(new_json["影像学表现"].keys())
            if old_keys != new_keys:
                print(f"  ⚠️ Key 集合变化：旧 {old_keys} → 新 {new_keys}，使用旧报告兜底")
                return old_json

    return new_json if isinstance(new_json, dict) and "error" not in new_json else old_json


# =============================================================================
# Stage 3C: 闲聊回复
# =============================================================================
def chat_reply(query, history, _emit=None):
    """直接回复闲聊"""
    messages = [
        {"role": "system", "content": "你是一个医疗影像报告助手。请友好、简洁地回答用户问题。"},
    ]
    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)
    messages.append({"role": "user", "content": query})

    if _emit:
        _emit("status", {"message": "正在回复..."})

    return chat_stream(messages, max_tokens=512, temperature=0.7, _emit=_emit, debug=True, caller="chat_reply")


# =============================================================================
# JSON 提取工具
# =============================================================================
def _extract_json(text):
    """从 LLM 输出中提取 JSON"""
    text = text.strip()

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    print(f"  ⚠️ 无法解析 JSON: {text[:200]}...")
    return {"error": "JSON 解析失败", "raw": text[:500]}


# =============================================================================
# JSON 报告 → Markdown 展示文本
# =============================================================================
def json_to_display(report_json):
    """将 JSON 报告转换为 Markdown 展示文本"""
    if isinstance(report_json, str):
        return report_json

    if not isinstance(report_json, dict) or "error" in report_json:
        return report_json.get("raw", str(report_json))

    lines = []

    imaging = report_json.get("影像学表现", {})
    if imaging:
        lines.append("## 一、影像学表现")
        lines.append("")
        for name, desc in imaging.items():
            lines.append(f"{desc}")
            lines.append("")

    diagnosis = report_json.get("诊断意见", {})
    if diagnosis:
        lines.append("## 二、诊断意见")
        lines.append("")
        for name, opinion in diagnosis.items():
            lines.append(f"{opinion}")
            lines.append("")

    return "\n".join(lines).strip()


# =============================================================================
# 主流程：run_pipeline
# =============================================================================
def run_pipeline(query, session_id, stm, ltm, client, last_report, _emit):
    """三段式工作流主流程"""
    # ── 预处理 ──
    if is_too_vague(query):
        clarification = get_clarification(query)
        _emit("message", {"content": clarification})
        return clarification

    enhanced = stm.resolve_context(session_id, query)
    enhanced = standardize_query(enhanced)
    if enhanced != query:
        _emit("context_resolve", {"original": query, "resolved": enhanced})

    if needs_rewrite(enhanced):
        original = enhanced
        rewritten = rewrite_query(enhanced)
        if rewritten and rewritten != enhanced:
            enhanced = rewritten
            _emit("query_rewrite", {"original": original, "rewritten": rewritten})

    history = stm.get_history(session_id)


    # ── Stage 1: 意图识别 ──
    intent = classify_intent(enhanced, history, _emit=_emit)
    print(f"  🎯 意图: {intent}")

    # ── Stage 2: 意图分叉 ──
    if intent == "CHAT":
        reply = chat_reply(enhanced, history, _emit=_emit)
        _emit("message", {"content": reply})
        if last_report:
            last_report[0] = last_report[0] or ""
        stm.add_turn(session_id, query, reply)
        return reply

    elif intent == "EDIT":
        result_json = edit_report(enhanced, last_report, history, _emit=_emit)
        if isinstance(result_json, dict) and "error" in result_json:
            _emit("error", {"message": result_json["error"]})
            return result_json["error"]

        new_json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
        last_report[0] = new_json_str
        display_text = json_to_display(result_json)
        _emit("report", {"content": display_text})
        stm.add_turn(session_id, query, display_text)
        return display_text

    elif intent == "SEARCH":
        _emit("status", {"message": f"开始检索：{enhanced}"})
        search_result = search_reports(enhanced, client=client, _emit=_emit)
        _emit("status", {"message": "检索完成，开始结构化提取..."})
        result_json = structure_report(search_result, history, last_report, _emit=_emit)
        if isinstance(result_json, dict) and "error" in result_json:
            _emit("error", {"message": result_json.get("raw", str(result_json))})
            return result_json.get("raw", str(result_json))

        # 合并新旧报告：已有病变保留，新增病变追加
        if last_report and last_report[0]:
            try:
                old_json = json.loads(last_report[0])
                merged = {}
                for section in ("影像学表现", "诊断意见"):
                    merged[section] = {}
                    if section in old_json and isinstance(old_json[section], dict):
                        merged[section].update(old_json[section])
                    if section in result_json and isinstance(result_json[section], dict):
                        for k, v in result_json[section].items():
                            if k not in merged[section]:
                                merged[section][k] = v
                new_reasoning = result_json.get("reasoning", "")
                old_reasoning = old_json.get("reasoning", "")
                if new_reasoning:
                    merged["reasoning"] = f"[第1轮] {old_reasoning}\n[本轮] {new_reasoning}" if old_reasoning else new_reasoning
                result_json = merged
            except Exception:
                pass  # 合并失败则用新报告

        new_json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
        last_report[0] = new_json_str
        display_text = json_to_display(result_json)
        _emit("report", {"content": display_text})
        stm.add_turn(session_id, query, display_text)
        return display_text

    else:
        _emit("error", {"message": f"未知意图: {intent}"})
        return f"未知意图: {intent}"


# =============================================================================
# Web 服务
# =============================================================================
_web_sessions = {}

def web_main(port=8000):
    if not WEB_AVAILABLE:
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    app = FastAPI(title="影像报告生成Agent v2")

    front_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "front")
    if os.path.isdir(front_dir):
        app.mount("/static", StaticFiles(directory=front_dir), name="static")

    @app.get("/")
    async def index():
        index_path = os.path.join(front_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "影像报告生成Agent v2", "docs": "/docs"}

    def _get_or_create_session(session_id):
        if session_id not in _web_sessions:
            stm = ShortTermMemory()
            ltm = LongTermMemory()
            client = MilvusClient(DB_PATH)
            client.load_collection(COLLECTION_NAME)
            _web_sessions[session_id] = {
                "stm": stm,
                "ltm": ltm,
                "client": client,
                "last_report": [""],
            }
        return _web_sessions[session_id]

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        query = body.get("query", "").strip()
        session_id = body.get("session_id", "default")

        if not query:
            return {"error": "query 不能为空"}

        session = _get_or_create_session(session_id)

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_sse(event_type, data):
                try:
                    event_queue.put_nowait(json.dumps({"type": event_type, **data}, ensure_ascii=False))
                except Exception:
                    pass

            def run():
                try:
                    run_pipeline(
                        query, session_id,
                        session["stm"], session["ltm"], session["client"],
                        session["last_report"],
                        _emit_sse,
                    )
                except Exception as e:
                    _emit_sse("error", {"message": str(e)})
                finally:
                    event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 120)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/info")
    async def info(session_id: str = "default"):
        session = _get_or_create_session(session_id)
        session_info = session["stm"].session_info(session_id)
        return {
            "current_turns": session_info.get("current_turns", 0),
            "entity_count": session_info.get("entity_count", 0),
            "has_last_report": bool(session["last_report"][0]),
        }

    @app.get("/api/memory")
    async def memory(session_id: str = "default"):
        session = _get_or_create_session(session_id)
        stm = session["stm"]
        info = stm.session_info(session_id)
        history = stm.get_history(session_id)
        entities = stm.get_entities(session_id)
        summaries = stm.get_summaries(session_id)

        turns = []
        for i in range(0, len(history), 2):
            user_msg = history[i]["content"] if i < len(history) else ""
            assistant_msg = history[i + 1]["content"] if i + 1 < len(history) else ""
            turns.append({
                "round": i // 2 + 1,
                "user": user_msg,
                "assistant": assistant_msg,
            })

        return {
            "turns": turns,
            "entities": entities,
            "summaries": summaries,
            "current_turns": info.get("current_turns", 0),
            "total_turns": info.get("total_turns", 0),
            "max_rounds": info.get("max_rounds", 5),
        }

    @app.get("/api/kb/status")
    async def kb_status():
        total = 0
        try:
            if os.path.exists(DB_PATH):
                client = MilvusClient(DB_PATH)
                total = len(client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"]))
                client.close()
        except Exception:
            pass
        slices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "data_pipeline", "xlsx_slices")
        md_count = len([f for f in os.listdir(slices_dir) if f.endswith(".md")]) if os.path.isdir(slices_dir) else 0
        metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template", "metadata.json")
        meta_exists = os.path.exists(metadata_path)
        return {"total": total, "md_count": md_count, "db_path": DB_PATH, "metadata_exists": meta_exists}

    @app.get("/api/kb/files")
    async def kb_files():
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        slices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "xlsx_slices")
        files = []
        if os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                if fname.endswith(".xlsx") and not fname.startswith("~$"):
                    fpath = os.path.join(report_dir, fname)
                    stat = os.stat(fpath)
                    basename = os.path.splitext(fname)[0]
                    slice_count = 0
                    if os.path.isdir(slices_dir):
                        slice_count = len([f for f in os.listdir(slices_dir) if f.startswith(basename) and f.endswith(".md")])
                    files.append({
                        "name": fname,
                        "slice_count": slice_count,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
        return {"files": files}

    @app.post("/api/kb/build")
    async def kb_build(request: Request):
        body = await request.json()
        rebuild = body.get("rebuild", False)
        batch_size = body.get("batch_size", 16)
        slices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "data_pipeline", "xlsx_slices")

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                build_db(slices_dir, batch_size=batch_size, rebuild=rebuild, progress_callback=_emit_log)
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 600)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/kb/extract-metadata")
    async def kb_extract_metadata():
        metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        metadata_path = os.path.join(metadata_dir, "metadata.json")

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                extract_metadata(metadata_dir, metadata_path, progress_callback=_emit_log)
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 120)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/kb/upload")
    async def kb_upload(file: UploadFile = File(...)):
        if not file.filename.endswith(".xlsx"):
            return {"error": "只支持 .xlsx 文件"}

        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        slices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "xlsx_slices")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(slices_dir, exist_ok=True)

        filepath = os.path.join(report_dir, file.filename)

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                _emit_log({"level": "info", "msg": f"上传文件: {file.filename}"})
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                _emit_log({"level": "info", "msg": "切片中..."})
                count = process_file(filepath, slices_dir, progress_callback=_emit_log)
                _emit_log({"level": "done", "msg": f"✅ 切片完成，共生成 {count} 个 md 文件"})
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 120)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.delete("/api/session")
    async def clear_session(session_id: str = "default"):
        if session_id in _web_sessions:
            session = _web_sessions[session_id]
            session["stm"].clear(session_id)
            session["last_report"] = [""]
        return {"status": "ok"}

    @app.post("/api/clear")
    async def clear_session_post(request: Request):
        body = await request.json()
        session_id = body.get("session_id", "default")
        if session_id in _web_sessions:
            session = _web_sessions[session_id]
            session["stm"].clear(session_id)
            session["last_report"] = [""]
        return {"status": "ok"}

    @app.get("/api/config")
    async def get_config():
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yml")
        if not os.path.exists(config_path):
            return {"error": "配置文件不存在"}
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # 如果 config.yml 中 api_key 为空但 .env 中有对应的 key，
        # 则返回掩码占位符，让前端知道已配置了 key
        for model_list_key, env_key_func in [
            ("llms", get_llm_api_key),
            ("embeddings", get_embed_api_key),
            ("reranks", get_rerank_api_key),
        ]:
            models = config_data.get(model_list_key)
            if isinstance(models, list):
                for m in models:
                    if not m.get("api_key") and env_key_func():
                        m["api_key"] = "••••••••••••••••••••••••"

        return {"config": config_data, "path": config_path}

    @app.post("/api/config")
    async def save_config(request: Request):
        body = await request.json()
        config_data = body.get("config")
        if config_data is None:
            return {"error": "缺少 config 参数"}

        # 如果 api_key 是前端掩码占位符，说明实际 key 在 .env 中，清空避免写入明文
        for model_list_key in ["llms", "embeddings", "reranks"]:
            models = config_data.get(model_list_key)
            if isinstance(models, list):
                for m in models:
                    if m.get("api_key") == "••••••••••••••••••••••••":
                        m["api_key"] = ""

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        reload_config()
        return {"status": "ok", "message": "配置已保存并生效"}

    @app.post("/api/test-model")
    async def test_model_connection(request: Request):
        body = await request.json()
        model_config = body.get("model_config", {})
        model_type = body.get("model_type", "llms")

        base_url = model_config.get("base_url", "")
        model_name = model_config.get("model", "")
        api_key = model_config.get("api_key", "")

        # 如果前端没传 key（空或掩码占位符），尝试从环境变量 / .env 兜底
        if not api_key or api_key == "••••••••••••••••••••••••":
            if model_type == "embeddings":
                api_key = get_embed_api_key()
            elif model_type == "reranks":
                api_key = get_rerank_api_key()
            else:
                api_key = get_llm_api_key()

        if not base_url:
            return {"success": False, "message": "API 地址不能为空"}
        if not model_name:
            return {"success": False, "message": "模型名不能为空"}

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if model_type == "embeddings":
            payload = {
                "model": model_name,
                "input": "你好",
            }
        elif model_type == "reranks":
            payload = {
                "model": model_name,
                "query": "测试查询",
                "documents": ["测试文档"],
            }
        else:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 16,
                "temperature": 0.0,
                "stream": False,
            }

        try:
            r = requests.post(base_url, headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if model_type == "embeddings":
                    emb_data = data.get("data", [])
                    if emb_data:
                        dim = len(emb_data[0].get("embedding", []))
                        return {"success": True, "message": f"连接成功，向量维度: {dim}"}
                    return {"success": True, "message": "连接成功"}
                elif model_type == "reranks":
                    results = data.get("results", [])
                    if results:
                        score = results[0].get("relevance_score", "N/A")
                        return {"success": True, "message": f"连接成功，相关性分数: {score}"}
                    return {"success": True, "message": "连接成功"}
                else:
                    choices = data.get("choices", [])
                    if choices:
                        reply = choices[0].get("message", {}).get("content", "")
                        return {"success": True, "message": f"连接成功，返回: {reply[:50]}"}
                    return {"success": True, "message": "连接成功"}
            else:
                error_msg = r.text[:200] if r.text else f"HTTP {r.status_code}"
                return {"success": False, "message": error_msg}
        except requests.exceptions.Timeout:
            return {"success": False, "message": "连接超时"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "message": "无法连接到服务器"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    print(f"\n{'='*60}")
    print(f"  影像报告生成Agent v2 Web 服务")
    print(f"  {'='*60}")
    print(f"  地址: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  生成模型: {CHAT_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}")
    print(f"  {'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)


# =============================================================================
# CLI 模式
# =============================================================================
def main():
    if "--web" in sys.argv:
        web_main()
        return

    print(f"\n{'='*60}")
    print(f"  影像报告生成Agent v2 CLI")
    print(f"  {'='*60}")
    print(f"  输入问题开始对话，输入 clear 清空会话，输入 quit 退出")
    print(f"  {'='*60}\n")

    SESSION_ID = f"rag_v2_{uuid.uuid4().hex[:8]}"
    stm = ShortTermMemory()
    ltm = LongTermMemory()
    client = MilvusClient(DB_PATH)
    client.load_collection(COLLECTION_NAME)
    last_report = [""]

    def _emit(event_type, data):
        if event_type == "report":
            print(f"\n📋 报告:\n{data['content']}\n")
        elif event_type == "reasoning":
            print(f"💭 推理: {data['content']}")
        elif event_type == "intent":
            print(f"🎯 意图: {data['intent']}")
        elif event_type == "error":
            print(f"❌ 错误: {data['message']}")

    try:
        while True:
            user_input = input("\n👤 用户: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("👋 再见！")
                break

            if user_input.lower() == "clear":
                stm.clear(SESSION_ID)
                last_report[0] = ""
                print("🧹 会话已清空\n")
                continue

            run_pipeline(
                user_input, SESSION_ID,
                stm, ltm, client,
                last_report,
                _emit,
            )
    except KeyboardInterrupt:
        print("\n👋 再见！")


if __name__ == "__main__":
    main()