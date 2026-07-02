"""ReAct 多轮推理对话终端（带 RAG 检索工具）- V2

改进：
- 先判断是否需要 RAG 检索，再决定走哪条路径
- 修改已有报告时不经过 RAG 流程（跳过模糊检测/标准化/改写/检索）
- 生成新报告时走完整 RAG 管道
- 路由决策由 skills/ 目录下的 markdown 文件定义，LLM 读取后决定走哪条路

用法：
  python rag_chat_v2.py
  python rag_chat_v2.py --debug  # 显示调试信息
"""

import json
import os
import re
import sys
import time
import uuid
from functools import wraps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv
from pymilvus import MilvusClient

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from rerank import get_rerank_config, rerank_documents
from retrieval import multi_recall
from query_rewrite import (
    parse_query_keywords,
    is_too_vague,
    get_clarification,
    standardize_query,
    needs_rewrite,
    rewrite_query,
)

# ── 配置 ──
SESSION_ID = f"rag_react_{uuid.uuid4().hex[:8]}"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(os.path.dirname(BASE_DIR), ".env")
load_dotenv(ENV_PATH)

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36-27b")
DB_PATH = os.path.join(BASE_DIR, "milvus_lite.db")
COLLECTION_NAME = "report_slices"
PROMPT_FILE = os.path.join(BASE_DIR, "prompt.md")
SKILLS_DIR = os.path.join(BASE_DIR, "skills")

MAX_STEPS = 5
RAG_TOP_K = 5
RERANK_TOP_K = 3
MAX_CONTEXT_STEPS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具函数
# ═══════════════════════════════════════════════════════════════════════════════

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


@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)


def search_reports(query, top_k=RAG_TOP_K, rerank_top_k=RERANK_TOP_K, client=None):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 系统：读取 skills/ 目录下的 markdown 文件
# ═══════════════════════════════════════════════════════════════════════════════

def read_skill_md(path: str) -> str:
    """读取 skill 目录下的 markdown 文件内容。"""
    full_path = os.path.join(SKILLS_DIR, path)
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def extract_section(md_text: str, section_name: str) -> str:
    """从 markdown 中提取指定 section 的内容（## 标题之后的内容）。"""
    pattern = rf'##\s+{re.escape(section_name)}\s*\n(.*?)(?=\n##\s|\Z)'
    match = re.search(pattern, md_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def classify_intent(user_input: str) -> str:
    """LLM 读取 skills/router.md 来决定走哪个 skill。

    Returns:
        "modify" 或 "generate"
    """
    router_prompt = read_skill_md("router.md")
    if not router_prompt:
        return "generate"

    messages = [
        {"role": "user", "content": router_prompt.format(user_input=user_input)},
    ]
    try:
        result = chat_stream(messages, max_tokens=10, temperature=0.0, debug=False)
        result = result.strip().lower()
        if "modify" in result:
            return "modify"
        return "generate"
    except Exception:
        return "generate"


def get_last_report(stm, session_id: str) -> str:
    """从短期记忆中获取最近一次生成的报告内容。"""
    history = stm.get_history(session_id)
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "[FINAL]" in content:
                return content.split("[FINAL]", 1)[1].strip()
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 执行函数
# ═══════════════════════════════════════════════════════════════════════════════

def execute_modify_skill(existing_report: str, modification_request: str, history: list[dict]) -> str:
    """执行"修改报告" skill：读取 report/modify.md 作为 system prompt，直接修改报告。"""
    skill_md = read_skill_md("report/modify.md")
    system_prompt = extract_section(skill_md, "System Prompt")
    if not system_prompt:
        system_prompt = "你是一个医疗影像报告修改助手。请根据用户要求修改报告。"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)
    messages.append({
        "role": "user",
        "content": f"当前报告内容：\n\n{existing_report}\n\n修改要求：{modification_request}\n\n请输出修改后的完整报告。",
    })
    try:
        result = chat_stream(messages, max_tokens=2048, temperature=0.3, debug=True)
        return result
    except Exception as e:
        return f"修改失败: {e}"


def execute_generate_skill(query: str, stm, ltm, client, session_id: str, debug: bool):
    """执行"生成报告" skill：读取 report/generate.md 作为 ReAct system prompt，走完整 RAG 流程。"""
    skill_md = read_skill_md("report/generate.md")
    react_prompt = extract_section(skill_md, "System Prompt")

    base_sys_prompt = load_system_prompt()
    sys_prompt = base_sys_prompt + "\n\n" + (react_prompt or "")
    pref_prompt = ltm.get_preference_prompt()
    history = stm.get_history(session_id)

    final_answer = None
    step = 0

    messages = [{"role": "system", "content": sys_prompt}]
    if pref_prompt:
        messages.append({"role": "system", "content": pref_prompt})
    for msg in history:
        if msg.get("content", "").strip():
            messages.append(msg)
    messages.append({"role": "user", "content": f"用户问题：{query}"})

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
                messages.append({"role": "assistant", "content": output})
                print(f"\n✅ 回答: \n{final_answer}")
                if debug:
                    print(f"✅ 推理完成（{step} 步）")
                break

            elif action_type == "action":
                action_name, action_input = payload
                if not action_input:
                    action_input = query

                print(f"  🔍 检索: {action_input}")
                messages.append({"role": "assistant", "content": output})
                try:
                    search_result = search_reports(action_input, client=client)
                    if debug:
                        print(f"  📋 检索结果长度: {len(search_result)} 字符")
                        print(f"  📋 [DEBUG] 完整检索结果:\n  {search_result}")
                        print("  ---")
                except Exception as e:
                    search_result = f"检索失败: {e}"
                    print(f"  ❌ {search_result}")

                messages.append({"role": "user", "content": f"观察（检索结果）：{search_result}\n请判断下一步。"})

            else:
                reasoning_text = payload
                print(f"  💭 推理: {reasoning_text[:50]}{'...' if len(reasoning_text) > 50 else ''}")
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
                print(f"\n✅ 回答: {final_answer}")
            except Exception as e:
                final_answer = f"抱歉，调用模型时出错: {e}"
                print(f"\nAI: {final_answer}")

    finally:
        if final_answer:
            stm.add_turn(session_id, query, final_answer)
            ltm.sync_from_short_term(stm, session_id)

    return final_answer


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    debug = "--debug" in sys.argv

    if not os.path.exists(DB_PATH):
        print("向量数据库不存在，请先运行 build_vector_db.py", file=sys.stderr)
        sys.exit(1)

    stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
    ltm = LongTermMemory(user_id=SESSION_ID)

    client = MilvusClient(uri=DB_PATH)
    client.load_collection(COLLECTION_NAME)

    print("=" * 60)
    print("=== ReAct 多轮推理对话（带 RAG 检索） V2 ===")
    print("=" * 60)
    print(f"生成模型: {CHAT_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Rerank: {get_rerank_config()['rerank_model']}")
    print(f"检索返回: top-{RAG_TOP_K} → rerank top-{RERANK_TOP_K}")
    print(f"最大推理步数: {MAX_STEPS}")
    print(f"用户ID: {ltm.user_id}")
    print(f"Skills 目录: {SKILLS_DIR}")
    print()
    print("命令:")
    print("  exit/quit - 退出")
    print("  clear     - 清空会话")
    print("  info      - 查看记忆状态")
    print("  直接输入   - LLM 读取 skills/ 决定走哪条路")
    print()

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

                # ================================================================
                # LLM 读取 skills/router.md 决定走哪条路
                # ================================================================
                intent = classify_intent(user_input)

                if debug:
                    print(f"[DEBUG] LLM 路由决策: {intent}")

                # ── 路径 A: 修改报告 skill ──
                if intent == "modify":
                    last_report = get_last_report(stm, SESSION_ID)
                    if last_report:
                        print(f"📝 LLM 路由到 [修改报告] skill，跳过 RAG 流程...")
                        if debug:
                            print(f"[DEBUG] 当前报告（前200字）: {last_report[:200]}...")

                        history = stm.get_history(SESSION_ID)
                        final_answer = execute_modify_skill(last_report, user_input, history)

                        if final_answer:
                            stm.add_turn(SESSION_ID, user_input, final_answer)
                            ltm.sync_from_short_term(stm, SESSION_ID)

                        print()
                        continue
                    else:
                        print(f"⚠️ LLM 路由到 [修改报告]，但未找到之前的报告，自动转入 [生成报告] skill...")

                # ── 路径 B: 生成报告 skill ──
                print(f"🔍 LLM 路由到 [生成报告] skill，进入 RAG 流程...")

                query = user_input

                if is_too_vague(query):
                    clarification = get_clarification(query)
                    print(f"⚠️ 查询过于模糊: {clarification}")
                    print()
                    continue

                original_query = query
                query = standardize_query(query)
                if needs_rewrite(query):
                    rewritten = rewrite_query(query)
                    if rewritten != query:
                        query = rewritten
                        print(f"🔄 查询改写: '{original_query}' → '{query}'")
                elif query != original_query:
                    print(f"📝 查询标准化: '{original_query}' → '{query}'")

                enhanced = stm.resolve_context(SESSION_ID, query)
                if debug and enhanced != query:
                    print(f"🔗 上下文消解: '{query}' → '{enhanced}'")

                execute_generate_skill(query, stm, ltm, client, SESSION_ID, debug)

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