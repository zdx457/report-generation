"""ReAct 多轮推理对话终端（带 RAG 检索工具）

用法：
  python rag_chat.py
  python rag_chat.py --debug  # 显示调试信息
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
from query_rewrite import parse_query_keywords

SESSION_ID = f"rag_react_{uuid.uuid4().hex[:8]}"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
load_dotenv(ENV_PATH)

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36-27b")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "milvus_lite.db")
COLLECTION_NAME = "report_slices"
PROMPT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompt.md")

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


def search_reports(query, top_k=RAG_TOP_K, rerank_top_k=RERANK_TOP_K, client=None):
    """RAG 检索：向量检索 + 多路召回 + Rerank，返回格式化文本"""
    if rerank_top_k > top_k:
        rerank_top_k = top_k

    query_vec = get_embedding(query)
    keywords = parse_query_keywords(query)

    candidates = multi_recall(query_vec, keywords, top_k=top_k, client=client)

    if not candidates:
        return "未检索到相关报告。"

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

    return "\n\n".join(contexts)


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

## 重要规则

- 第一轮不要直接输出 [FINAL]，至少先检索或推理一步
- 检索时使用简洁的关键词，不要用完整句子
- [FINAL] 回答要基于检索结果，注明信息来源（如"参考1显示..."）
- 如果检索结果不足以回答问题，如实说明"""


@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)


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

                query = user_input
                enhanced = stm.resolve_context(SESSION_ID, query)
                if debug and enhanced != query:
                    print(f"🔗 上下文消解: '{query}' → '{enhanced}'")

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
                            if "[CONTINUE]" in output:
                                cont_match = re.search(r'\[CONTINUE\](.*?)\[FINAL\]', output, re.DOTALL)
                                if cont_match:
                                    reasoning_steps.append(cont_match.group(1).strip())
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
                            print(f"\n✅ 回答: {final_answer}")
                        except Exception as e:
                            final_answer = f"抱歉，调用模型时出错: {e}"
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