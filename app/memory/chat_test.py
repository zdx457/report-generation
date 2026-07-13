"""记忆模块测试终端。

纯多轮对话，测试 ShortTermMemory + LongTermMemory 的对话历史管理和上下文消解。
不涉及特定业务领域操作。

用法：
  python chat_test.py
  python chat_test.py --debug  # 显示上下文消解和记忆状态
"""
import json
import os
import sys
import uuid

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.entity_tracker import EntityTracker
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

SESSION_ID = f"test_{uuid.uuid4().hex[:8]}"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


def chat_stream(messages, max_tokens=1024, temperature=0.7):

    load_dotenv(ENV_PATH)
    CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}

    full_reply = ""
    with requests.post(CHAT_URL, headers=headers, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
            else:
                data = line
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                if isinstance(obj, dict) and "choices" in obj:
                    for c in obj["choices"]:
                        delta = c.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_reply += content
            except Exception:
                pass
    print()
    return full_reply


def summarize_fn(messages: list[dict]) -> str:

    load_dotenv(ENV_PATH)
    CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")

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


def main():
    debug = "--debug" in sys.argv
    stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
    entity_tracker = EntityTracker()
    ltm = LongTermMemory(user_id=SESSION_ID)

    load_dotenv(ENV_PATH)
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")

    base_system = "你是一个有帮助的AI助手。"

    print("=== 多轮对话（带记忆）已启动 ===")
    print(f"模型: {CHAT_MODEL}")
    print(f"短期记忆: 最近 {stm.max_rounds} 轮")
    print(f"长期记忆: {ltm.db_path}")
    print(f"用户ID: {ltm.user_id}")
    if debug:
        print("调试模式: 开启（显示上下文消解和记忆状态）")
    print()
    print("命令:")
    print("  exit/quit - 退出")
    print("  clear     - 清空短期记忆")
    print("  info      - 查看短期记忆状态")
    print("  ltminfo   - 查看长期记忆状态")
    print()

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            ltm.on_session_end(stm, SESSION_ID, entity_tracker)
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            ltm.on_session_end(stm, SESSION_ID, entity_tracker)
            ltm.close()
            break
        if user_input.lower() == "clear":
            stm.clear(SESSION_ID)
            print("🧹 短期记忆已清空\n")
            continue
        if user_input.lower() == "ltminfo":
            prefs = ltm.get_preferences()
            stats = ltm.get_stats()
            print(f"📊 长期记忆: {ltm.user_id}")
            print(f"   数据库: {ltm.db_path}")
            print(f"   统计: {stats['total_sessions']} 次会话, {stats['total_turns']} 轮对话")
            if prefs:
                print(f"   偏好:")
                for key, info in prefs.items():
                    top = info.get("top", [])
                    scores = info.get("scores", {})
                    items = ", ".join(f"{v}({scores.get(v, 0):.2f})" for v in top)
                    print(f"     {key}: {items}")
            else:
                print(f"   偏好: (空)")
            prompt = ltm.get_preference_prompt()
            if prompt:
                print(f"   提示文本:\n{prompt}")
            print()
            continue
        if user_input.lower() == "info":
            info = stm.session_info(SESSION_ID)
            print(f"📊 短期记忆: {info['session_id']}")
            print(f"   轮次: {info['turns']}/{info['max_rounds']} (总计: {info['total_turns']})")
            print(f"   实体槽位: {entity_tracker.slots}")
            print(f"   摘要: {info['summary_count']} 条")
            summaries = stm.get_summaries(SESSION_ID)
            if summaries:
                for i, s in enumerate(summaries, 1):
                    print(f"   摘要{i}: {s}")
            history = stm.get_history(SESSION_ID)
            if history:
                print(f"   历史消息: {len(history)} 条")
                for i, msg in enumerate(history):
                    preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                    print(f"     [{i}] {msg['role']}: {preview}")
            ltm_stats = ltm.get_stats()
            print(f"   长期记忆: {ltm_stats['total_sessions']} 次会话, {ltm_stats['total_turns']} 轮")
            print()
            continue

        entity_tracker.update_from_query(user_input)
        enhanced = entity_tracker.resolve_context(user_input)
        if debug and enhanced != user_input:
            print(f"🔗 上下文消解: '{user_input}' → '{enhanced}'")

        sys_prompt = base_system
        pref_prompt = ltm.get_preference_prompt()
        if pref_prompt:
            sys_prompt += "\n\n" + pref_prompt

        messages = stm.build_messages(SESSION_ID, "", enhanced)
        messages[0]["content"] = sys_prompt

        if debug:
            info = stm.session_info(SESSION_ID)
            print(f"📊 短期记忆: {info['turns']}/{info['max_rounds']} 轮 | 实体槽位: {entity_tracker.slots}")
            if pref_prompt:
                print(f"📊 长期记忆: 已注入偏好提示")

        print("AI: ", end="", flush=True)
        try:
            reply = chat_stream(messages)
            stm.add_turn(SESSION_ID, user_input, reply)
            ltm.sync_from_short_term(stm, SESSION_ID, entity_tracker)
        except Exception as e:
            print(f"\n（出错: {e}）")
            continue

        print()


if __name__ == "__main__":
    main()