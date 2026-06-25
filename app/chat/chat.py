"""ReAct 多轮推理对话终端（纯推理，无工具/RAG）

用法：
  python chat.py
  python chat.py --debug  # 显示调试信息
"""

import json
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory

SESSION_ID = f"react_{uuid.uuid4().hex[:8]}"
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


def chat_stream(messages, max_tokens=1024, temperature=0.7, prefix=""):
    """流式调用 LLM，边生成边打印，返回完整文本"""
    import requests
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
    CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36-27b")

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
    if prefix:
        print(prefix, end="", flush=True)
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
                    print(content, end="", flush=True)
                    full_text += content
            except json.JSONDecodeError:
                continue
    print()
    return full_text.strip()


def summarize_fn(messages: list[dict]) -> str:
    import requests
    from dotenv import load_dotenv

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


REACT_SYSTEM_PROMPT = """你是一个具备多步推理能力的 AI 助手。你需要通过多轮推理来回答用户问题。

## 输出格式

每轮你必须输出以下两种格式之一：

### 继续推理
```
[CONTINUE]
你对当前问题的推理分析（可以是一段话，也可以是多点分析）
```

### 最终回答
```
[FINAL]
你的最终回答（Markdown 格式，简洁准确）
```

## 工作方式

1. 收到问题后，先输出 [CONTINUE] 进行第一步推理
2. 系统会把你之前的推理都展示给你，你判断是否需要继续
3. 如果还需要更多推理，继续输出 [CONTINUE]
4. 当推理充分后，输出 [FINAL] 给出最终回答

## 重要规则

- 第一轮必须输出 [CONTINUE]，不要直接输出 [FINAL]
- 简单问题至少推理 1 步，复杂问题推理 2-4 步
- [FINAL] 之后不要再输出任何内容"""

MAX_STEPS = 5


def main():
    debug = "--debug" in sys.argv

    stm = ShortTermMemory(max_rounds=5, summarize_fn=summarize_fn)
    ltm = LongTermMemory(user_id=SESSION_ID)

    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
    CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")

    print("=" * 60)
    print("=== ReAct 多轮推理对话（纯推理，无工具） ===")
    print("=" * 60)
    print(f"模型: {CHAT_MODEL}")
    print(f"用户ID: {ltm.user_id}")
    print()
    print("命令:")
    print("  exit/quit - 退出")
    print("  clear     - 清空会话")
    print("  info      - 查看记忆状态")
    print("  直接输入   - 进入 ReAct 推理循环（[CONTINUE]→[FINAL]）")
    print()

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            ltm.on_session_end(stm, SESSION_ID)
            ltm.close()
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            ltm.on_session_end(stm, SESSION_ID)
            ltm.close()
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

        sys_prompt = REACT_SYSTEM_PROMPT
        pref_prompt = ltm.get_preference_prompt()
        history = stm.get_history(SESSION_ID)

        reasoning_steps = []
        final_answer = None
        step = 0

        while step < MAX_STEPS:
            step += 1

            messages = [{"role": "system", "content": sys_prompt}]
            if pref_prompt:
                messages.append({"role": "system", "content": pref_prompt})
            for msg in history:
                if msg.get("content", "").strip():
                    messages.append(msg)
            messages.append({"role": "user", "content": f"用户问题：{query}"})

            if reasoning_steps:
                reasoning_text = "## 你之前的推理\n\n" + "\n\n".join(
                    f"第{i+1}步：[CONTINUE]\n{step_text}"
                    for i, step_text in enumerate(reasoning_steps)
                )
                reasoning_text += "\n\n请判断推理是否充分。如果还需推理，输出 [CONTINUE]；如果推理充分，输出 [FINAL]。"
                messages.append({"role": "assistant", "content": reasoning_text})

            if debug:
                print(f"\n--- 第 {step} 轮 ---")
                for i, msg in enumerate(messages):
                    preview = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
                    print(f"  [{i}] {msg['role']}: {preview}")

            try:
                output = chat_stream(messages, max_tokens=2048, temperature=0.3,
                                     prefix=f"  💭 [第{step}步] " if not debug else "")
            except Exception as e:
                print(f"\nLLM 调用失败: {e}")
                final_answer = f"抱歉，调用模型时出错: {e}"
                break

            if debug:
                print(f"[完整输出]:\n{output}")

            output = output.strip()
            is_final = "[FINAL]" in output

            if is_final:
                idx = output.find("[FINAL]")
                final_answer = output[idx + len("[FINAL]"):].strip()
                if debug:
                    print(f"✅ 推理完成（{step} 步）")
                break
            else:
                reasoning = output
                if "[CONTINUE]" in reasoning:
                    idx = reasoning.find("[CONTINUE]")
                    reasoning = reasoning[idx + len("[CONTINUE]"):]
                reasoning = re.sub(r'\[(CONTINUE|FINAL)\]', '', reasoning).strip()
                if not reasoning:
                    reasoning = output

                reasoning_steps.append(reasoning)

        if final_answer is None:
            if debug:
                print("⚠️ 达到最大步数，基于已有推理生成最终回答")
            force_messages = [{"role": "system", "content": sys_prompt}]
            if pref_prompt:
                force_messages.append({"role": "system", "content": pref_prompt})
            for msg in history:
                if msg.get("content", "").strip():
                    force_messages.append(msg)
            force_messages.append({"role": "user", "content": f"用户问题：{query}"})
            all_reasoning = "\n\n".join(
                f"第{i+1}步推理：{s}" for i, s in enumerate(reasoning_steps)
            )
            force_messages.append({"role": "user", "content": f"以下是你的推理过程：\n\n{all_reasoning}\n\n请基于以上推理，输出最终回答。只输出 [FINAL] 和你的回答，不要输出 [CONTINUE]。\n\n[FINAL]"})
            try:
                force_output = chat_stream(force_messages, max_tokens=2048, temperature=0.3,
                                           prefix="\nAI: ")
                force_output = force_output.strip()
                if "[FINAL]" in force_output:
                    idx = force_output.find("[FINAL]")
                    final_answer = force_output[idx + len("[FINAL]"):].strip()
                else:
                    final_answer = force_output
            except Exception as e:
                final_answer = f"抱歉，调用模型时出错: {e}"
                print(f"\nAI: {final_answer}")

        print()
        stm.add_turn(SESSION_ID, query, final_answer)
        ltm.sync_from_short_term(stm, SESSION_ID)


if __name__ == "__main__":
    main()