"""调用本地 vLLM 服务（qwen36_27b_lora）的交互式聊天脚本。

用法示例：
  python call_qwen_model_test.py
  python call_qwen_model_test.py --no-stream

说明：
- 启动后进入交互模式，输入文字后回车即可获得模型回复。
- 输入 exit 或 quit 退出。
"""
import json
import os
import requests
import sys

BASE_URL = "http://14.22.86.97:11001/v1"
MODEL = "qwen36_27b_lora"
PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_prompt.md")


def load_system_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一个有用的AI助手，请用中文回答问题。"


def chat_stream(messages, max_tokens=512, temperature=0.7, api_key=None):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    full_reply = ""
    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        try:
            r.raise_for_status()
        except Exception as e:
            print("流式请求失败：", e, file=sys.stderr)
            print("响应内容：", r.text, file=sys.stderr)
            raise

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


def chat_no_stream(messages, max_tokens=512, temperature=0.7, api_key=None):
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    try:
        r.raise_for_status()
    except Exception as e:
        print("请求失败：", e, file=sys.stderr)
        print("响应内容：", r.text, file=sys.stderr)
        raise

    result = r.json()
    reply = result["choices"][0]["message"]["content"]
    print(reply)
    return reply


def main():
    use_stream = True
    if "--no-stream" in sys.argv:
        use_stream = False

    messages = [{"role": "system", "content": load_system_prompt()}]

    print("=== 交互式聊天已启动 ===")
    print(f"模型: {MODEL}")
    print(f"流式输出: {'开启' if use_stream else '关闭'}")
    print("输入 exit 或 quit 退出\n")

    while True:
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

        messages.append({"role": "user", "content": user_input})

        print("AI: ", end="", flush=True)
        try:
            if use_stream:
                reply = chat_stream(messages)
            else:
                reply = chat_no_stream(messages)
            messages.append({"role": "assistant", "content": reply})
        except Exception:
            messages.pop()
            print("（请求出错，请重试）")

        print()


if __name__ == "__main__":
    main()