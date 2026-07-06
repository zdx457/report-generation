import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
chat_dir = os.path.join(project_root, "app", "chat")
app_dir = os.path.join(project_root, "app")
sys.path.insert(0, app_dir)
os.chdir(chat_dir)

sys.argv = ["rag_chat_v2.py", "--web"]

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag_chat_v2", os.path.join(chat_dir, "rag_chat_v2.py"))
    rag_chat_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag_chat_module)
    rag_chat_module.main()
except FileNotFoundError:
    print("错误: rag_chat_v2.py 未找到")
    sys.exit(1)
except Exception as e:
    print(f"启动失败: {e}")
    sys.exit(1)