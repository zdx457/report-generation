import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
chat_dir = os.path.join(project_root, "app", "chat")
app_dir = os.path.join(project_root, "app")
sys.path.insert(0, app_dir)
os.chdir(chat_dir)

sys.argv = ["main.py", "--web"]

try:
    from chat.main import main
    main()
except Exception as e:
    print(f"启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
