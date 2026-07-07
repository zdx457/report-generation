import os

_PROMPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE = {}


def load_prompt(name):
    """加载 prompt 文件内容，支持缓存。

    Args:
        name: 文件名（不含扩展名），如 'intent', 'structure', 'report_generation'

    Returns:
        str: prompt 文本内容
    """
    if name in _CACHE:
        return _CACHE[name]

    path = os.path.join(_PROMPT_DIR, f"{name}.md")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[name] = f.read().strip()
    else:
        _CACHE[name] = ""

    return _CACHE[name]


def reload_prompt(name):
    """强制重新加载 prompt 文件（清除缓存后加载）。

    Args:
        name: 文件名（不含扩展名）
    """
    _CACHE.pop(name, None)
    return load_prompt(name)