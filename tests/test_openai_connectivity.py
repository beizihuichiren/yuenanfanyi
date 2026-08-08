import os

import requests


def test_openai_proxy_connectivity() -> None:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")

    candidates = [
        ("gpt-5.5-pro", "/chat/completions", {"messages": [{"role": "user", "content": "请回复一个短句：接口连通测试成功。"}]}, False),
        ("gpt-5.5-pro", "/completions", {"prompt": "请回复一个短句：接口连通测试成功。"}, False),
        ("gpt-4o-mini", "/chat/completions", {"messages": [{"role": "user", "content": "请回复一个短句：接口连通测试成功。"}]}, False),
        ("gpt-4o-mini", "/completions", {"prompt": "请回复一个短句：接口连通测试成功。"}, False),
        ("gpt-4o", "/chat/completions", {"messages": [{"role": "user", "content": "请回复一个短句：接口连通测试成功。"}]}, False),
        ("gpt-4o", "/completions", {"prompt": "请回复一个短句：接口连通测试成功。"}, False),
    ]

    last_error = None
    for model, path, payload, _ in candidates:
        data = {"model": model, "temperature": 0.2}
        data.update(payload)
        response = requests.post(
            f"{base_url}{path}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=data,
            timeout=60,
        )
        if response.status_code == 200:
            payload_json = response.json()
            assert payload_json.get("choices") or payload_json.get("id"), payload_json
            return
        last_error = response.text

    assert False, last_error or "No candidate succeeded"
