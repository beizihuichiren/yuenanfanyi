"""Grsai API 连通性测试"""
import os
import requests

url = "https://grsai.dakka.com.cn/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json",
}
body = {
    "model": "gemini-3.1-pro",
    "stream": False,
    "messages": [{"role": "user", "content": "你好"}],
}

try:
    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    print("连通成功:", data["choices"][0]["message"]["content"])
except Exception as e:
    print("连接失败:", e)
