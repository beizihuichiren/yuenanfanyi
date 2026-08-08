import logging
import time
from typing import Optional

import requests

from .config import get_openai_proxy_settings

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(self, settings: Optional[object] = None) -> None:
        self.settings = settings
        self.api_key = getattr(self.settings, "openai_api_key", "") if self.settings else ""
        self.model = getattr(self.settings, "openai_model", "gpt-4o-mini") if self.settings else "gpt-4o-mini"
        self.base_url = getattr(self.settings, "openai_api_base", "https://api.openai.com/v1") if self.settings else "https://api.openai.com/v1"

    def translate(self, text: str, retries: int = 2) -> str:
        api_key, base_url, model = get_openai_proxy_settings()
        if not api_key:
            logger.warning("OPENAI_API_KEY is not configured, returning original text")
            return text

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": (
                    "你是专业中越翻译。将用户提供的中文文本翻译成自然流畅的越南语。\n"
                    "规则：\n"
                    "1. 只输出翻译结果，不要解释、不要注释、不要emoji\n"
                    "2. 如果输入含乱码或错别字，根据上下文推断意思并翻译\n"
                    "3. 译文要简洁，适合做字幕（每条不超过15个词）\n"
                    "4. 保持口语化，符合短剧对话风格"
                )},
                {"role": "user", "content": text},
            ],
        }

        logger.info("Starting translation request via %s", base_url)
        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=20,
                )
                response.raise_for_status()
                choice = response.json()["choices"][0]
                translated = choice["message"]["content"].strip()
                if translated:
                    logger.info("Translation succeeded")
                    return translated
            except requests.RequestException as exc:
                logger.warning("Translation attempt %s failed: %s", attempt, exc)
                if attempt == retries:
                    break
                time.sleep(2 ** (attempt - 1))

        logger.warning("Translation unavailable, returning original text")
        return text
