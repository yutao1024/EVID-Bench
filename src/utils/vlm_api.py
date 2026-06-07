"""Multi-provider VLM routing (Gemini / Claude / OpenAI / DashScope Qwen)."""

from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def resolve_provider(model: str) -> str:
    name = str(model or "").lower()
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "claude"
    if "qwen" in name:
        return "qwen"
    return "openai"


def normalize_model_name(model: str) -> str:
    text = str(model or "").strip()
    if "/" in text:
        prefix, rest = text.split("/", 1)
        if prefix.lower() in {"google", "openai", "anthropic"}:
            return rest.strip()
    return text


def _extract_text_from_contents(contents: list[Any]) -> str:
    parts: list[str] = []
    for item in contents or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "\n".join(p for p in parts if p).strip()


def _extract_image_blobs_from_contents(contents: list[Any]) -> list[bytes]:
    blobs: list[bytes] = []
    for item in contents or []:
        if not isinstance(item, dict) or item.get("type") != "image_url":
            continue
        url = str((item.get("image_url") or {}).get("url") or "")
        if "," not in url:
            continue
        try:
            blobs.append(base64.b64decode(url.split(",", 1)[1]))
        except Exception:
            continue
    return blobs


def _empty_tokens() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _tokens_from_openai_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return _empty_tokens()
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _require_key(config: dict[str, Any], provider: str) -> str:
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_key = key_map[provider]
    value = str(config.get(env_key) or "").strip()
    if not value:
        raise ValueError(f"{env_key} is not set (required for model provider '{provider}').")
    return value


class ModelAPIHandler:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def _call_openai(
        self,
        *,
        model_name: str,
        prompt_text: str,
        image_paths: list[str],
        image_blobs: list[bytes],
        json_mode: bool,
        temperature: float,
        api_key: str,
        base_url: str | None,
    ) -> tuple[str, dict[str, int]]:
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
        if image_paths:
            for path in image_paths:
                with open(path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
        else:
            for blob in image_blobs:
                img_b64 = base64.b64encode(blob).decode("utf-8")
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": message_content}],
            "temperature": temperature,
        }
        if json_mode and "qwen" not in model_name.lower():
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError(f"vlm_no_content: response has no choices (response={response!r})")
        msg = getattr(choices[0], "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None or not str(content).strip():
            err_obj = getattr(choices[0], "error", None) or getattr(response, "error", None)
            raise ValueError(
                f"vlm_no_content: message.content is empty (provider_error={err_obj!r})"
            )
        return str(content), _tokens_from_openai_usage(getattr(response, "usage", None))

    def _call_gemini(
        self,
        *,
        model_name: str,
        prompt_text: str,
        image_paths: list[str],
        image_blobs: list[bytes],
        json_mode: bool,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        import google.generativeai as genai
        from google.generativeai import types

        genai.configure(api_key=_require_key(self.config, "gemini"))
        uploaded_files: list[Any] = []
        try:
            parts: list[Any] = []
            if image_paths:
                for path in image_paths:
                    uploaded = genai.upload_file(path=path, mime_type="image/jpeg")
                    uploaded_files.append(uploaded)
            elif image_blobs:
                for blob in image_blobs:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(blob)
                        tmp_path = tmp.name
                    try:
                        uploaded = genai.upload_file(path=tmp_path, mime_type="image/jpeg")
                        uploaded_files.append(uploaded)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

            for uploaded in uploaded_files:
                while uploaded.state.name == "PROCESSING":
                    time.sleep(1)
                    uploaded = genai.get_file(uploaded.name)
                if uploaded.state.name == "FAILED":
                    raise ValueError(f"Gemini file upload failed: {uploaded.name}")

            generation_config = types.GenerateContentConfig(temperature=temperature)
            if json_mode:
                generation_config.response_mime_type = "application/json"

            model = genai.GenerativeModel(normalize_model_name(model_name))
            contents = list(uploaded_files) + [prompt_text]
            response = model.generate_content(contents, generation_config=generation_config)
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                raise ValueError("vlm_no_content: Gemini returned empty text.")

            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
            return text, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        finally:
            for uploaded in uploaded_files:
                try:
                    genai.delete_file(uploaded.name)
                except Exception:
                    pass

    def _call_claude(
        self,
        *,
        model_name: str,
        prompt_text: str,
        image_paths: list[str],
        image_blobs: list[bytes],
        json_mode: bool,
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        from anthropic import Anthropic

        client = Anthropic(api_key=_require_key(self.config, "claude"))
        message_content: list[dict[str, Any]] = []

        if image_paths:
            for path in image_paths:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                message_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": data,
                        },
                    }
                )
        else:
            for blob in image_blobs:
                data = base64.b64encode(blob).decode("utf-8")
                message_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": data,
                        },
                    }
                )

        user_text = prompt_text
        if json_mode:
            user_text += "\n\nRespond with a single valid JSON object only."
        message_content.append({"type": "text", "text": user_text})

        response = client.messages.create(
            model=normalize_model_name(model_name),
            messages=[{"role": "user", "content": message_content}],
            max_tokens=8192,
            temperature=temperature,
        )
        chunks = getattr(response, "content", None) or []
        text_parts = [str(getattr(c, "text", "") or "") for c in chunks]
        text = "".join(text_parts).strip()
        if not text:
            raise ValueError("vlm_no_content: Claude returned empty text.")
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        return text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def call_vlm(
        self,
        *,
        model: str,
        contents: list[Any],
        image_paths: list[str] | None = None,
        json_mode: bool = True,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        logger: Any = None,
        log_prefix: str = "",
    ) -> tuple[str, dict[str, int]]:
        prompt_text = _extract_text_from_contents(contents)
        if not prompt_text:
            raise ValueError("VLM call requires non-empty text prompt.")
        paths = [str(p) for p in (image_paths or []) if str(p).strip()]
        blobs = [] if paths else _extract_image_blobs_from_contents(contents)
        model_name = normalize_model_name(model)
        provider = resolve_provider(model_name)

        def _dispatch() -> tuple[str, dict[str, int]]:
            if provider == "gemini":
                return self._call_gemini(
                    model_name=model_name,
                    prompt_text=prompt_text,
                    image_paths=paths,
                    image_blobs=blobs,
                    json_mode=json_mode,
                    temperature=temperature,
                )
            if provider == "claude":
                return self._call_claude(
                    model_name=model_name,
                    prompt_text=prompt_text,
                    image_paths=paths,
                    image_blobs=blobs,
                    json_mode=json_mode,
                    temperature=temperature,
                )
            if provider == "qwen":
                return self._call_openai(
                    model_name=model_name,
                    prompt_text=prompt_text,
                    image_paths=paths,
                    image_blobs=blobs,
                    json_mode=json_mode,
                    temperature=temperature,
                    api_key=_require_key(self.config, "qwen"),
                    base_url=str(self.config.get("DASHSCOPE_BASE_URL") or "").strip() or None,
                )
            return self._call_openai(
                model_name=model_name,
                prompt_text=prompt_text,
                image_paths=paths,
                image_blobs=blobs,
                json_mode=json_mode,
                temperature=temperature,
                api_key=_require_key(self.config, "openai"),
                base_url=None,
            )

        for attempt in range(max_retries):
            try:
                content, tokens = _dispatch()
                if callable(logger):
                    logger(
                        f"{log_prefix}VLM call success on attempt {attempt + 1}/{max_retries} "
                        f"(tokens: {tokens['prompt_tokens']}p + {tokens['completion_tokens']}c = {tokens['total_tokens']}t)"
                    )
                return content, tokens
            except Exception as exc:
                if callable(logger):
                    logger(
                        f"{log_prefix}VLM call failed on attempt {attempt + 1}/{max_retries}: {exc}"
                    )
                if attempt == max_retries - 1:
                    raise RuntimeError(f"VLM call failed after retries: {exc}") from exc
                sleep_s = retry_backoff_sec * (2**attempt)
                if callable(logger):
                    logger(f"{log_prefix}Retrying after {sleep_s:.1f}s")
                time.sleep(sleep_s)
        raise RuntimeError("VLM call failed after retries.")


def call_model_with_retry(
    api_config: dict[str, Any],
    model: str,
    contents: list[Any],
    *,
    image_paths: list[str] | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
    json_mode: bool = True,
    retry_backoff_sec: float = 1.5,
    logger: Any = None,
    log_prefix: str = "",
) -> tuple[str, dict[str, int]]:
    handler = ModelAPIHandler(api_config)
    return handler.call_vlm(
        model=model,
        contents=contents,
        image_paths=image_paths,
        json_mode=json_mode,
        temperature=temperature,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        logger=logger,
        log_prefix=log_prefix,
    )
