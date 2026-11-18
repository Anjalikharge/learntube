"""Provider-agnostic LLM layer: Gemini / OpenAI / Ollama / offline.

Features: per-instance API keys (never written to os.environ, so keys are not shared between
users of a deployed app), disk cache, retry with backoff, JSON mode helper and embeddings.
"""
import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass
class LLMConfig:
    provider: str = "auto"  # auto | gemini | openai | ollama | none
    gemini_key: str = ""
    openai_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1"
    ollama_url: str = "http://localhost:11434"
    use_cache: bool = True
    cache_dir: str = ".cache"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        e = os.getenv
        return cls(
            provider=e("LLM_PROVIDER", "auto").lower(),
            gemini_key=e("GEMINI_API_KEY", ""),
            openai_key=e("OPENAI_API_KEY", ""),
            gemini_model=e("GEMINI_MODEL", "gemini-2.5-flash"),
            openai_model=e("OPENAI_MODEL", "gpt-4o-mini"),
            ollama_model=e("OLLAMA_MODEL", "llama3.1"),
            ollama_url=e("OLLAMA_URL", "http://localhost:11434"),
            cache_dir=e("LEARNTUBE_CACHE", ".cache"),
        )


class LLM:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig.from_env()
        self.calls = 0
        self.cache_hits = 0

    # ---- provider resolution ----
    @property
    def provider(self) -> str:
        p = self.cfg.provider
        if p in ("gemini", "openai", "ollama", "none"):
            return p
        if self.cfg.gemini_key:
            return "gemini"
        if self.cfg.openai_key:
            return "openai"
        return "none"

    def available(self) -> bool:
        return self.provider != "none"

    @property
    def model(self) -> str:
        return {"gemini": self.cfg.gemini_model, "openai": self.cfg.openai_model,
                "ollama": self.cfg.ollama_model}.get(self.provider, "none")

    # ---- cache ----
    def _cache_path(self, prompt: str, temperature: float) -> Path:
        h = hashlib.sha256(f"{self.provider}|{self.model}|{temperature}|{prompt}".encode()).hexdigest()
        return Path(self.cfg.cache_dir) / f"{h}.txt"

    # ---- generation ----
    def _call(self, prompt: str, temperature: float) -> str:
        p = self.provider
        if p == "gemini":
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.cfg.gemini_key)
            r = client.models.generate_content(
                model=self.cfg.gemini_model, contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature))
            return (r.text or "").strip()
        if p == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.cfg.openai_key, timeout=90)
            r = client.chat.completions.create(
                model=self.cfg.openai_model, temperature=temperature,
                messages=[{"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip()
        if p == "ollama":
            body = json.dumps({"model": self.cfg.ollama_model, "prompt": prompt, "stream": False,
                               "options": {"temperature": temperature}}).encode()
            req = urllib.request.Request(self.cfg.ollama_url.rstrip("/") + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())["response"].strip()
        raise RuntimeError("No LLM configured")

    def generate(self, prompt: str, temperature: float = 0.3, retries: int = 3) -> str:
        if not self.available():
            raise RuntimeError("No LLM configured")
        path = self._cache_path(prompt, temperature)
        if self.cfg.use_cache and path.exists():
            self.cache_hits += 1
            return path.read_text(encoding="utf-8")
        last = None
        for attempt in range(retries):
            try:
                out = self._call(prompt, temperature)
                self.calls += 1
                if out and self.cfg.use_cache:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(out, encoding="utf-8")
                return out
            except Exception as e:  # network / rate limit
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last}")

    def generate_json(self, prompt: str, temperature: float = 0.2):
        """Ask for JSON and parse it; retries once with a stricter reminder."""
        for extra in ("", "\n\nReturn ONLY valid JSON, no commentary, no code fences."):
            raw = self.generate(prompt + extra, temperature)
            try:
                return parse_json(raw)
            except Exception:
                continue
        raise ValueError("LLM did not return valid JSON")

    # ---- embeddings (optional, for dense retrieval) ----
    def embed(self, texts):
        import numpy as np
        p = self.provider
        out = []
        if p == "gemini":
            from google import genai
            client = genai.Client(api_key=self.cfg.gemini_key)
            for i in range(0, len(texts), 90):
                r = client.models.embed_content(model="gemini-embedding-001", contents=texts[i:i + 90])
                out += [e.values for e in r.embeddings]
        elif p == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.cfg.openai_key)
            for i in range(0, len(texts), 90):
                r = client.embeddings.create(model="text-embedding-3-small", input=texts[i:i + 90])
                out += [d.embedding for d in r.data]
        else:
            raise RuntimeError("Embeddings need Gemini or OpenAI")
        m = np.array(out, dtype=float)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def parse_json(raw: str):
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    starts = [i for i in (raw.find("["), raw.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON found")
    a = min(starts)
    closer = "]" if raw[a] == "[" else "}"
    b = raw.rfind(closer)
    return json.loads(raw[a:b + 1])
