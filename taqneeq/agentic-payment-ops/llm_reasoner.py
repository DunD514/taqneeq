"""
LLM-based hypothesis generator. Uses Gemini 2.5 Flash to interpret metrics
and output strict JSON hypotheses. No actions; interpretation only.
"""
import json
import os
import re
from typing import Optional

from models import Hypothesis, WindowMetrics


# Conservative prompt: strict JSON only, no suggested actions.
REASONING_PROMPT = """You are a payment operations analyst. Given the following metrics from a sliding window of payment traffic, identify any emerging failure pattern and output exactly one hypothesis in valid JSON.

Be conservative: if no strong pattern exists, return a hypothesis with cause "Unknown" and confidence 0.0.

Metrics:
- success_rate: {success_rate:.2%}
- p95_latency_ms: {p95_latency_ms:.1f}
- retry_amplification: {retry_amplification:.2f}
- success_rate_by_issuer: {success_rate_by_issuer}
- error_distribution: {error_distribution}

Output ONLY a single JSON object with these exact keys (no markdown, no extra text, no suggested actions):
{{"cause": "<short identifier e.g. Issuer_HDFC_Degradation or Unknown>", "confidence": <float 0-1>, "evidence": "<one sentence summary>"}}
"""


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from LLM response (handles markdown code blocks)."""
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_hypothesis_llm(metrics: WindowMetrics) -> Optional[Hypothesis]:
    """
    Call Gemini 2.5 Flash to generate a single hypothesis from metrics.
    Returns None on API error or invalid response; caller must fall back to heuristics.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = REASONING_PROMPT.format(
            success_rate=metrics.success_rate,
            p95_latency_ms=metrics.p95_latency_ms,
            retry_amplification=metrics.retry_amplification,
            success_rate_by_issuer=json.dumps(metrics.success_rate_by_issuer, indent=0),
            error_distribution=json.dumps(metrics.error_distribution, indent=0),
        )
        try:
            config = genai.types.GenerationConfig(temperature=0.2, max_output_tokens=256)
        except (AttributeError, TypeError):
            config = {"temperature": 0.2, "max_output_tokens": 256}
        response = model.generate_content(prompt, generation_config=config)
        content = getattr(response, "text", None) or ""
        if not content and response.candidates:
            parts = response.candidates[0].content.parts
            content = (parts[0].text if parts else "") or ""
        content = content.strip()
        data = _extract_json(content)
        if not data or "cause" not in data or "confidence" not in data:
            return None
        cause = str(data.get("cause", "Unknown"))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        evidence = str(data.get("evidence", ""))
        return Hypothesis(cause=cause, confidence=confidence, evidence=evidence, source="llm")
    except Exception:
        return None
