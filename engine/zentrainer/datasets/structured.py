"""Structured captions: Fibo 2.0's JSON scene description, written by a local vision-language model.

Fibo 2.0 is trained on, and prompted with, a JSON object per image (short_description, objects,
background_setting, lighting, aesthetics, photographic_characteristics, style_medium, ...).
Bria's VGL writes them with an API model; its prompts and schema are in vgl/ (MIT), and here they
run on the captioner's own text encoder (Qwen3-VL, Gemma 4...). Its output is checked and kept
compact, as Fibo's pipeline serialises a caption; text that isn't a JSON object is rejected, so a
caption set never holds a broken one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VGL = Path(__file__).resolve().parent / "vgl"
FORMATS = ("text", "fibo_json")
KEYS = ("short_description", "objects", "background_setting", "lighting", "aesthetics",
        "photographic_characteristics", "style_medium", "artistic_style", "context", "text_render")
DROP = ("aesthetic_score", "preference_score")  # Fibo's pipeline drops these from a caption


def _read(name: str) -> str:
    return (VGL / name).read_text(encoding="utf-8").strip()


def caption_instruction() -> str:
    """What the model is told to caption an image (VGL's caption system + final prompt, one turn)."""
    system = _read("caption_system.txt").replace("{llm_input_schema}", _read("schema.txt"))
    return f"{system}\n\n{_read('caption_final.txt')}"


def generate_instruction(prompt: str) -> str:
    """What the model is told to expand a short prompt into a caption (VGL's generate prompts)."""
    system = _read("generate_system.txt").replace("{style_preset_block}", "").replace("{llm_input_schema}", _read("schema.txt"))
    return f"{system}\n\n{_read('generate_final.txt').replace('{user_prompt}', prompt.strip())}"


def _prune(v):
    """Empty values out, as Fibo's pipeline does with a caption."""
    if isinstance(v, dict):
        out = {k: _prune(x) for k, x in v.items() if k not in DROP}
        return {k: x for k, x in out.items() if x not in (None, "", [], {})}
    if isinstance(v, list):
        return [x for x in (_prune(x) for x in v) if x not in (None, "", [], {})]
    return v


def parse(text: str) -> dict:
    """The JSON object in a model's answer (code fences, a preamble or trailing commas are
    forgiven); raises ValueError when there's none, or it holds no short_description."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the answer")
    body = t[start:end + 1]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", body))  # trailing commas
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON ({e.msg} at {e.pos})") from None
    if not isinstance(obj, dict):
        raise ValueError("the JSON isn't an object")
    obj = _prune(obj)
    if not isinstance(obj.get("short_description"), str):
        raise ValueError("no short_description")
    # schema order first, anything else the model added after
    return {**{k: obj[k] for k in KEYS if k in obj}, **{k: v for k, v in obj.items() if k not in KEYS}}


def compact(obj: dict) -> str:
    """One line, as Fibo's pipeline serialises a caption."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def is_json_caption(text: str) -> bool:
    t = text.strip()
    if not (t.startswith("{") and t.endswith("}")):
        return False
    try:
        return isinstance(json.loads(t), dict)
    except json.JSONDecodeError:
        return False


def with_trigger(text: str, trigger: str) -> str:
    """A JSON caption with the trigger word in it, at the start of short_description (where a
    plain caption has it in front), unless it's already somewhere in the caption."""
    obj = json.loads(text)
    if trigger.lower() in text.lower():
        return text
    desc = str(obj.get("short_description") or "").strip()
    obj["short_description"] = f"{trigger}, {desc}" if desc else trigger
    return compact({"short_description": obj.pop("short_description"), **obj})
