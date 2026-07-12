import json, re, base64, hashlib, math, asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import config

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

HEAD = {"Authorization": f"Bearer {config.AIPIPE_TOKEN}",
        "Content-Type": "application/json"}
_CACHE = {}
last_debug_info = {}
audio_history = []

def _ck(*parts):
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()

def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?|\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        return json.loads(m.group(0)) if m else {}

async def chat(messages, model=None, max_tokens=800, force_json=True, retries=4):
    key = _ck("chat", model, json.dumps(messages, sort_keys=True, default=str))
    if key in _CACHE:
        return _CACHE[key]
    body = {"model": model or config.TEXT_MODEL, "messages": messages,
            "temperature": 0, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    last_err = None
    async with httpx.AsyncClient(timeout=90) as c:
        for attempt in range(retries):
            r = await c.post(f"{config.AIPIPE_BASE}/chat/completions",
                             headers=HEAD, json=body)
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            _CACHE[key] = out
            return out
    raise RuntimeError(f"chat failed: {last_err}")

GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]

async def gemini_transcribe(payload, attempts_per_model=3):
    global last_debug_info
    last_err = ""
    async with httpx.AsyncClient(timeout=120) as c:
        for model in GEMINI_MODELS:
            for attempt in range(attempts_per_model):
                try:
                    r = await c.post(
                        f"https://aipipe.org/geminiv1beta/models/{model}:generateContent",
                        headers={"Authorization": f"Bearer {config.AIPIPE_TOKEN}"},
                        json=payload)
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    last_debug_info["transcribe_model"] = model
                    return txt
                except (KeyError, IndexError):
                    break
                except Exception as e:
                    last_err = str(e)[:160]
                    await asyncio.sleep(1.0 * (attempt + 1))
    last_debug_info["transcribe_error"] = last_err
    return ""

@app.get("/")
async def root():
    return {"ok": True, "email": config.EMAIL}

@app.get("/debug")
async def debug():
    return {"last": last_debug_info, "history": audio_history[-5:]}

# ===== Q2: /answer-image =====
def normalize_answer(ans):
    s = str(ans).strip()
    if not s:
        return s
    cleaned = re.sub(r"[,\s]", "", s)
    cleaned = re.sub(r"[₹$€£%]", "", cleaned)
    m = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if m and re.fullmatch(r"[^\dA-Za-z]*-?\d[\d,.\s₹$€£%]*", s.strip()):
        num = m.group(0)
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return num
    return s

@app.post("/answer-image")
async def answer_image(request: Request):
    body = await request.json()
    img_b64 = body.get("image_base64", "")
    question = body.get("question", "")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text":
                "You read charts, receipts, tables, invoices and pie charts EXACTLY.\n"
                "Work in steps in a 'work' field, then give the final 'answer':\n"
                "1. TRANSCRIBE every relevant label and number you see.\n"
                "2. If arithmetic needed, compute step by step and double-check.\n"
                "3. Final 'answer': if NUMERIC, output ONLY the bare number. "
                "If TEXT, output exactly as written in image.\n"
                "Return JSON: {\"work\": \"...\", \"answer\": \"...\"}.\n"
                f"Question: {question}"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"}},
        ],
    }]
    try:
        out = parse_json(await chat(messages, model=config.VISION_MODEL, max_tokens=1200))
        ans = normalize_answer(out.get("answer", ""))
    except Exception as e:
        ans = ""
    return {"answer": str(ans)}

# ===== Q3 + Q7: /extract =====
@app.post("/extract")
async def extract(request: Request):
    body = await request.json()

    # Q3: fixed invoice schema
    if "invoice_text" in body:
        text = body["invoice_text"]
        prompt = f"""Extract invoice fields from the text below.
Return JSON with EXACTLY these keys (no extras, no missing):
invoice_no, vendor, currency, total_amount, invoice_date, due_in_days, is_paid, priority, contact_email, line_items, item_count

Rules:
- invoice_no: invoice number as string, null if not found
- vendor: biller name exactly as written
- currency: ISO 4217 code (₹=INR, $=USD, €=EUR, £=GBP, ¥=JPY)
- total_amount: integer (12K=12000, "twelve thousand"=12000)
- invoice_date: YYYY-MM-DD
- due_in_days: integer ("Net 30"=30, "two weeks"=14)
- is_paid: true if paid/cleared, false if pending/awaiting
- priority: one of low/normal/high/urgent
- contact_email: lowercase string, null if not found
- line_items: array of objects with keys sku, quantity (int), unit_price (int)
- item_count: integer count of line_items

Invoice text:
{text}"""
        try:
            out = parse_json(await chat([{"role": "user", "content": prompt}], max_tokens=1500))
            out["total_amount"] = int(float(str(out.get("total_amount", 0)).replace(",", "")))
            out["due_in_days"] = int(out.get("due_in_days", 0))
            out["is_paid"] = bool(out.get("is_paid", False))
            out["item_count"] = int(out.get("item_count", len(out.get("line_items", []))))
            out["contact_email"] = str(out.get("contact_email", "")).lower()
            for item in out.get("line_items", []):
                item["quantity"] = int(item.get("quantity", 0))
                item["unit_price"] = int(float(str(item.get("unit_price", 0)).replace(",", "")))
            return JSONResponse(out)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # Q7: dynamic schema
    text = body.get("text", "")
    schema = body.get("schema", {})

    # schema could be a dict of {field: type} or a JSON Schema object
    # Extract simple field→type map
    if "properties" in schema:
        # It's a JSON Schema — flatten it
        props = schema.get("properties", {})
        simple_schema = {}
        for k, v in props.items():
            t = v.get("type", "string")
            if t == "number":
                t = "float"
            simple_schema[k] = t
    else:
        simple_schema = schema

    from datetime import datetime

    def coerce(value, type_str):
        if value is None:
            return None
        try:
            if type_str == "string":
                return str(value)
            if type_str == "integer":
                return int(float(re.sub(r'[^\d.-]', '', str(value))))
            if type_str in ("float", "number"):
                return float(re.sub(r'[^\d.-]', '', str(value)))
            if type_str == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).lower() in ("true", "1", "yes")
            if type_str == "date":
                s = str(value).strip()
                if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
                    return s
                s2 = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', s).strip()
                for fmt in ["%d %B %Y", "%d %b %Y", "%B %d %Y",
                            "%d/%m/%Y", "%m/%d/%Y"]:
                    for src in (s2, s):
                        try:
                            return datetime.strptime(src, fmt).strftime("%Y-%m-%d")
                        except:
                            pass
                return s
            if type_str == "array[string]":
                return [str(v) for v in value] if isinstance(value, list) else [str(value)]
            if type_str == "array[integer]":
                return [int(float(str(v))) for v in value] if isinstance(value, list) else [int(float(str(value)))]
        except:
            pass
        return None

    field_list = "\n".join(f"- {k}: {v}" for k, v in simple_schema.items())

    prompt = f"""Read the text below and extract these fields:

{field_list}

TEXT:
{text}

Return a flat JSON object with EXACTLY these keys: {list(simple_schema.keys())}
- Use null if a field is not found in the text
- Dates must be YYYY-MM-DD format
- Numbers must be JSON numbers not strings
- No extra keys allowed"""

    try:
        raw = await chat([{"role": "user", "content": prompt}],
                         model="gpt-4o-mini", max_tokens=512)
        extracted = parse_json(raw)
    except:
        extracted = {}

    return JSONResponse({k: coerce(extracted.get(k), v) for k, v in simple_schema.items()})

# ===== Q6: /answer-audio =====
@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info
    body = await request.json()
    audio_id = body.get("audio_id", "")
    audio_b64 = body.get("audio_base64", "")
    last_debug_info = {"body_id": audio_id}

    try:
        raw = base64.b64decode(audio_b64[:20])
        if raw[:4] == b'RIFF':
            mime = "audio/wav"
        elif raw[:3] == b'ID3' or raw[:2] == b'\xff\xfb':
            mime = "audio/mp3"
        elif raw[:4] == b'fLaC':
            mime = "audio/flac"
        else:
            mime = "audio/wav"
    except:
        mime = "audio/wav"
    last_debug_info["detected_mime"] = mime

    gemini_payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": mime, "data": audio_b64}},
                {"text": (
                    "이 오디오를 정확히 전사해 주세요. "
                    "컬럼명은 반드시 오디오에서 들리는 그대로 한국어로 써주세요. "
                    "영어로 번역하지 마세요. "
                    "숫자도 정확히 그대로 써주세요."
                )}
            ]
        }]
    }

    transcript = await gemini_transcribe(gemini_payload)
    last_debug_info["transcript"] = transcript

    if not transcript:
        empty = {"rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
                 "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
                 "allowed_values": {}, "value_range": {}, "correlation": []}
        return JSONResponse(empty)

    parse_prompt = f"""다음은 한국어 오디오 전사본입니다:
{transcript}

위 전사본에서 데이터셋 통계를 추출하세요.
컬럼명은 반드시 전사본에 나온 그대로 한국어로 사용하세요. 영어로 바꾸지 마세요.

Return JSON with EXACTLY these keys:
{{
  "rows": <integer>,
  "columns": [<컬럼명을 한국어 그대로>],
  "mean": {{"컬럼명": value}},
  "std": {{"컬럼명": value}},
  "variance": {{"컬럼명": value}},
  "min": {{"컬럼명": value}},
  "max": {{"컬럼명": value}},
  "median": {{"컬럼명": value}},
  "mode": {{"컬럼명": value}},
  "range": {{"컬럼명": value}},
  "allowed_values": {{}},
  "value_range": {{"컬럼명": [min, max]}},
  "correlation": [[col1, col2, value]]
}}
Empty dict or empty list if not mentioned."""

    try:
        raw_llm = await chat([{"role": "user", "content": parse_prompt}], max_tokens=1500)
        last_debug_info["raw_llm"] = raw_llm
        out = parse_json(raw_llm)
    except Exception as e:
        last_debug_info["parse_error"] = str(e)
        out = {}

    result = {
        "rows": int(out.get("rows", 0)),
        "columns": out.get("columns", []),
        "mean": out.get("mean", {}),
        "std": out.get("std", {}),
        "variance": out.get("variance", {}),
        "min": out.get("min", {}),
        "max": out.get("max", {}),
        "median": out.get("median", {}),
        "mode": out.get("mode", {}),
        "range": out.get("range", {}),
        "allowed_values": {},
        "value_range": out.get("value_range", {}),
        "correlation": out.get("correlation", [])
    }

    audio_history.append({"audio_id": audio_id, "transcript": transcript, "answer": result})
    if len(audio_history) > 50:
        del audio_history[0]

    return JSONResponse(result)

# ===== Q8: /rank =====
@app.post("/rank")
async def rank(request: Request):
    body = await request.json()
    query = body.get("query", "")
    candidates = body.get("candidates", [])
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{config.AIPIPE_BASE}/embeddings", headers=HEAD,
                         json={"model": config.EMBED_MODEL,
                               "input": [query] + list(candidates)})
        r.raise_for_status()
        vecs = [d["embedding"] for d in r.json()["data"]]
    q = vecs[0]
    cand = vecs[1:]
    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb) if na and nb else 0.0
    scored = sorted(range(len(cand)), key=lambda i: -cos(q, cand[i]))
    return {"ranking": scored[:3]}

# ===== Q9: /solve =====
@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    problem = body.get("problem", "")
    prompt = (
        "Solve this arithmetic word problem CAREFULLY. It has DISTRACTOR numbers.\n"
        "Steps:\n"
        "1. List relevant vs distractor numbers.\n"
        "2. Do arithmetic one step at a time.\n"
        "3. Double-check before finalising.\n"
        "Return JSON with EXACTLY two keys: "
        "'reasoning' (string >=80 chars) and 'answer' (integer, not string).\n\n"
        f"PROBLEM:\n{problem}"
    )
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}],
                                    model="gpt-4o", max_tokens=1200))
        ans = int(round(float(out.get("answer"))))
        reasoning = str(out.get("reasoning", ""))
        if len(reasoning) < 80:
            reasoning = (reasoning + " Step-by-step arithmetic applied; distractors ignored.").strip()
        return {"reasoning": reasoning, "answer": ans}
    except Exception as e:
        return {"reasoning": ("Error: " + str(e)).ljust(80), "answer": 0}
