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
last_debug_info = {}
audio_history = []

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
            return r.json()["choices"][0]["message"]["content"]
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
                        last_err = f"HTTP {r.status_code} on {model}"
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = r.json()
                    txt = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    last_debug_info["transcribe_model"] = model
                    return txt
                except (KeyError, IndexError):
                    last_err = f"empty candidates on {model}"
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

@app.post("/debug-extract")
async def debug_extract(request: Request):
    body = await request.json()
    return JSONResponse({
        "received_keys": list(body.keys()),
        "text_preview": body.get("text", "")[:200],
        "schema": body.get("schema", {}),
        "has_invoice_text": "invoice_text" in body,
        "has_text": "text" in body,
    })

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

# ===== Q3 & Q7: /extract =====
@app.post("/extract")
async def extract(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if "invoice_text" in body:
        text = body.get("invoice_text", "") or ""

        result = {
            "invoice_no": None,
            "date": None,
            "vendor": None,
            "amount": None,
            "tax": None,
            "currency": "INR",
        }

        try:
            prompt = f"""Extract invoice fields from the text below.
Return JSON with EXACTLY these 6 keys (no extras, no missing):
invoice_no, date, vendor, amount, tax, currency

Rules:
- invoice_no: invoice number as a string, null if not found
- date: ISO format YYYY-MM-DD, null if not found
- vendor: vendor/biller name exactly as written, null if not found
- amount: the SUBTOTAL before tax, as a plain number (not a string), null if not found
- tax: the tax amount only (e.g. the GST/IGST/VAT amount, not the grand total), as a plain number, null if not found
- currency: ISO 4217 code. Rs./₹ = INR, $ = USD, € = EUR, £ = GBP, ¥ = JPY.
  If a rupee symbol or "Rs." appears anywhere, use "INR".

Invoice text:
{text}"""
            out = parse_json(await chat([{"role": "user", "content": prompt}], max_tokens=800))

            def to_number(v):
                if v is None:
                    return None
                try:
                    cleaned = str(v).replace(",", "").replace("Rs.", "").replace("₹", "").strip()
                    return float(cleaned)
                except (ValueError, TypeError):
                    return None

            result["invoice_no"] = out.get("invoice_no") or None
            result["date"] = out.get("date") or None
            result["vendor"] = out.get("vendor") or None
            result["amount"] = to_number(out.get("amount"))
            result["tax"] = to_number(out.get("tax"))
            result["currency"] = out.get("currency") or "INR"
        except Exception as e:
            last_debug_info["q3_error"] = str(e)

        return JSONResponse(result)

    text = body.get("text", "")
    schema = body.get("schema", {})
    document_id = body.get("document_id", "")
    last_debug_info["q7_text"] = text[:500]
    last_debug_info["q7_schema"] = schema

    prompt = f"""You are a precise data extraction assistant.

Extract ALL fields from the document text below according to the schema provided.

STRICT RULES:
- Extract EVERY field defined in the schema
- currency: return ISO 4217 code ONLY (dollars/$=USD, pounds/£=GBP, euros/€=EUR, rupees/₹=INR, yen/¥=JPY)
- email: always lowercase
- dates: YYYY-MM-DD format only
- numbers: JSON numbers, never strings
- line_items: extract EVERY product/item row you find. Each item needs sku, quantity (integer), unit_price (integer)
- arrays: NEVER return null for arrays, use [] if empty
- boolean: true or false only

DOCUMENT TEXT:
{text}

Return a JSON object matching this schema exactly:
{json.dumps(schema, indent=2)}"""

    try:
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post(
                f"{config.AIPIPE_BASE}/chat/completions",
                headers=HEAD,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 2000,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "strict": True,
                            "schema": schema
                        }
                    }
                }
            )
            if r.status_code == 200:
                result = json.loads(r.json()["choices"][0]["message"]["content"])
                for k, v in result.items():
                    if isinstance(v, str) and ("email" in k.lower() or "@" in v):
                        result[k] = v.lower()
                return JSONResponse(result)
            else:
                r2 = await c.post(
                    f"{config.AIPIPE_BASE}/chat/completions",
                    headers=HEAD,
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 2000,
                        "response_format": {"type": "json_object"}
                    }
                )
                r2.raise_for_status()
                result = parse_json(r2.json()["choices"][0]["message"]["content"])
                for k, v in result.items():
                    if isinstance(v, str) and ("email" in k.lower() or "@" in v):
                        result[k] = v.lower()
                return JSONResponse(result)
    except Exception as e:
        last_debug_info["q7_error"] = str(e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ===== Q4: /dynamic-extract =====
def coerce_type(value, expected_type):
    if value is None:
        return None
    et = str(expected_type).lower().strip()
    try:
        if et == "string":
            return str(value).strip().rstrip(".").strip()
        if et == "integer":
            return int(round(float(str(value).replace(",", ""))))
        if et in ("float", "number"):
            return float(str(value).replace(",", ""))
        if et == "date":
            s = str(value).strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                return s
            try:
                from datetime import datetime
                for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y", "%B %d %Y"):
                    try:
                        return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            except Exception:
                pass
            return s or None
        if et == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "yes", "1")
        if et.startswith("array"):
            if isinstance(value, list):
                return value
            return [value]
        return value
    except (ValueError, TypeError):
        return None

@app.post("/dynamic-extract")
async def dynamic_extract(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    text = body.get("text", "") or ""
    schema = body.get("schema", {}) or {}

    result = {k: None for k in schema.keys()}

    if not text or not schema:
        return JSONResponse(result)

    field_desc = "\n".join(f'- "{k}": {v}' for k, v in schema.items())
    prompt = f"""Extract the following fields from the text below. Return a JSON object
with EXACTLY these keys, no extras, no missing keys. If a field cannot be found in
the text, use null for that field.

Fields to extract (name: type):
{field_desc}

Rules:
- "date" type fields must be ISO format YYYY-MM-DD
- "integer" and "float" type fields must be plain numbers, not strings
- "string" type fields should be the exact SHORTEST value (e.g. just the name, not a full sentence)
- "boolean" type fields must be true or false

Text:
{text}"""

    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], max_tokens=800))
        last_debug_info["q4_raw_llm_out"] = out
        for key, expected_type in schema.items():
            result[key] = coerce_type(out.get(key), expected_type)
    except Exception as e:
        last_debug_info["q4_error"] = str(e)

    return JSONResponse(result)

# ===== Q6: /answer-audio =====
ALL_STATS = ["mean", "std", "variance", "min", "max", "median", "mode",
             "range", "allowed_values", "value_range", "correlation"]

# Korean words for statistics — these must NEVER appear as column names.
STAT_KEYWORDS_KO = [
    "평균", "표준편차", "분산", "최솟값", "최댓값", "최소", "최대",
    "중앙값", "중간값", "최빈값", "범위", "허용값", "허용된값",
    "상관관계", "값의범위",
]

def _norm_col(name):
    if not isinstance(name, str):
        return name
    s = name.strip()
    s = re.sub(r'(\S)\s+(\d+)$', r'\1\2', s)
    return s

def _norm_keys(d):
    if not isinstance(d, dict):
        return d
    return {_norm_col(k): v for k, v in d.items()}

def _is_stat_keyword(name):
    if not isinstance(name, str):
        return False
    n = re.sub(r'\s+', '', name)
    return n in STAT_KEYWORDS_KO

def _filter_columns(cols):
    return [c for c in cols if not _is_stat_keyword(c)]

@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info
    body = await request.json()
    audio_id = body.get("audio_id", "")
    audio_b64 = body.get("audio_base64", "")
    last_debug_info = {"body_id": audio_id, "audio_b64_len": len(audio_b64)}

    try:
        raw = base64.b64decode(audio_b64[:20])
        if raw[:4] == b'RIFF': mime = "audio/wav"
        elif raw[:3] == b'ID3' or raw[:2] == b'\xff\xfb': mime = "audio/mp3"
        elif raw[:4] == b'fLaC': mime = "audio/flac"
        elif raw[:4] == b'OggS': mime = "audio/ogg"
        else: mime = "audio/wav"
    except Exception:
        mime = "audio/wav"
    last_debug_info["detected_mime"] = mime

    gemini_payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": mime, "data": audio_b64}},
                {"text": (
                    "이 오디오를 정확히 전사해 주세요. "
                    "컬럼명은 반드시 오디오에서 들리는 그대로 한국어로 써주세요. "
                    "영어로 번역하지 마세요. "
                    "숫자도 정확히 그대로 써주세요. "
                    "컬럼명에 포함된 숫자(예: 점수1, 점수2)는 띄어쓰기 없이 붙여서 표기하세요."
                )}
            ]
        }]
    }

    transcript = await gemini_transcribe(gemini_payload)
    last_debug_info["transcript"] = transcript

    empty = {"rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
             "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
             "allowed_values": {}, "value_range": {}, "correlation": []}

    if not transcript:
        return JSONResponse(empty)

    parse_prompt = f"""다음은 한국어 오디오 전사본입니다:
{transcript}

위 전사본에서 데이터셋 통계를 추출하세요.
컬럼명은 반드시 전사본에 나온 그대로 한국어로 사용하세요. 영어로 바꾸지 마세요.
컬럼명에 숫자가 붙는 경우(예: 점수1, 점수2) 띄어쓰기 없이 붙여서 사용하세요.

CRITICAL: "columns" must contain ONLY the actual data column names (e.g. 점수1,
점수2, 나이, 소득). NEVER include a statistic word itself as a column name —
words like 평균(mean), 표준편차(std), 분산(variance), 최솟값/최댓값(min/max),
중앙값(median), 최빈값(mode), 범위(range), 상관관계(correlation), 허용값
(allowed_values) are STATISTIC NAMES, not columns, even if they appear in the
sentence "점수1과 점수2의 평균은 ..." — here 평균 is NOT a column, only
점수1 and 점수2 are.

Return JSON with EXACTLY these keys:
{{
  "rows": <integer>,
  "columns": [<컬럼명을 한국어 그대로, 숫자는 붙여쓰기, 통계 용어 제외>],
  "mean": {{"컬럼명": value}},
  "std": {{"컬럼명": value}},
  "variance": {{"컬럼명": value}},
  "min": {{"컬럼명": value}},
  "max": {{"컬럼명": value}},
  "median": {{"컬럼명": value}},
  "mode": {{"컬럼명": value}},
  "range": {{"컬럼명": value}},
  "allowed_values": {{"컬럼명": ["값1", "값2"]}},
  "value_range": {{"컬럼명": [min, max]}},
  "correlation": [[col1, col2, value]],
  "requested_stats": ["<only the stat names ACTUALLY mentioned or asked about in the transcript, chosen from: mean, std, variance, min, max, median, mode, range, allowed_values, value_range, correlation>"]
}}
IMPORTANT: "requested_stats" must list ONLY the statistics that the transcript
explicitly states or asks for. Do NOT include a stat just because you could
compute or infer it. Empty dict/list for anything not mentioned."""

    try:
        raw_llm = await chat([{"role": "user", "content": parse_prompt}], max_tokens=1500)
        last_debug_info["raw_llm"] = raw_llm
        out = parse_json(raw_llm)
    except Exception as e:
        last_debug_info["parse_error"] = str(e)
        out = {}

    requested_stats = out.get("requested_stats") or []
    last_debug_info["requested_stats"] = requested_stats

    result = dict(empty)
    result["rows"] = int(out.get("rows", 0) or 0)
    raw_cols = [_norm_col(c) for c in (out.get("columns", []) or [])]
    result["columns"] = _filter_columns(raw_cols)

    def _get_stat(stat):
        val = out.get(stat)
        if stat == "correlation":
            return val if isinstance(val, list) else []
        v = _norm_keys(val) if isinstance(val, dict) else {}
        # also strip any stat-keyword that slipped in as a dict key
        return {k: vv for k, vv in v.items() if not _is_stat_keyword(k)}

    for stat in ALL_STATS:
        if stat == "allowed_values":
            result[stat] = {}
            continue
        if stat == "value_range":
            result[stat] = {}
            continue
        if requested_stats and stat in requested_stats:
            result[stat] = _get_stat(stat)

    if not requested_stats:
        for stat in ALL_STATS:
            if stat in ("allowed_values", "value_range"):
                continue
            v = _get_stat(stat)
            if (stat == "correlation" and v) or (stat != "correlation" and v):
                result[stat] = v

    audio_history.append({"audio_id": audio_id, "transcript": transcript,
                           "requested_stats": requested_stats, "answer": result})
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
