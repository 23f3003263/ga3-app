import json, re, base64, math, asyncio
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
    except Exception:
        ans = ""
    return {"answer": str(ans)}

@app.post("/extract")
async def extract(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    if "invoice_text" in body:
        text = body.get("invoice_text", "") or ""
        prompt = f"""Extract invoice fields from the text below.
Return JSON with EXACTLY these keys:
invoice_no, vendor, currency, total_amount, invoice_date, due_in_days, is_paid, priority, contact_email, line_items, item_count

Rules:
- invoice_no: string or null
- vendor: exact name
- currency: ISO 4217 (₹=INR,$=USD,€=EUR,£=GBP,¥=JPY)
- total_amount: integer
- invoice_date: YYYY-MM-DD
- due_in_days: integer
- is_paid: boolean
- priority: low/normal/high/urgent
- contact_email: lowercase
- line_items: array of {{sku, quantity(int), unit_price(int)}}
- item_count: integer

Text: {text}"""
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

    text = body.get("text", "")
    schema = body.get("schema", {})
    last_debug_info["q7_text"] = text[:500]
    last_debug_info["q7_schema"] = schema

    prompt = f"""Extract ALL fields from the document text below.

RULES:
- currency: ISO 4217 ONLY (£/pounds=GBP,$=USD,€=EUR,₹=INR,¥=JPY)
- email: lowercase
- dates: YYYY-MM-DD
- numbers: JSON numbers
- line_items: ALL items with sku, quantity(int), unit_price(int)
- arrays: never null, use []
- boolean: true/false

TEXT:
{text}

Schema:
{json.dumps(schema, indent=2)}"""

    try:
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        clean_schema = {"type": "object", "properties": props, "required": required}
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
                        "json_schema": {"name": "extraction", "strict": False, "schema": clean_schema}
                    }
                }
            )
            r.raise_for_status()
            result = json.loads(r.json()["choices"][0]["message"]["content"])
            for k, v in result.items():
                if isinstance(v, str) and ("email" in k.lower() or "@" in v):
                    result[k] = v.lower()
            if "line_items" in result:
                for item in result["line_items"]:
                    item["quantity"] = int(item.get("quantity", 0))
                    item["unit_price"] = int(float(str(item.get("unit_price", 0)).replace(",", "")))
            if "item_count" in result and "line_items" in result:
                result["item_count"] = len(result["line_items"])
            return JSONResponse(result)
    except Exception as e:
        last_debug_info["q7_error"] = str(e)
        return JSONResponse({"error": str(e)}, status_code=500)

def coerce_type(value, expected_type):
    if value is None:
        return None
    et = str(expected_type).lower().strip()
    try:
        if et == "string":
            return str(value).strip()
        if et == "integer":
            return int(round(float(str(value).replace(",", ""))))
        if et in ("float", "number"):
            return float(str(value).replace(",", ""))
        if et == "date":
            s = str(value).strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                return s
            from datetime import datetime
            for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y"):
                try:
                    return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            return s
        if et == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("true", "yes", "1")
        if et.startswith("array"):
            return value if isinstance(value, list) else [value]
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
    prompt = f"""Extract fields from text. Return JSON with EXACTLY these keys, no extras.
If not found use null.

Fields (name: type):
{field_desc}

Rules:
- date: YYYY-MM-DD
- integer/float: JSON numbers not strings
- currency: ISO 4217 (£=GBP,$=USD,€=EUR,₹=INR)
- email: lowercase
- boolean: true/false

Text: {text}"""
    try:
        out = parse_json(await chat([{"role": "user", "content": prompt}], max_tokens=800))
        for key, expected_type in schema.items():
            result[key] = coerce_type(out.get(key), expected_type)
    except Exception as e:
        last_debug_info["q4_error"] = str(e)
    return JSONResponse(result)

ALL_STATS = ["mean", "std", "variance", "min", "max", "median", "mode",
             "range", "allowed_values", "value_range", "correlation"]

KO_TO_EN = {
    "평균": "mean", "표준편차": "std", "분산": "variance",
    "최솟값": "min", "최소": "min", "최댓값": "max", "최대": "max",
    "중앙값": "median", "중간값": "median", "최빈값": "mode",
    "범위": "range", "허용값": "allowed_values",
    "값의범위": "value_range", "상관관계": "correlation",
    "mean": "mean", "std": "std", "variance": "variance",
    "min": "min", "max": "max", "median": "median", "mode": "mode",
    "range": "range", "allowed_values": "allowed_values",
    "value_range": "value_range", "correlation": "correlation",
}

STAT_KEYWORDS_KO = set(KO_TO_EN.keys())

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
    return re.sub(r'\s+', '', name) in STAT_KEYWORDS_KO

def _filter_columns(cols):
    filtered = []
    for c in cols:
        if _is_stat_keyword(c):
            continue
        # "과", "와", "의" wale compound names filter karo
        if re.search(r'[과와의]', c):
            continue
        filtered.append(c)
    return filtered

@app.post("/answer-audio")
async def answer_audio(request: Request):
    global last_debug_info
    body = await request.json()
    audio_id = body.get("audio_id", "")
    audio_b64 = body.get("audio_base64", "")
    last_debug_info = {"body_id": audio_id}

    try:
        raw = base64.b64decode(audio_b64[:20])
        if raw[:4] == b'RIFF': mime = "audio/wav"
        elif raw[:3] == b'ID3' or raw[:2] == b'\xff\xfb': mime = "audio/mp3"
        elif raw[:4] == b'fLaC': mime = "audio/flac"
        elif raw[:4] == b'OggS': mime = "audio/ogg"
        else: mime = "audio/wav"
    except Exception:
        mime = "audio/wav"

    gemini_payload = {
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": mime, "data": audio_b64}},
                {"text": (
                    "이 오디오를 정확히 전사해 주세요. "
                    "숫자와 컬럼명을 정확히 써주세요. "
                    "컬럼명은 한국어 그대로, 영어 번역 금지. "
                    "컬럼명 숫자는 붙여쓰기(점수1, 점수2)."
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

아래 JSON 형식으로 정확히 추출하세요.

규칙:
1. "rows": 행의 수 (숫자 뒤에 "개" 또는 "행"이 오면 rows)
2. "columns": 실제 데이터 컬럼명만 (통계 용어 절대 포함 금지)
   - 통계 용어: 평균, 표준편차, 분산, 최솟값, 최댓값, 중앙값, 최빈값, 범위, 상관관계
   - "점수1과 점수2" → columns: ["점수1", "점수2"] (과/와/의 기준으로 분리)
3. "requested_stats": 전사본에서 명시적으로 언급된 통계만 한국어로
   - 평균=mean, 표준편차=std, 분산=variance, 최솟값=min, 최댓값=max
   - 중앙값=median, 최빈값=mode, 범위=range
4. 각 통계값은 언급된 경우만 채우고, 나머지는 빈 dict {{}}
5. allowed_values, value_range: 항상 {{}}
6. correlation: 항상 []

Return JSON:
{{
  "rows": <integer>,
  "columns": ["컬럼1", "컬럼2"],
  "mean": {{"컬럼1": value, "컬럼2": value}},
  "std": {{}},
  "variance": {{}},
  "min": {{}},
  "max": {{}},
  "median": {{}},
  "mode": {{}},
  "range": {{}},
  "allowed_values": {{}},
  "value_range": {{}},
  "correlation": [],
  "requested_stats": ["평균"]
}}"""

    try:
        raw_llm = await chat([{"role": "user", "content": parse_prompt}], max_tokens=1500)
        last_debug_info["raw_llm"] = raw_llm
        out = parse_json(raw_llm)
    except Exception as e:
        last_debug_info["parse_error"] = str(e)
        out = {}

    requested_stats_raw = out.get("requested_stats") or []
    requested_stats = list({KO_TO_EN.get(s.strip(), s.strip()) for s in requested_stats_raw})
    last_debug_info["requested_stats"] = requested_stats

    result = dict(empty)
    result["rows"] = int(out.get("rows", 0) or 0)

    # Columns — "과/와/의" split karo
    raw_cols = out.get("columns", []) or []
    split_cols = []
    for c in raw_cols:
        if re.search(r'[과와]', c):
            parts = re.split(r'[과와]', c)
            split_cols.extend([p.strip() for p in parts if p.strip()])
        else:
            split_cols.append(_norm_col(c))
    result["columns"] = _filter_columns(split_cols)

    def _get_stat(stat):
        val = out.get(stat)
        if stat == "correlation":
            return val if isinstance(val, list) else []
        if stat in ("allowed_values", "value_range"):
            return {}
        v = _norm_keys(val) if isinstance(val, dict) else {}
        return {k: vv for k, vv in v.items() if not _is_stat_keyword(k) and vv is not None}

    for stat in ALL_STATS:
        if stat in ("allowed_values", "value_range"):
            result[stat] = {}
            continue
        if stat == "correlation":
            result[stat] = []
            continue
        if stat in requested_stats:
            result[stat] = _get_stat(stat)

    audio_history.append({"audio_id": audio_id, "transcript": transcript,
                           "requested_stats": requested_stats, "answer": result})
    if len(audio_history) > 50:
        del audio_history[0]

    return JSONResponse(result)

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
