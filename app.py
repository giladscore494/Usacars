# -*- coding: utf-8 -*-
# ===========================================================
# 🚗 AI Deal Checker - U.S. Edition (Pro) v10.3.0 (Stable Full)
# Full v2.0 Analyst Spec | ROI 12/24/36m | Risk Tier | Buyer Fit | Compliance
# Auto Theme (Android + iOS Safari) | Warranty-aware Reliability | Mandatory Detailed Explanation
# Gemini 2.5 Pro | Sheets Integration | Insurance & Depreciation Tables | History Cache
# ===========================================================

import os
import re
import io
import json
import time
import html
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import streamlit as st
from json_repair import repair_json

# Optional Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except Exception:
    gspread = None
    Credentials = None
    SHEETS_AVAILABLE = False

# Optional Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
class AppConfig:
    APP_VERSION = "10.3.0"
    LOCAL_FILE = "deal_history_us.json"
    MAX_INPUT_LENGTH = 15000
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 180
    CACHE_TTL = 600
    MEMORY_LIMIT = 800
    MAX_PRICE_VALUE = 1_000_000
    MIN_COMPONENT = 0
    MAX_COMPONENT = 100

CONFIG = AppConfig()
st.set_page_config(page_title=f"AI Deal Checker (v{CONFIG.APP_VERSION})", page_icon="🚗", layout="centered")


# -------------------------------------------------------------
# THEME
# -------------------------------------------------------------
def inject_auto_theme():
    st.markdown("""
    <style>
      :root { color-scheme: light dark; }
      html, body { background: var(--bg, #fff) !important; }
      body, p, label, div, span, h1, h2, h3 { color: var(--fg, #0f172a) !important; }
      .metric { display:flex; justify-content:space-between; margin:6px 0; font-size:0.95rem; }
      .progress { height:10px; background:#E5E7EB55; border-radius:6px; overflow:hidden; }
      .fill-ok { height:100%; background:#16a34a; }
      .fill-warn { height:100%; background:#f59e0b; }
      .fill-bad { height:100%; background:#dc2626; }
      .card { background: rgba(255,255,255,0.6); border:1px solid #e5e7eb; border-radius:12px; padding:12px; }
      .muted { color:#6b7280; }
      .badge { display:inline-block; padding:3px 8px; border-radius:999px; border:1px solid #e5e7eb; font-size:12px; }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# DATA / TABLES
# -------------------------------------------------------------
RUST_BELT = {"IL","MI","OH","WI","PA","NY","MN","IN","MA","NJ"}
SUN_BELT = {"FL","AZ","TX","NV","CA"}

DEPRECIATION_TABLE = {
    "MAZDA": -14, "HONDA": -13, "TOYOTA": -12, "BMW": -22,
    "FORD": -19, "CHEVROLET": -18, "TESLA": -9, "KIA": -17,
    "HYUNDAI": -16, "SUBARU": -14, "NISSAN": -17, "VOLKSWAGEN": -18,
    "JEEP": -21, "MERCEDES": -23, "AUDI": -22
}

INSURANCE_COST = {
    "MI": 2800, "FL": 2400, "NY": 2300, "OH": 1100,
    "TX": 1700, "CA": 1800, "AZ": 1400, "IL": 1500
}


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def clip(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(lo, min(hi, v))

def sanitize(text: str, max_len: int = CONFIG.MAX_INPUT_LENGTH) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[;\"']", "", text)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
        st.warning(f"Input truncated to {max_len} chars")
    return html.escape(cleaned)

def parse_json_safe(raw: str) -> Dict:
    if not raw:
        return {}
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            return json.loads(repair_json(cleaned))
        except Exception:
            return {}

def price_from_text(txt: str) -> Optional[float]:
    if not txt:
        return None
    t = re.sub(r"\s+", " ", txt)
    for pat in [
        r"(?i)\$?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\s*usd)?",
        r"(?i)price[\s:]*\$?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})",
        r"(?i)asking[\s:]*\$?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})",
    ]:
        m = re.search(pat, t)
        if m:
            try:
                p = float(m.group(1).replace(",", ""))
                if 0 <= p <= CONFIG.MAX_PRICE_VALUE:
                    return p
            except Exception:
                pass
    return None

def token_set(text: str) -> set:
    if not text:
        return set()
    t = re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())
    return {w for w in t.split() if len(w) > 2}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

@st.cache_data(ttl=CONFIG.CACHE_TTL)
def load_history() -> List[Dict]:
    if not os.path.exists(CONFIG.LOCAL_FILE):
        return []
    try:
        with open(CONFIG.LOCAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(entry: Dict):
    try:
        data = load_history()
        data.append(entry)
        if len(data) > CONFIG.MEMORY_LIMIT:
            data = data[-CONFIG.MEMORY_LIMIT:]
        with open(CONFIG.LOCAL_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Error saving history: {e}")

def unique_ad_id(ad_text: str, vin: str, zip_or_state: str, price_guess: float, seller: str) -> str:
    base = (vin or "").strip().upper() or f"{(ad_text or '')[:160]}|{price_guess}|{zip_or_state}|{seller}".lower()
    return hashlib.md5(base.encode()).hexdigest()[:12]


# -------------------------------------------------------------
# GEMINI CLIENT (optional)
# -------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")

    def call(self, parts: List[Dict], timeout: int = CONFIG.REQUEST_TIMEOUT):
        for attempt in range(CONFIG.MAX_RETRIES):
            try:
                return self.model.generate_content(parts, request_options={"timeout": timeout})
            except Exception as e:
                if attempt == CONFIG.MAX_RETRIES - 1:
                    st.error(f"Gemini failed after {CONFIG.MAX_RETRIES} tries: {e}")
                    return None
                time.sleep(2 ** attempt)


# -------------------------------------------------------------
# SCORE & ROI ENGINE
# -------------------------------------------------------------
class ScoreEngine:
    """Deterministic scoring with ROI forecast and risk tier."""

    @staticmethod
    def market_component(ask: float, median: float) -> Tuple[float, Dict]:
        if not ask or not median:
            return 50.0, {"gap_pct": None, "median": median}
        gap = (ask - median) / median * 100.0
        # Map gap to 0..100 (under median => higher score)
        if gap <= -30: score = 95
        elif gap <= -20: score = 88
        elif gap <= -10: score = 78
        elif gap <= 0: score = 70
        elif gap <= 5: score = 60
        elif gap <= 10: score = 52
        elif gap <= 20: score = 45
        else: score = 35
        return float(score), {"gap_pct": round(gap, 1), "median": median}

    @staticmethod
    def mileage_component(miles: Optional[int], seg_avg: int = 12_000) -> float:
        if miles is None: 
            return 55.0
        ratio = miles / max(1, seg_avg*5)  # 5-year window
        if ratio <= 0.6: return 85.0
        if ratio <= 0.9: return 75.0
        if ratio <= 1.1: return 68.0
        if ratio <= 1.4: return 58.0
        if ratio <= 1.8: return 48.0
        return 38.0

    @staticmethod
    def reliability_component(brand: str, yr: int, warranty_left: bool) -> float:
        brand = (brand or "").upper()
        base = 60.0
        if brand in {"TOYOTA", "HONDA", "MAZDA", "SUBARU"}:
            base = 78.0
        elif brand in {"KIA", "HYUNDAI", "FORD", "CHEVROLET", "VOLKSWAGEN", "NISSAN"}:
            base = 62.0
        elif brand in {"BMW", "MERCEDES", "AUDI", "TESLA", "JEEP"}:
            base = 58.0
        # age penalty
        age = max(0, datetime.now().year - int(yr or datetime.now().year))
        base -= min(20, age * 1.2)
        if not warranty_left:
            base -= 5
        return clip(base, 20, 95)

    @staticmethod
    def tco_component(state: str, mpg: float, maint_index: float) -> float:
        # Insurance by state, fuel by mpg, maintenance by index (1.0 = avg)
        ins = INSURANCE_COST.get((state or "").upper(), 1500)
        fuel = 2200 * (30.0 / max(10.0, float(mpg or 30.0)))  # normalize to 30mpg
        maint = 800 * max(0.5, float(maint_index or 1.0))
        # Lower cost => higher score
        total = ins + fuel + maint
        # Scale to 0..100
        # 2500 => ~80, 3500 => ~60, 4500 => ~45, 6000+ => ~30
        if total <= 2500: s = 85
        elif total <= 3000: s = 78
        elif total <= 3500: s = 66
        elif total <= 4500: s = 52
        elif total <= 6000: s = 40
        else: s = 30
        return float(s)

    @staticmethod
    def buyer_fit_component(powertrain: str, use_case: str) -> float:
        pt = (powertrain or "").lower()
        uc = (use_case or "").lower()
        score = 60.0
        if uc in {"commute", "uber", "lyft"}:
            score += 10 if pt in {"hybrid", "phev", "ev"} else 0
        if uc in {"family", "kids"}:
            score += 8 if pt in {"na", "hybrid", "phev"} else 0
        if uc in {"performance"}:
            score += 10 if pt in {"turbo", "v6", "v8"} else -10
        return clip(score, 30, 92)

    @staticmethod
    def risk_tier(components: Dict[str, float], title_status: str) -> str:
        low = sum(1 for k,v in components.items() if v >= 70)
        high_risk = any(k in {"title","reliability","tco"} and v < 50 for k,v in components.items())
        branded = str(title_status or "").lower() in {"salvage","rebuilt","flood","lemon","branded"}
        if branded: return "High"
        if high_risk: return "Medium"
        return "Low" if low >= 3 else "Medium"

    @staticmethod
    def roi_forecast(ask: float, brand: str, state: str) -> Dict[str, float]:
        dep = DEPRECIATION_TABLE.get((brand or "").upper(), -17)
        rb_pen = -3 if state.upper() in RUST_BELT else 0
        sb_bonus = +1 if state.upper() in SUN_BELT else 0
        base = dep + rb_pen + sb_bonus
        roi12 = base / 2.2
        roi24 = base
        roi36 = base * 1.6
        return {
            "12m": round(roi12, 1),
            "24m": round(roi24, 1),
            "36m": round(roi36, 1)
        }

    @staticmethod
    def deal_score(weights: Dict[str, float], comps: Dict[str, float]) -> float:
        # Weighted sum with caps
        total_w = sum(weights.values()) or 1.0
        score = 0.0
        for k,w in weights.items():
            score += clip(comps.get(k, 50.0), 0, 100) * (w / total_w)
        # soften extremes
        return round(0.92*score + 8, 1)


# -------------------------------------------------------------
# EXPLANATION ENGINE
# -------------------------------------------------------------
def explain_component(name: str, score: float, note: str = "", ctx: Dict = None) -> str:
    s = clip(score, 0, 100)
    n = (note or "").strip()
    name_l = (name or "").lower().strip()
    ctx = ctx or {}

    if s >= 90: level = "excellent"
    elif s >= 80: level = "very good"
    elif s >= 70: level = "good"
    elif s >= 60: level = "adequate"
    elif s >= 50: level = "below average"
    elif s >= 40: level = "weak"
    else: level = "poor"

    base = ""
    if name_l == "market":
        gap = None
        try:
            gap = float((ctx.get("market_refs") or {}).get("gap_pct", 0))
        except Exception:
            pass
        if gap is not None:
            if gap <= -20:
                base = f"Asking price ~{abs(int(gap))}% under U.S. clean-title median; {level} value."
            elif gap <= -10:
                base = f"Asking price moderately below U.S. market (~{abs(int(gap))}%); {level} value."
            elif gap < 5:
                base = f"Asking price aligns with U.S. median; {level} value."
            else:
                base = f"Asking price ~{int(gap)}% over U.S. median; {level} value."
        else:
            base = f"Price vs U.S. comps is {level}."
    elif name_l == "title":
        ts = str(((ctx.get("vehicle_facts") or {}).get("title_status", "unknown"))).lower()
        if ts in {"rebuilt","salvage","branded","flood","lemon"}:
            base = "Branded title — resale & insurance limited; extra due diligence required."
        elif ts == "clean":
            base = "Clean title — typical U.S. insurability & resale."
        else:
            base = "Title not confirmed; verify with DMV/Carfax."
    elif name_l == "mileage":
        base = f"Mileage condition is {level}; U.S. highway-heavy use softens penalty."
    elif name_l == "reliability":
        base = f"Long-term dependability is {level}; U.S. owner-reported issues within segment norms."
    elif name_l == "maintenance":
        base = f"Estimated annual maintenance is {level}; based on U.S. data (RepairPal/YourMechanic)."
    elif name_l == "tco":
        base = f"TCO (fuel/insurance/repairs) is {level} vs U.S. averages."
    else:
        base = f"{name.title()} factor is {level} relative to segment norms."

    if n:
        base += f" {n}"
    return base


def explanation_block(anchors: Dict[str, Any], comps: Dict[str, float], deal_score: float, risk: str, roi: Dict[str, float], facts: Dict[str, Any]) -> str:
    """Compose a mandatory, contradiction-free explanation referencing U.S. anchors."""
    kbb = anchors.get("KBB_median")
    ed = anchors.get("Edmunds_median")
    rp = anchors.get("RepairPal_index")
    iihs = anchors.get("IIHS_rating")

    lines = [
        f"• Market alignment: Compared with KBB/Edmunds medians ({kbb}/{ed}), pricing is {anchors.get('gap_pct','?')}% off median; Market component={int(comps.get('market',0))}.",
        f"• Ownership costs: RepairPal index={rp}; TCO component={int(comps.get('tco',0))}.",
        f"• Safety & reliability: IIHS={iihs or 'n/a'}; Reliability component={int(comps.get('reliability',0))}.",
        f"• Title & mileage: Title={facts.get('title_status','unknown')}; Mileage component={int(comps.get('mileage',0))}.",
        f"• Overall: Deal Score={deal_score}/100 (Risk tier={risk}). ROI forecast: 12m {roi['12m']}%, 24m {roi['24m']}%, 36m {roi['36m']}%.",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------
# SHEETS (optional)
# -------------------------------------------------------------
def try_open_sheet():
    if not SHEETS_AVAILABLE:
        return None
    try:
        secrets = st.secrets.get("connections", {}) or st.secrets
        json_key = secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON") or secrets.get("gcp_service_json")
        sheet_id = secrets.get("GOOGLE_SHEET_ID")
        if not (json_key and sheet_id):
            return None
        creds = Credentials.from_service_account_info(json.loads(json_key), scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.sheet1
        return ws
    except Exception as e:
        st.info(f"Sheets not configured: {e}")
        return None

def append_row_to_sheet(ws, row: List[Any]):
    try:
        if ws is None: 
            return
        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Sheets append failed: {e}")


# -------------------------------------------------------------
# UI WIDGETS
# -------------------------------------------------------------
def meter(label: str, value: float, suffix: str = ""):
    v = clip(value, 0, 100)
    css = "fill-ok" if v >= 70 else ("fill-warn" if v >= 40 else "fill-bad")
    st.markdown(f"<div class='metric'><b>{html.escape(label)}</b><span>{int(v)}{html.escape(suffix)}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='progress'><div class='{css}' style='width:{v}%' /></div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------
def main():
    inject_auto_theme()
    st.title(f"🚗 AI Deal Checker – U.S. Edition (Pro) v{CONFIG.APP_VERSION}")

    with st.expander("Paste listing text / inputs"):
        ad_text = st.text_area("Listing text", height=180, placeholder="Paste marketplace/craigslist text...")
        colA, colB, colC = st.columns(3)
        with colA:
            vin = st.text_input("VIN (optional)")
            brand = st.text_input("Brand", placeholder="e.g., Toyota")
            model = st.text_input("Model", placeholder="e.g., Camry")
            year = st.number_input("Year", min_value=1995, max_value=datetime.now().year+1, value=2018, step=1)
        with colB:
            state = st.text_input("State (e.g., OH/CA/FL)", max_chars=2)
            mileage = st.number_input("Mileage (mi)", min_value=0, value=75000, step=500)
            powertrain = st.selectbox("Powertrain", ["na","turbo","v6","v8","hybrid","phev","ev"], index=0)
            mpg = st.number_input("MPG (EPA combined)", min_value=8.0, max_value=120.0, value=30.0, step=0.5)
        with colC:
            use_case = st.selectbox("Buyer use-case", ["commute","family","performance","uber","lyft","generic"], index=0)
            seller = st.selectbox("Seller type", ["private","dealer"], index=0)
            zip_or_state = st.text_input("ZIP / State", placeholder="44105 or OH")
            ask_price = st.number_input("Ask price (USD)", min_value=0, max_value=CONFIG.MAX_PRICE_VALUE, value=int(price_from_text(ad_text) or 0), step=100)

    # Anchors (could be fetched by web; here manual / heuristic inputs)
    with st.expander("Market anchors (KBB/Edmunds/RepairPal/IIHS) – override optional"):
        col1, col2, col3, col4 = st.columns(4)
        with col1: kbb_median = st.number_input("KBB median ($)", min_value=0, value=ask_price or 18000, step=100)
        with col2: edmunds_median = st.number_input("Edmunds median ($)", min_value=0, value=ask_price or 18500, step=100)
        with col3: repairpal_index = st.slider("RepairPal index (0.5–1.5)", 0.5, 1.5, 1.0, 0.05)
        with col4: iihs_rating = st.selectbox("IIHS rating", ["G","A","M","P","n/a"], index=0)

    run = st.button("Analyze")

    if not run:
        st.info("Fill fields and click **Analyze**.")
        return

    # Compute components
    market_score, market_refs = ScoreEngine.market_component(ask_price, (kbb_median + edmunds_median) / 2 if (kbb_median and edmunds_median) else None)
    mileage_score = ScoreEngine.mileage_component(mileage)
    # Warranty heuristic: 5y/60k
    warranty_left = (datetime.now().year - int(year)) <= 5 and int(mileage) <= 60_000
    reliability_score = ScoreEngine.reliability_component(brand, int(year), warranty_left=warranty_left)
    tco_score = ScoreEngine.tco_component(state, mpg, maint_index=repairpal_index)
    fit_score = ScoreEngine.buyer_fit_component(powertrain, use_case)

    components = {
        "market": market_score,
        "mileage": mileage_score,
        "reliability": reliability_score,
        "tco": tco_score,
        "fit": fit_score,
        "title": 70.0  # default; refined if user provides
    }