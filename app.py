# -*- coding: utf-8 -*-
# ===========================================================
# 🚗 AI Deal Checker - U.S. Edition (Pro) v10.3.0
# Full v2.0 Analyst Spec | ROI 12/24/36m | Risk Tier | Buyer Fit | Compliance
# Auto Theme (Android + iOS Safari) | Warranty-aware Reliability | Mandatory Detailed Explanation
# Gemini 2.5 Pro | Sheets Integration | Insurance & Depreciation Tables
# ===========================================================

import os
import json
import re
import hashlib
import time
import html
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import streamlit as st
from json_repair import repair_json

# Optional Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    SHEETS_AVAILABLE = True
except ImportError:
    gspread = None
    Credentials = None
    SHEETS_AVAILABLE = False

# Google Generative AI (Gemini)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# -------------------------------------------------------------
# CONFIG & CONSTANTS
# -------------------------------------------------------------
class AppConfig:
    APP_VERSION = "10.3.0"
    LOCAL_FILE = "deal_history_us.json"
    MEMORY_LIMIT = 600
    MAX_INPUT_LENGTH = 10000
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 180
    CACHE_TTL = 300
    
    # Security settings
    MAX_PRICE_VALUE = 1000000
    MAX_COMPONENT_SCORE = 100
    MIN_COMPONENT_SCORE = 0

CONFIG = AppConfig()
st.set_page_config(page_title="AI Deal Checker", page_icon="🚗", layout="centered")

# -------------------------------------------------------------
# SECURITY & VALIDATION
# -------------------------------------------------------------
class SecurityHelper:
    @staticmethod
    def sanitize_input(text: str, max_length: int = CONFIG.MAX_INPUT_LENGTH) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not text:
            return ""
        
        # Remove potentially dangerous patterns
        cleaned = re.sub(r'[;\"\']', '', text)
        
        # Limit length
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
            st.warning(f"Input truncated to {max_length} characters")
        
        return html.escape(cleaned)
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Validate price is within reasonable bounds"""
        return 0 <= price <= CONFIG.MAX_PRICE_VALUE
    
    @staticmethod
    def validate_component_score(score: float) -> bool:
        """Validate component score is within bounds"""
        return CONFIG.MIN_COMPONENT_SCORE <= score <= CONFIG.MAX_COMPONENT_SCORE

# -------------------------------------------------------------
# THEME MANAGEMENT
# -------------------------------------------------------------
class ThemeManager:
    @staticmethod
    def inject_auto_theme():
        """Inject CSS for automatic light/dark theme"""
        st.markdown("""
        <style>
        :root { color-scheme: light dark; }

        /* Default (Light) */
        :root {
          --bg: #ffffff;
          --fg: #0f172a;
          --card: #ffffff;
          --border: #e5e7eb;
          --muted: #6b7280;
          --track: #e5e7eb55;
          --ok: #16a34a;
          --warn: #f59e0b;
          --bad: #dc2626;
          --chip: #eef2ff22;
        }

        /* Dark by system preference */
        @media (prefers-color-scheme: dark) {
          :root {
            --bg: #0b0f14;
            --fg: #e9eef2;
            --card: #11161c;
            --border: #1f2a37;
            --muted: #9aa4b2;
            --track: #33415588;
          }
          img, video, canvas, svg { 
            filter: none !important; 
            mix-blend-mode: normal !important; 
          }
        }

        html, body { background: var(--bg) !important; }
        body, .stMarkdown, .stText, p, label, div, span, code, h1, h2, h3, h4, h5, h6 {
          color: var(--fg) !important;
          -webkit-text-stroke: 0 transparent; 
          text-shadow: none;
        }

        .card { 
            background: var(--card); 
            border: 1px solid var(--border); 
            border-radius: 12px; 
            padding: 12px; 
        }
        .section { margin-top: 12px; }
        .metric { 
            display: flex; 
            align-items: center; 
            justify-content: space-between; 
            margin: 6px 0; 
            font-size: 0.95rem; 
        }
        .progress { 
            height: 10px; 
            background: var(--track); 
            border-radius: 6px; 
            overflow: hidden; 
        }
        .fill-ok { background: var(--ok); height: 100%; }
        .fill-warn { background: var(--warn); height: 100%; }
        .fill-bad { background: var(--bad); height: 100%; }
        small.muted { color: var(--muted); }
        hr { 
            border: none; 
            border-top: 1px solid var(--border); 
            margin: 18px 0; 
        }
        .expl { 
            font-size: 0.98rem; 
            line-height: 1.4; 
        }
        .expl p { margin: 6px 0; }
        .badge { 
            display: inline-block; 
            padding: 4px 8px; 
            border-radius: 999px; 
            font-size: 12px; 
            background: var(--chip); 
            border: 1px solid var(--border); 
        }
        .badge.warn { background: #fff7ed22; }
        .badge.err { background: #fee2e222; }
        .kpi { font-weight: 600; }
        .grid3 { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 10px; 
        }
        .grid2 { 
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 10px; 
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <script>
        (function(){
          try {
            var head = document.getElementsByTagName('head')[0];
            var m1 = document.createElement('meta');
            m1.name = 'color-scheme';
            m1.content = 'light dark';
            head.appendChild(m1);
            
            var m2 = document.createElement('meta');
            m2.name = 'theme-color';
            m2.content = '#ffffff';
            m2.media = '(prefers-color-scheme: light)';
            head.appendChild(m2);
            
            var m3 = document.createElement('meta');
            m3.name = 'theme-color';
            m3.content = '#0b0f14';
            m3.media = '(prefers-color-scheme: dark)';
            head.appendChild(m3);
            
            var fix = document.createElement('style');
            fix.innerHTML = '@supports (-webkit-touch-callout: none) { html,body{background:var(--bg)!important;color:var(--fg)!important;} img,video,canvas,svg{filter:none!important;mix-blend-mode:normal!important;} }';
            head.appendChild(fix);
          } catch(e) {}
        })();
        </script>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# U.S.-SPECIFIC DATA TABLES
# -------------------------------------------------------------
class USMarketData:
    RUST_BELT_STATES = {"IL", "MI", "OH", "WI", "PA", "NY", "MN", "IN", "MA", "NJ"}
    SUN_BELT_STATES = {"FL", "AZ", "TX", "NV", "CA"}
    
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
# DATA HELPERS
# -------------------------------------------------------------
class DataHelper:
    @staticmethod
    def meter(label: str, value: float, suffix: str = ""):
        """
        Display a metric with progress bar
        """
        try:
            v = float(value)
        except (ValueError, TypeError):
            v = 0
        
        v = max(0, min(100, v))
        css = "fill-ok" if v >= 70 else ("fill-warn" if v >= 40 else "fill-bad")
        
        st.markdown(
            f"<div class='metric'><b>{html.escape(str(label))}</b>"
            f"<span class='kpi'>{int(v)}{html.escape(str(suffix))}</span></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='progress'><div class='{css}' style='width:{v}%'></div></div>",
            unsafe_allow_html=True
        )
    
    @staticmethod
    def clip(x: Any, lo: float, hi: float) -> float:
        """
        Clip value between bounds
        """
        try:
            x = float(x)
        except (ValueError, TypeError):
            x = 0.0
        return max(lo, min(hi, x))
    
    @staticmethod
    def extract_price_from_text(txt: str) -> Optional[float]:
        """
        Extract price from text using regex patterns
        """
        if not txt:
            return None
        
        t = re.sub(r'\s+', ' ', txt)
        
        # Multiple price patterns
        patterns = [
            r'(?i)(?:\$?\s*)(\d{1,3}(?:,\d{3})+|\d{4,6})(?:\s*usd)?',
            r'(?i)price[\s:]*\$?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})',
            r'(?i)asking[\s:]*\$?\s*(\d{1,3}(?:,\d{3})+|\d{4,6})'
        ]
        
        for pattern in patterns:
            m = re.search(pattern, t)
            if m:
                try:
                    price = float(m.group(1).replace(',', ''))
                    if SecurityHelper.validate_price(price):
                        return price
                except (ValueError, TypeError):
                    continue
        
        return None
    
    @staticmethod
    def parse_json_safe(raw: str) -> Dict:
        """
        Safely parse JSON with repair fallback
        """
        if not raw:
            return {}
        
        cleaned = raw.replace('```json', '').replace('```', '').strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                return json.loads(repair_json(cleaned))
            except Exception:
                return {}
    
    @staticmethod
    def unique_ad_id(ad_text: str, vin: str, zip_or_state: str, price_guess: float, seller: str) -> str:
        """
        Generate unique ad ID for deduplication
        """
        base = (
            vin.strip().upper() 
            if vin 
            else f"{ad_text[:160]}|{price_guess}|{zip_or_state}|{seller}".lower()
        )
        return hashlib.md5(base.encode()).hexdigest()[:12]
    
    @staticmethod
    def token_set(text: str) -> set:
        """
        Create token set for similarity comparison
        """
        if not text:
            return set()
        
        t = re.sub(r'[^a-z0-9 ]+', ' ', str(text).lower())
        return {w for w in t.split() if len(w) > 2}
    
    @staticmethod
    def similarity_score(ad_a: Dict, ad_b: Dict) -> float:
        """
        Calculate similarity score between two ads
        """
        ta = DataHelper.token_set(ad_a.get("raw_text", ""))
        tb = DataHelper.token_set(ad_b.get("raw_text", ""))
        
        # Jaccard similarity
        if not ta and not tb:
            j = 0.0
        else:
            j = len(ta & tb) / max(1, len(ta | tb))
        
        # Price similarity
        p_a = float(ad_a.get("price_guess", 0) or 0)
        p_b = float(ad_b.get("price_guess", 0) or 0)
        price_sim = 1.0 - min(1.0, abs(p_a - p_b) / max(1000.0, max(p_a, p_b, 1.0)))
        
        # Location similarity
        loc_sim = 1.0 if (ad_a.get("zip_or_state") == ad_b.get("zip_or_state")) else 0.7
        
        return 0.6 * j + 0.3 * price_sim + 0.1 * loc_sim
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL)
    def load_history() -> List[Dict]:
        """
        Load deal history from local file
        """
        if not os.path.exists(CONFIG.LOCAL_FILE):
            return []
        
        try:
            with open(CONFIG.LOCAL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading history: {e}")
            return []
    
    @staticmethod
    def save_history(entry: Dict):
        """
        Save deal entry to history
        """
        try:
            data = DataHelper.load_history()
            data.append(entry)
            
            # Enforce memory limit
            if len(data) > CONFIG.MEMORY_LIMIT:
                data = data[-CONFIG.MEMORY_LIMIT:]
            
            with open(CONFIG.LOCAL_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            st.error(f"Error saving history: {e}")

# -------------------------------------------------------------
# API CLIENT WITH RETRY LOGIC
# -------------------------------------------------------------
class GeminiClient:
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not available")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
    
    def generate_content_with_retry(self, parts: List, max_retries: int = CONFIG.MAX_RETRIES) -> Any:
        """
        Generate content with exponential backoff retry logic
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    parts, 
                    request_options={"timeout": CONFIG.REQUEST_TIMEOUT}
                )
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"API failed after {max_retries} attempts: {e}")
                    return None
                
                wait_time = 2 ** attempt  # Exponential backoff
                st.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        return None

# -------------------------------------------------------------
# EXPLANATION ENGINE
# -------------------------------------------------------------
class ExplanationEngine:
    @staticmethod
    def _needs_explanation_fix(txt: str) -> bool:
        """
        Check if explanation needs repair
        """
        if not txt:
            return True
        
        t = txt.strip()
        bad_markers = [
            "Plain-English rationale summarizing",
            "Write the explanation here", 
            "DO NOT COPY ANY PLACEHOLDER",
            "Always avoid narrative/score contradictions",
        ]
        
        if any(m.lower() in t.lower() for m in bad_markers):
            return True
        
        if len(t) < 120:
            return True
        
        anchors = ["KBB", "Edmunds", "RepairPal", "iSeeCars", "NHTSA", "IIHS", "Autotrader", "Cars.com"]
        if sum(1 for a in anchors if a.lower() in t.lower()) < 2:
            return True
        
        return False
    
    @staticmethod
    def _repair_explanation(model, parsed: Dict) -> Optional[str]:
        """
        Repair poorly generated explanation
        """
        fields = {
            "from_ad": parsed.get("from_ad", {}),
            "ask_price_usd": parsed.get("ask_price_usd"),
            "vehicle_facts": parsed.get("vehicle_facts", {}),
            "market_refs": parsed.get("market_refs", {}),
            "deal_score": parsed.get("deal_score"),
            "components": parsed.get("components", []),
            "roi_forecast_24m": parsed.get("roi_forecast_24m", {}),
            "web_search_performed": parsed.get("web_search_performed", False),
            "roi_forecast": parsed.get("roi_forecast", {}),
            "risk_tier": parsed.get("risk_tier", ""),
        }
        
        repair_prompt = f"""
You failed to provide a proper score_explanation. Produce ONLY the explanation text.
Constraints:
- 120–400 words; 3–6 concise bullets or short paragraphs.
- Reference at least two U.S. anchors by name (e.g., KBB, Edmunds, RepairPal, iSeeCars, NHTSA, IIHS).
- Must align with the provided numbers (deal_score={fields.get('deal_score')}, market median/gap={fields.get('market_refs')}).
- Verify warranty status via manufacturer website; if warranty expired, lower reliability and raise failure-risk weighting accordingly.
- No placeholders, no instructions text, no JSON — just the explanation.

Context (immutable numbers):
{json.dumps(fields, ensure_ascii=False)}
"""
        try:
            response = model.generate_content(
                [{"text": repair_prompt}], 
                request_options={"timeout": 120}
            )
            txt = (getattr(response, "text", "") or "").strip().replace("```", "").strip()
            
            if ExplanationEngine._needs_explanation_fix(txt):
                return None
            return txt
            
        except Exception:
            return None
    
    @staticmethod
    def explain_component(name: str, score: float, note: str = "", ctx: Dict = None) -> str:
        """
        Generate human-readable component explanation
        """
        s = DataHelper.clip(score, 0, 100)
        n = (note or "").strip()
        name_l = (name or "").lower().strip()
        ctx = ctx or {}
        
        # Score level mapping
        if s >= 90:
            level = "excellent"
        elif s >= 80:
            level = "very good" 
        elif s >= 70:
            level = "good"
        elif s >= 60:
            level = "adequate"
        elif s >= 50:
            level = "below average"
        elif s >= 40:
            level = "weak"
        else:
            level = "poor"
        
        # Component-specific explanations
        base = ""
        if name_l == "market":
            gap = None
            try:
                gap = float((ctx.get("market_refs") or {}).get("gap_pct", 0))
            except (ValueError, TypeError):
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
            if ts in {"rebuilt", "salvage", "branded", "flood", "lemon"}:
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
            base = f"TCO (fuel/insurance/repairs) is {level} vs U.S. peers."
        elif name_l == "accidents":
            base = f"Accident risk is {level}; confirm Carfax/AutoCheck and repair documentation."
        elif name_l == "owners":
            base = f"Ownership history is {level}; fewer owners typically better in U.S. market."
        elif name_l == "rust":
            base = f"Rust/flood exposure is {level}; pay attention to Rust Belt/coastal operation."
        elif name_l == "demand":
            base = f"Buyer demand/DOM is {level}; may affect resale timing."
        elif name_l == "resale_value":
            base = f"Projected resale retention is {level} for this MY in U.S. market."
        else:
            base = f"{name.capitalize()} factor is {level}."
        
        # Brand-specific insights
        brand = str((ctx.get("from_ad") or {}).get("brand", "")).upper()
        if brand in {"TOYOTA", "HONDA", "MAZDA", "SUBARU"} and name_l in {"reliability", "resale_value"}:
            base += " Japanese-brand advantage recognized."
        if brand in {"FORD", "CHEVROLET", "JEEP"} and name_l in {"depreciation", "resale_value"}:
            base += " Verify 3-year depreciation trend for domestic brands."
        
        if n:
            return f"{name.capitalize()} — {int(s)}/100 → {base} ({n})"
        return f"{name.capitalize()} — {int(s)}/100 → {base}"
    
    @staticmethod
    def classify_deal(score: float) -> str:
        """
        Classify deal based on score
        """
        if score >= 80:
            return "✅ Good deal — price and condition align well with U.S. market value."
        if score >= 60:
            return "⚖️ Fair deal — acceptable, but verify title/history before proceeding."
        return "❌ Bad deal — overpriced or carries notable risk factors."

# -------------------------------------------------------------
# SHEETS INTEGRATION
# -------------------------------------------------------------
class SheetsManager:
    def __init__(self, sheet_id: str, service_json: Any):
        self.sheet_id = sheet_id
        self.service_json = service_json
        self.sheet = None
        self._connect()
    
    def _connect(self):
        """Connect to Google Sheets"""
        if not SHEETS_AVAILABLE or not self.sheet_id or not self.service_json:
            return
        
        try:
            if isinstance(self.service_json, str):
                self.service_json = json.loads(self.service_json)
            
            creds = Credentials.from_service_account_info(
                self.service_json,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            self.sheet = gspread.authorize(creds).open_by_key(self.sheet_id).sheet1
            st.toast("✅ Connected to Google Sheets")
            
        except Exception as e:
            st.warning(f"⚠️ Sheets connection failed: {e}")
    
    def append_deal(self, entry: Dict):
        """Append deal to Google Sheets"""
        if not self.sheet:
            return
        
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fa = entry.get("from_ad", {}) or {}
            roi = entry.get("roi_forecast_24m", {}) or {}
            gaps = entry.get("market_refs", {}) or {}
            uid = entry.get("unique_ad_id", "")
            
            self.sheet.append_row([
                ts,
                fa.get("brand", ""),
                fa.get("model", ""), 
                fa.get("year", ""),
                entry.get("deal_score", ""),
                roi.get("expected", ""),
                entry.get("web_search_performed", ""),
                entry.get("confidence_level", ""),
                gaps.get("median_clean", ""),
                gaps.get("gap_pct", ""),
                uid,
                fa.get("state_or_zip", ""),
            ], value_input_option="USER_ENTERED")
            
        except Exception as e:
            st.warning(f"Sheets write failed: {e}")

# -------------------------------------------------------------
# PROMPT BUILDER (FULL VERSION)
# -------------------------------------------------------------
class PromptBuilder:
    @staticmethod
    def build_prompt_us(ad: str, extra: str, must_id: str, exact_prev: Dict, similar_summ: List) -> str:
        """
        Build the full analysis prompt with all edge cases and business logic
        """
        exact_json = json.dumps(exact_prev or {}, ensure_ascii=False)
        similar_json = json.dumps(similar_summ or [], ensure_ascii=False)
        
        return f"""
You are a senior U.S. used-car analyst (2023–2025). Web reasoning is REQUIRED.

Stages:
1) Extract listing facts: ask_price_usd, brand, model, year, trim, powertrain, miles, title_status, owners, accidents,
   options_value_usd, state_or_zip, days_on_market (if present).
2) Do live U.S.-centric lookups (REQUIRED) for the exact year/model:
   - Market comps & CLEAN-title median: Cars.com, Autotrader, Edmunds, and KBB (Kelley Blue Book).
   - Reliability & common issues: Consumer Reports style + RepairPal.
   - Typical annual maintenance cost: RepairPal or YourMechanic (U.S. 2023–2025).
   - Depreciation trend (24–36m): CarEdge or iSeeCars.
   - Demand/DOM averages; brand/model resale retention (CarEdge/iSeeCars).
   - Safety/recalls context: NHTSA; insurance risk context: IIHS (qualitative).
   - Verify warranty status via manufacturer website; if warranty expired, lower reliability and raise failure-risk weighting accordingly and explain it in the reliability section.
   - Verify open recalls and TSBs via NHTSA/manufacturer; check lemon-law/buyback if VIN present.
   Consider U.S. realities (Rust Belt vs Sun Belt, dealer vs private, mileage normalization).

Use prior only for stabilization (do NOT overfit):
- exact_prev (same listing id): weight ≤ 25% -> {exact_json}
- similar_previous (very similar ads): anchors only, weight ≤ 10% -> {similar_json}

Scoring rules for U.S. buyers (adjusted weights):
- Title condition (clean > rebuilt > salvage) ~20%; if 'rebuilt'/'salvage'/branded -> CAP deal_score ≤ 75.
- Price vs CLEAN-title median ~25%.
- Mileage impact ~10% (U.S. highway-heavy driving reduces penalty).
- Reliability & maintenance together ~20%.
- TCO (fuel + insurance + repairs) ~8% (U.S. costs).
- Accidents + owners ~9%.
- Rust/flood zone ~4% (Rust Belt/coastal exposure).
- Demand/resale ~4%.

Critical adjustment guidelines (U.S.-market realism):
Edge-case heuristic layer (20 scenarios — apply in addition to base weights):
1) OEM new engine → Reliability +25–35; Market +15; Resale +10.
2) Used/unknown-provenance engine → ≤ +5; add caution flag ("verify installation origin").
3) OEM new transmission → Reliability +15; Market +10.
4) Rebuilt / Salvage / Branded title → cap deal_score ≤ 75; ROI_expected −5.
5) Carfax "minor damage" → −5 reliability; −5 resale (acceptable if repaired).
6) Structural damage / airbag deployed → set ceiling ≤ 55 overall; strong warning.
7) Repainted panels / full repaint → −5 market; −5 resale.
8) Clean Carfax + 1 owner + dealer maintained → +10 reliability; +10 resale.
9) High-insurance states (MI, NY, NJ, FL) → −5 TCO; mention insurance context.
10) Sun Belt (FL, AZ, CA, TX, NV) → +5 rust; −2 interior (sun wear) if hinted.
11) Rust Belt origin/operation → −10 rust; add underbody inspection warning.
12) Suspiciously low miles for age with no documentation → −10 reliability until explained.
13) Fleet/Rental history → −10 reliability; −10 resale.
14) Private owner + full service records → +10 reliability; +5 resale.
15) High-performance trims (AMG/M/M S-line/Hellcat) → +10 demand/market; −5 TCO (insurance).
16) Extensive aftermarket mods/tuning → −10 resale; −5 reliability (unless track-documented).
17) Canada-import / grey market → −10 market; −10 resale; mention potential registration/insurance frictions.
18) Major recall fixed with proof → +5 reliability.
19) Hybrid/EV traction battery recently replaced → +20 reliability; +10 resale.

Extended risk and compliance logic:
• Always cross-check safety recalls via NHTSA and active TSBs; if open recalls found, reduce reliability −5 and include note.
• If VIN indicates manufacturer buyback or lemon law history → cap deal_score ≤ 65 and flag as "Lemon/Buyback risk".
• If odometer discrepancy or title mileage not actual → cap ≤ 60 and mention "Not actual mileage".
• If warranty expired (<5yr/60k mi for mainstream, <4yr/50k for luxury) → −10 reliability, −5 resale; explain in reliability rationale.
• If factory powertrain warranty active → +10 reliability confidence.
• If Carfax shows "commercial use" (fleet, rental, ride-share) → −10 reliability, −10 resale; warn about heavy wear.
• If service records show all-dealer maintenance → +10 reliability, +5 resale.
• If listing location is in flood-prone ZIP (Louisiana, Florida coastal, Texas Gulf) → −8 rust/flood; mention flood risk explicitly.
• If ad mentions "as-is sale", "no warranty", or "mechanic special" → reduce confidence and market valuation significantly (−10 to −20 combined).
• If luxury performance (AMG/M/Hellcat) and tire/maintenance costs > $2k/year → −10 TCO, note high cost-of-ownership.
• If EV with degraded or replaced battery (verified via manufacturer) → adjust reliability ±20 depending on replacement status.
• If EV range <80% of original → cap deal_score ≤ 70 and mention "battery degradation".
• Always ensure numeric consistency: explanation text must never contradict any component score.
• End each explanation with a short ROI summary: expected return (12/24/36m) and key U.S. buyer takeaway.

20) "As-is" sale with no warranty → −10 confidence; −10 resale; emphasize PPI.
• If listing text mentions any of these keywords:
["new engine", "engine replaced", "factory engine replaced", "rebuilt transmission", "new transmission", "engine under warranty", "factory rebuild", "powertrain warranty", "short block replaced"]
→ Apply a strong positive adjustment:
- Reliability: +25 to +35 points
- Mileage: +20 points
- Market: +15 points
- Resale_value: +10 points
- Add explicit note in "score_explanation" about OEM/warranty-grade replacement.
• If replacement keywords appear without "OEM", "warranty", or "dealer-installed":
→ Moderate/neutral (+10–15 total) and flag provenance uncertainty.
• Align numeric component scores with narrative (no contradictions).

Explanation contract (MANDATORY):
- Return a specific, human-readable explanation tying PRICE vs CLEAN median, TITLE, MILEAGE, RELIABILITY/MAINTENANCE (with U.S. sources), warranty status, and ROI.
- 120–400 words, 3–6 bullets/short paragraphs.
- Mention at least two anchors by name (KBB, Edmunds, RepairPal, iSeeCars, etc.).
- DO NOT copy any instruction text or placeholders.

Output STRICT JSON only:
{{
    "from_ad": {{"brand":"","model":"","year":null,"vin":"","seller_type":""}},
    "ask_price_usd": 0,
    "vehicle_facts": {{
        "title_status":"unknown","accidents":0,"owners":1,"dealer_reputation":null,
        "rarity_index":0,"options_value_usd":0,"days_on_market":0,"state_or_zip":"","miles":null
    }},
    "market_refs": {{"median_clean":0,"gap_pct":0}},
    "web_search_performed": true,
    "confidence_level": 0.75,
    "components": [
        {{"name":"market","score":0,"note":""}},
        {{"name":"title","score":0,"note":""}},
        {{"name":"mileage","score":0,"note":""}},
        {{"name":"reliability","score":0,"note":""}},
        {{"name":"maintenance","score":0,"note":""}},
        {{"name":"tco","score":0,"note":""}},
        {{"name":"accidents","score":0,"note":""}},
        {{"name":"owners","score":0,"note":""}},
        {{"name":"rust","score":0,"note":""}},
        {{"name":"demand","score":0,"note":""}},
        {{"name":"resale_value","score":0,"note":""}}
    ],
    "deal_score": 0,
    "roi_forecast_24m": {{"expected":0,"optimistic":0,"pessimistic":0}},
    "roi_forecast": {{"12m":0,"24m":0,"36m":0}},
    "risk_tier": "Tier 2 (average-risk)",
    "relative_rank": "",
    "buyer_fit": "",
    "verification_summary": "",
    "benchmark": {{"segment":"","rivals":[]}},
    "score_explanation": "<<WRITE DETAILED EXPLANATION — NO PLACEHOLDERS>>",
    "listing_id_used": "{must_id}"
}}

LISTING (title + description):
\"\"\"{ad}\"\"\"

Extra:
{extra}

Hard constraints:
- Always perform web lookups and set web_search_performed=true; if not possible, list which sources failed but still estimate.
- Numeric fields must be numbers. deal_score: 0..100. ROI parts: -50..50.
- Per-component short notes required.
- If title_status is 'rebuilt', 'salvage' or any branded title: CAP deal_score ≤ 75 and clearly warn in score_explanation.
- If market gap (gap_pct) ≤ -35: warn to verify insurance/accident history before purchase.
- Enforce alignment between narrative and component scores (no contradictions).
"""

# -------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------
class AIDealCheckerApp:
    def __init__(self):
        self.config = CONFIG
        self.security = SecurityHelper()
        self.theme_manager = ThemeManager()
        self.data_helper = DataHelper()
        self.explanation_engine = ExplanationEngine()
        self.us_market_data = USMarketData()
        self.prompt_builder = PromptBuilder()
        
        # Initialize components
        self._setup_api()
        self._setup_sheets()
        self._setup_ui()
    
    def _setup_api(self):
        """Setup API client"""
        API_KEY = st.secrets.get("GEMINI_API_KEY", "")
        if not API_KEY:
            st.error("Missing GEMINI_API_KEY in Streamlit secrets.")
            st.stop()
        
        if not GEMINI_AVAILABLE:
            st.error("Google Generative AI package not available.")
            st.stop()
        
        self.gemini_client = GeminiClient(API_KEY)
    
    def _setup_sheets(self):
        """Setup Google Sheets integration"""
        SHEET_ID = st.secrets.get("GOOGLE_SHEET_ID", "")
        SERVICE_JSON = st.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", None)
        self.sheets_manager = SheetsManager(SHEET_ID, SERVICE_JSON)
    
    def _setup_ui(self):
        """Setup user interface"""
        self.theme_manager.inject_auto_theme()
        st.title("🚗 AI Deal Checker")
        st.caption(f"U.S. Edition (Pro) v{CONFIG.APP_VERSION} | Auto Theme • KBB/Edmunds/RepairPal/iSeeCars anchors • Insurance & Depreciation • ROI Forecasting (Gemini 2.5 Pro)")
    
    def build_extra_context(self, vin: str, zip_code: str, seller: str, imgs: List) -> str:
        """Build extra context string"""
        extra = ""
        if vin:
            extra += f"\nVIN: {vin}"
        if zip_code:
            extra += f"\nZIP/State: {zip_code}"
        if seller:
            extra += f"\nSeller: {seller}"
        if imgs:
            extra += f"\nPhotos provided: {len(imgs)} file(s) (content parsed by model if supported)."
        return extra

    def run_analysis(self, ad: str, vin: str, zip_code: str, seller: str, imgs: List):
        """Run complete analysis pipeline"""
        # Input validation
        ad = self.security.sanitize_input(ad)
        if not ad.strip():
            st.error("Please paste listing text first.")
            return
        
        # Build context and prepare analysis
        extra = self.build_extra_context(vin, zip_code, seller, imgs)
        price_guess = self.data_helper.extract_price_from_text(ad) or 0
        must_id = self.data_helper.unique_ad_id(ad, vin, zip_code, price_guess, seller)
        
        # Load and analyze history
        history = self.data_helper.load_history()
        exact_prev = next((h for h in history if h.get("unique_ad_id") == must_id), None)
        
        # Find similar ads
        current_struct = {"raw_text": ad, "price_guess": price_guess, "zip_or_state": zip_code or ""}
        sims = []
        
        for h in history:
            prior_struct = {
                "raw_text": h.get("raw_text") or "",
                "price_guess": self.data_helper.extract_price_from_text(h.get("raw_text") or "") or 0,
                "zip_or_state": (h.get("from_ad") or {}).get("state_or_zip", ""),
            }
            s = self.data_helper.similarity_score(current_struct, prior_struct)
            if s >= 0.85 and h.get("unique_ad_id") != must_id:
                sims.append({
                    "id": h.get("unique_ad_id"),
                    "score": h.get("deal_score"), 
                    "when": h.get("timestamp", ""),
                    "sim": round(s, 3)
                })
        
        sims = sorted(sims, key=lambda x: -x["sim"])[:5]
        
        # Prepare API call with full prompt
        parts = [{"text": self.prompt_builder.build_prompt_us(ad, extra, must_id, exact_prev or {}, sims)}]
        
        # Add images if available
        for img in imgs or []:
            try:
                mime = "image/png" if "png" in img.type.lower() else "image/jpeg"
                parts.append({"mime_type": mime, "data": img.read()})
            except Exception:
                pass
        
        # Execute analysis
        with st.spinner("Analyzing with Gemini 2.5 Pro (U.S. web reasoning)..."):
            response = self.gemini_client.generate_content_with_retry(parts)
            
            if not response:
                st.error("Analysis failed. Please try again.")
                return
            
            data = self.data_helper.parse_json_safe(getattr(response, "text", None))
            if not data:
                st.error("Failed to parse analysis results.")
                return
        
        # Process and display results
        self._display_results(data, must_id, ad, exact_prev, sims, price_guess, zip_code)
    
    def _display_results(self, data: Dict, must_id: str, ad: str, exact_prev: Dict, 
                        sims: List, price_guess: float, zip_code: str):
        """Display analysis results"""
        # Process base score and ROI
        base_score = self.data_helper.clip(data.get("deal_score", 60), 0, 100)
        roi24 = data.get("roi_forecast_24m", {}) or {}
        for k in ["expected", "optimistic", "pessimistic"]:
            roi24[k] = self.data_helper.clip(roi24.get(k, 0), -50, 50)
        
        roi_triple = data.get("roi_forecast", {}) or {}
        for k in ["12m", "24m", "36m"]:
            roi_triple[k] = self.data_helper.clip(roi_triple.get(k, 0), -50, 50)
        
        facts = data.get("vehicle_facts", {}) or {}
        title_status = str(facts.get("title_status", "unknown")).strip().lower()
        market_refs = data.get("market_refs", {}) or {}
        gap_pct = float(market_refs.get("gap_pct", 0)) if market_refs.get("gap_pct") is not None else 0.0
        
        # Memory stabilization (score + ROI expected)
        final_score = base_score
        if exact_prev and sims:
            similar_avg = None
            vals = [v["score"] for v in sims if isinstance(v.get("score"), (int, float))]
            if vals:
                similar_avg = round(sum(vals) / len(vals), 2)
            if similar_avg is not None:
                final_score = round(0.80 * base_score + 0.15 * float(exact_prev.get("deal_score", base_score)) + 0.05 * similar_avg, 1)
        elif exact_prev:
            final_score = round(0.75 * base_score + 0.25 * float(exact_prev.get("deal_score", base_score)), 1)
        
        # Strict rebuilt/salvage handling
        warnings_ui = []
        branded = title_status in {"rebuilt", "salvage", "branded", "flood", "lemon"}
        if branded:
            final_score = min(75.0, final_score - 5.0)
            roi24["expected"] = round(roi24.get("expected", 0) - 5.0, 1)
            warnings_ui.append("Branded/salvage title detected — insurers and lenders may limit options; resale harder.")
        
        # Rust belt / insurance context adjustments
        state_or_zip = (facts.get("state_or_zip") or "").strip().upper()
        state_code = ""
        if re.fullmatch(r"[A-Z]{2}", state_or_zip):
            state_code = state_or_zip
        elif re.fullmatch(r"\d{5}", state_or_zip):
            state_code = ""
        
        if state_code in USMarketData.RUST_BELT_STATES:
            final_score = round(final_score - 1.5, 1)
            warnings_ui.append("Rust Belt region — inspect underbody/brakes/lines for corrosion.")
        
        if state_code in USMarketData.INSURANCE_COST and USMarketData.INSURANCE_COST[state_code] >= 2000:
            warnings_ui.append(f"High average insurance cost in {state_code} — include in TCO.")
        
        # Confidence
        confidence = self.data_helper.clip(float(data.get("confidence_level", 0.7)) * 100, 0, 100)
        
        # Components → human text lines
        comp_lines = []
        components = data.get("components", []) or []
        ctx_for_exp = {"market_refs": market_refs, "vehicle_facts": facts, "from_ad": data.get("from_ad") or {}}
        for c in components:
            name = c.get("name", "")
            score = c.get("score", 0)
            note = c.get("note", "")
            try:
                comp_lines.append(self.explanation_engine.explain_component(name, score, note, ctx=ctx_for_exp))
            except Exception:
                comp_lines.append(f"{name.capitalize()} — {int(self.data_helper.clip(score, 0, 100))}/100")
        
        # Classification
        verdict = self.explanation_engine.classify_deal(final_score)
        
        # Explanation quality check + repair if needed
        raw_exp = data.get("score_explanation", "") or ""
        if self.explanation_engine._needs_explanation_fix(raw_exp):
            fixed = self.explanation_engine._repair_explanation(self.gemini_client.model, data)
            if fixed:
                data["score_explanation"] = fixed
                raw_exp = fixed
            else:
                raw_exp = "Model did not provide a sufficient rationale. Please ensure the listing includes year, trim, mileage, price, title, and location, then retry."
        
        # UI OUTPUT
        st.markdown("### Deal Score")
        self.data_helper.meter("Deal Score", final_score, "/100")
        st.markdown(f"<div><span class='badge'>{html.escape(verdict)}</span></div>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        with cols[0]:
            self.data_helper.meter("Confidence", confidence, "%")
        with cols[1]:
            try:
                ask = float(data.get("ask_price_usd", 0))
            except Exception:
                ask = 0.0
            st.markdown(f"**Asking price:** ${int(ask):,}")
            if market_refs.get("median_clean"):
                med = float(market_refs["median_clean"])
                st.markdown(f"**Clean-title median:** ${int(med):,}")
            st.markdown(f"**Market gap:** {gap_pct:+.0f}%")
        with cols[2]:
            brand = str((data.get("from_ad") or {}).get("brand", "")).upper()
            yr = (data.get("from_ad") or {}).get("year", "")
            model_name = (data.get("from_ad") or {}).get("model", "")
            st.markdown(f"**Vehicle:** {html.escape((brand or '---'))} {html.escape(str(model_name or ''))} {html.escape(str(yr or ''))}")
            st.markdown(f"**Title:** {html.escape(title_status or 'unknown')}")
            st.markdown(f"**Location:** {html.escape(state_or_zip or '---')}")
        
        # Score explanation
        score_exp = html.escape(raw_exp).replace("\n", "<br/>")
        if score_exp:
            st.markdown(f"<div class='section card'><b>Why this score?</b><br/>{score_exp}</div>", unsafe_allow_html=True)
        
        # ROI & Risk Tier
        st.markdown("### ROI Forecast (12/24/36 months) & Risk Tier")
        rcols = st.columns(3)
        with rcols[0]:
            st.metric("12m ROI", f"{roi_triple.get('12m', 0):+.1f}%")
        with rcols[1]:
            st.metric("24m ROI", f"{roi_triple.get('24m', 0):+.1f}%")
        with rcols[2]:
            st.metric("36m ROI", f"{roi_triple.get('36m', 0):+.1f}%")
        
        # Legacy 24-month triplet
        st.markdown("<div class='section'><small class='muted'>Legacy 24m forecast kept for compatibility</small></div>", unsafe_allow_html=True)
        r2 = st.columns(3)
        with r2[0]:
            st.metric("Expected (24m)", f"{roi24.get('expected', 0):+.1f}%")
        with r2[1]:
            st.metric("Optimistic (24m)", f"{roi24.get('optimistic', 0):+.1f}%")
        with r2[2]:
            st.metric("Pessimistic (24m)", f"{roi24.get('pessimistic', 0):+.1f}%")
        
        # Risk Tier & Buyer Fit
        rt = data.get("risk_tier", "").strip() or "Tier 2 (average-risk)"
        bf = data.get("buyer_fit", "").strip()
        rr = data.get("relative_rank", "").strip()
        verif = data.get("verification_summary", "").strip()
        
        st.markdown("<div class='section card'>", unsafe_allow_html=True)
        st.markdown(f"**Risk Tier:** {html.escape(rt)}", unsafe_allow_html=True)
        if rr:
            st.markdown(f"**Relative Rank:** {html.escape(rr)}", unsafe_allow_html=True)
        if bf:
            st.markdown(f"**Buyer Fit:** {html.escape(bf)}", unsafe_allow_html=True)
        if verif:
            st.markdown(f"**Compliance/Verification:** {html.escape(verif)}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Component breakdown
        if comp_lines:
            st.markdown("<div class='section'><b>Component breakdown</b></div>", unsafe_allow_html=True)
            safe_lines = [f"<p>• {html.escape(str(x))}</p>" for x in comp_lines]
            st.markdown("<div class='card expl'>" + "<br/>".join(safe_lines) + "</div>", unsafe_allow_html=True)
        else:
            st.info("No component breakdown available.")
        
        # Warnings block
        warnings_ui = list(dict.fromkeys(warnings_ui))
        if warnings_ui:
            warn_html = "".join([f"<li>{html.escape(w)}</li>" for w in warnings_ui])
            st.markdown(f"<div class='section card'><b>Warnings</b><ul>{warn_html}</ul></div>", unsafe_allow_html=True)
        
        # Web lookup badge
        web_done = bool(data.get("web_search_performed", False))
        st.markdown(
            f"<div class='section'>Web lookup: <span class='badge {'warn' if not web_done else ''}'>{'NOT performed (model fallback)' if not web_done else 'performed'}</span></div>",
            unsafe_allow_html=True
        )
        
        # Save history
        out_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "unique_ad_id": must_id,
            "raw_text": ad,
            "from_ad": data.get("from_ad", {}),
            "deal_score": final_score,
            "confidence_level": round(confidence / 100, 3),
            "market_refs": market_refs,
            "roi_forecast_24m": roi24,
            "roi_forecast": roi_triple,
            "risk_tier": rt,
            "relative_rank": rr,
            "buyer_fit": bf,
            "verification_summary": verif,
            "web_search_performed": web_done,
        }
        
        try:
            self.data_helper.save_history(out_entry)
            self.sheets_manager.append_deal(out_entry)
        except Exception as e:
            st.warning(f"Save failed: {e}")
        
        # Debug panels
        with st.expander("Debug JSON (model output)"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2))
        
        with st.expander("Similar previous (anchors ≤10%)"):
            if sims:
                st.json(sims)
            else:
                st.write("None")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.caption(f"AI Deal Checker — U.S. Edition (Pro) v{CONFIG.APP_VERSION} © 2025 | Gemini 2.5 Pro | Auto Theme Edition")

    def render_ui(self):
        """Render main user interface"""
        st.subheader("Paste the listing text:")
        
        ad = st.text_area(
            "",
            height=230,
            placeholder="Year • Make • Model • Trim • Mileage • Price • Title • Location • Options ...",
            key="ad_text_main",
        )
        
        imgs = st.file_uploader(
            "Upload photos (optional):",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="images_uploader",
        )
        
        c1, c2, c3 = st.columns(3)
        with c1:
            vin = st.text_input("VIN (optional)", key="vin_input")
        with c2:
            zip_code = st.text_input("ZIP / State (e.g., 44105 or OH)", key="zip_input")
        with c3:
            seller = st.selectbox("Seller type", ["", "private", "dealer"], key="seller_select")
        
        if st.button("Analyze Deal", use_container_width=True, type="primary", key="analyze_button"):
            self.run_analysis(ad, vin, zip_code, seller, imgs)

# -------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------
def main():
    """Main application entry point"""
    try:
        app = AIDealCheckerApp()
        app.render_ui()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()