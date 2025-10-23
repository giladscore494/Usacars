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
            base = f"TCO (fuel/insurance/repairs) is {level} vs U.S. 