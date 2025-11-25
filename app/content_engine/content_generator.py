# ============================================================
# content_generator3.py (UPDATED FULL VERSION)
# ============================================================

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# LLM Clients
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    import genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False

# Dynamic prompt builder
from .dynamic_prompt import generate_engaging_prompt

# Trend optimizer (real-time trends)
from app.integrations.trend_fetcher import TrendFetcher
from app.content_engine.trend_based_optimizer import TrendBasedOptimizer

# Sheets logging
from app.integrations.sheets_connector import append_row

# Optional tools
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

try:
    import language_tool_python
    LT_AVAILABLE = True
except Exception:
    LT_AVAILABLE = False


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


# ============================================================
# Local fallback
# ============================================================

def _local_generate(prompt: str, n: int = 3) -> List[str]:
    return [f"{prompt} — variant {i+1}" for i in range(n)]


# ============================================================
# LLM Call Helpers
# ============================================================

def _call_groq(prompt: str, model: str = None) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7"))
    )
    return resp.choices[0].message.content


def _call_gemini(prompt: str, model: str = None) -> str:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    gmodel = genai.GenerativeModel(model)
    response = gmodel.generate_content(prompt)

    return response.text if hasattr(response, "text") else str(response)


# ============================================================
# Main LLM generation function
# ============================================================

def generate_variations(prompt: str, n: int = 2) -> List[str]:
    """
    Generate n variations using the strongest available LLM:
      1. Groq → LLaMA 3.3
      2. Gemini → 1.5 Flash
      3. Local fallback
    """
    if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
        try:
            logger.info("Using Groq for content generation...")
            return [_call_groq(prompt) for _ in range(n)]
        except Exception:
            logger.exception("Groq failed → trying Gemini...")

    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
        try:
            logger.info("Using Gemini fallback...")
            return [_call_gemini(prompt) for _ in range(n)]
        except Exception:
            logger.exception("Gemini failed → using local fallback...")

    logger.warning("Using LOCAL fallback generator.")
    return _local_generate(prompt, n)


# ============================================================
# Quality scoring (readability + grammar)
# ============================================================

def score_quality(text: str) -> Dict:
    readability_score = None
    grammar_issues = None

    if TEXTSTAT_AVAILABLE:
        try:
            readability_score = textstat.flesch_reading_ease(text)
        except:
            readability_score = None

    if LT_AVAILABLE:
        try:
            tool = language_tool_python.LanguageTool('en-US')
            matches = tool.check(text)
            grammar_issues = len(matches)
        except:
            grammar_issues = None

    return {
        "readability_score": readability_score,
        "grammar_issues": grammar_issues
    }


# ============================================================
# Hashtag Cleaning Utilities
# ============================================================

def clean_punctuation_hashtags(text: str) -> str:
    words = text.split()
    cleaned = []
    for w in words:
        if w.startswith("#"):
            w = w.rstrip(",.?!;:")
        cleaned.append(w)
    return " ".join(cleaned)


def dedupe_hashtags(text: str):
    seen = set()
    out = []
    for w in text.split():
        if w.startswith("#"):
            if w.lower() not in seen:
                seen.add(w.lower())
                out.append(w)
        else:
            out.append(w)
    return " ".join(out)


def move_hashtags_to_end(text: str):
    words = text.split()
    tags = [w for w in words if w.startswith("#")]
    others = [w for w in words if not w.startswith("#")]
    return " ".join(others + tags)


def clean_and_order_hashtags(text: str):
    text = clean_punctuation_hashtags(text)
    text = dedupe_hashtags(text)
    text = move_hashtags_to_end(text)
    return text


# ============================================================
# Engagement-Aware Ranking
# ============================================================

def optimize_with_engagement(candidates: List[Dict], past_metrics: Optional[Dict] = None):
    top_keywords = []
    if past_metrics:
        top_keywords = list(past_metrics.get("top_keywords", []))[:3]

    scored = []

    for c in candidates:
        text = c.get("optimized_text", "")
        score = 0.0

        q = score_quality(text)
        if q["readability_score"] is not None:
            score += q["readability_score"] / 100
        if q["grammar_issues"] is not None:
            score -= min(1.0, 0.1 * q["grammar_issues"])

        # Keyword-boosting
        for kw in top_keywords:
            if kw.lower() in text.lower():
                score += 0.2

        c["engagement_score"] = score
        scored.append((score, c))

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_sorted]


# ============================================================
# FINAL PIPELINE — FULLY UPDATED
# ============================================================

def generate_final_variations(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    n: int = 2,
    word_count: int = 50,
    past_metrics: Optional[Dict] = None
) -> List[Dict]:

    # Normalize keywords
    if isinstance(keywords, str):
        keywords = keywords.split(",")

    # NEW: Retrieve REAL trending keywords
    tf = TrendFetcher()
    real_trends = tf.fetch_google_global_trends()

    injected_keywords = keywords + [t for t in real_trends if t not in keywords]

    # Build dynamic prompt with trend-aware keywords
    prompt = generate_engaging_prompt(
        topic,
        platform,
        injected_keywords,
        audience,
        tone,
        trends=real_trends,
        word_count=word_count
    )

    # Step 1: Generate raw content
    raw_variants = generate_variations(prompt, n=n)

    # Step 2: Trend-Based Optimization
    optimizer = TrendBasedOptimizer()

    optimized_candidates = []
    for text in raw_variants:
        opt = optimizer.run(text)

        # Hashtag cleanup
        cleaned = clean_and_order_hashtags(opt["optimized"])
        opt["optimized_text"] = cleaned

        optimized_candidates.append(opt)

    # Step 3: Engagement Ranking
    final_order = optimize_with_engagement(optimized_candidates, past_metrics)

    # Step 4: Build output objects
    results = []
    for item in final_order:
        optimized_text = item.get("optimized_text", "")

        results.append({
            "text": optimized_text,
            "quality": score_quality(optimized_text),
            "meta": {
                "topic": topic,
                "platform": platform,
                "audience": audience,
                "injected_keywords": injected_keywords,
                "trend_score": item.get("trend_score", 0),
                "trend_insights": item.get("insights", {}),
            }
        })

        # Log each generated variant to Sheets
        try:
            append_row("generated_content", [
                datetime.utcnow().isoformat(),
                platform,
                topic[:40] + "...",
                optimized_text[:80] + "...",
                item.get("trend_score", 0)
            ])
        except:
            pass

    return results


# ============================================================
# Test Run
# ============================================================

if __name__ == "__main__":
    out = generate_final_variations(
        "AI in Marketing",
        "Twitter",
        ["#AI", "#Marketing"],
        "marketers",
        "positive",
        n=2
    )

    for i, r in enumerate(out, 1):
        print(f"\nVariant {i}:")
        print("Text:", r["text"])
        print("Quality:", r["quality"])
        print("Meta:", r["meta"])
