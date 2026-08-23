"""
Improved dynamic_prompt.py

Upgrades added:
- Platform-specific content rules (Twitter, LinkedIn, Instagram, YouTube, Facebook)
- Tone variations (positive, emotional, educational, humorous, persuasive)
- CTA (Call-To-Action) injection system
- Trend keyword integration inside the prompt
- Clear, teacher-friendly examples for students
- Highly modular prompt builder so anyone can extend it easily

This version helps create highly personalized, platform-aware, trend-aware
prompts for Generative AI content generation.

CHANGELOG (vs. previous version)
-------------------------------------------------------------
1. FIXED: combined_keywords_str used `", ".join(set(combined_keywords))`.
   set() has two problems here: (a) it scrambles iteration order
   non-deterministically, so the exact same input produces a
   differently-ordered keyword list in the prompt on every run -- bad
   for reproducibility when comparing generated variants or debugging
   why an LLM's output changed run to run. (b) it only dedupes on exact
   string match, so "#AI" and "#ai" (or "AI" without the hash) both
   survive as separate "duplicate" entries. Replaced with an
   order-preserving, case/hash-normalized dedup.

   Note: this was partially masking a real double-counting issue coming
   from content_generator3.py, which calls this function with
   `injected_keywords` (already = keywords + deduped(real_trends)) AND
   passes the same `real_trends` again via `trends=`. The old set()
   call absorbed that duplication silently. The new dedup logic here
   still absorbs it correctly (exact/normalized duplicates are removed
   regardless of which file introduced them), but if you're debugging
   and see fewer keywords in the prompt than you expected, that's why --
   it's not a bug, it's two call sites redundantly including the same
   trends.

2. IMPROVED: platform display names for the prompt header now use a
   proper capitalization map instead of `.title()`, since
   "linkedin".title() -> "Linkedin" and "youtube".title() -> "Youtube",
   neither of which match the actual brand names ("LinkedIn",
   "YouTube"). Purely cosmetic (only affects the text the LLM sees, not
   functionality) but worth getting right since it's part of what's
   sent to the model.
-------------------------------------------------------------
"""

from typing import List, Optional


# ----------------------------------------
# 1. PLATFORM-SPECIFIC STYLES
# ----------------------------------------

PLATFORM_GUIDELINES = {
    "twitter": """
- Keep the post short, punchy, and fast-paced.
- Use strong hooks and 1–2 trending hashtags.
- Emojis are recommended but not too many.
- Make it shareable and conversation-friendly.
""",

    "instagram": """
- Use emojis heavily for emotional expression.
- Include storytelling elements.
- Add 2–4 hashtags at the end.
- Focus on visuals, feelings, and lifestyle tone.
""",

    "linkedin": """
- Use a professional and informative tone.
- Avoid excessive emojis.
- Include insights, value, and takeaways.
- End with a question or professional CTA.
""",

    "facebook": """
- Friendly, casual tone.
- Mix storytelling + information.
- Use emojis moderately.
""",

    "youtube": """
- Focus on curiosity hooks and value.
- Add call-to-action to watch, like, or subscribe.
- Include SEO-friendly keywords naturally.
"""
}

# Proper brand-name capitalization for platform names shown in the prompt.
# .title() alone gives "Linkedin"/"Youtube", which don't match real branding.
PLATFORM_DISPLAY_NAMES = {
    "twitter": "Twitter",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
    "youtube": "YouTube",
}


# ----------------------------------------
# 2. TONE PRESETS
# ----------------------------------------

TONE_STYLES = {
    "positive": "Use an energetic, uplifting, motivational tone.",
    "educational": "Explain concepts in a clear, simple, beginner-friendly way.",
    "emotional": "Add emotional depth, empathy, and relatability.",
    "humorous": "Include light humor, funny comparisons, or playful lines.",
    "persuasive": "Use convincing language, benefits, urgency, and social proof."
}


# ----------------------------------------
# 3. CALL-TO-ACTION (CTA) PRESETS
# ----------------------------------------

CTA_OPTIONS = [
    "Click to learn more!",
    "Share your thoughts below!",
    "Save this for later!",
    "Follow for more insights!",
    "Try it out today!",
    "What do YOU think?",
    "Join the conversation!"
]


def choose_cta(audience: str) -> str:
    """
    Choose a CTA based on audience context.

    Note: checks are first-match-wins in this fixed priority order
    (marketers > students > founders > default). If `audience` matches
    more than one keyword (e.g. "Marketers & Founders"), only the first
    match's CTA is used -- this is a deliberate simplification, not a
    bug, but worth knowing if you're wondering why a mixed-audience
    string didn't produce a "founders" CTA.
    """
    audience_lower = audience.lower()
    if "marketers" in audience_lower:
        return "Follow for more marketing insights!"
    if "students" in audience_lower:
        return "Save this tip for your next project!"
    if "founders" in audience_lower:
        return "Try this strategy today and scale faster!"
    return "Share your thoughts below!"


# ----------------------------------------
# 4. MAIN PROMPT GENERATOR
# ----------------------------------------

def _normalize_kw(kw: str) -> str:
    """Lowercase + strip leading '#' for duplicate comparison, so '#AI',
    'ai', and '#ai' are all recognized as the same keyword."""
    return kw.strip().lstrip("#").lower()


def _dedupe_ordered(items: List[str]) -> List[str]:
    """Order-preserving, normalized de-dup. Keeps the FIRST occurrence's
    original casing/formatting (e.g. keeps '#AI' if it appeared before
    a later plain 'ai')."""
    seen = set()
    out = []
    for item in items:
        key = _normalize_kw(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def generate_engaging_prompt(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    word_count: int = 50,
    trends: Optional[List[str]] = None,
    add_cta: bool = True
) -> str:
    """
    Build a highly adaptive AI prompt for content generation.
    - platform → controls style
    - tone → emotional, positive, etc.
    - keywords → user-defined
    - trends → real-time trending hashtags or keywords
    """
    platform_key = platform.lower()
    platform_rules = PLATFORM_GUIDELINES.get(platform_key, PLATFORM_GUIDELINES["twitter"])
    platform_display = PLATFORM_DISPLAY_NAMES.get(platform_key, platform.title())

    tone_rule = TONE_STYLES.get(tone.lower(), TONE_STYLES["positive"])

    # Combine keywords + trends, deduped and order-preserved (fixes the
    # non-deterministic set() ordering + case/hash-insensitive dupes).
    combined_keywords = _dedupe_ordered(keywords + (trends or []))
    combined_keywords_str = ", ".join(combined_keywords)

    # CTA
    cta_text = choose_cta(audience) if add_cta else ""

    prompt = f"""
You are a top-tier social media content creator.

Create an engaging, viral-ready post based on the following details:

Topic: {topic}
Platform: {platform_display}
Target Audience: {audience}
Keywords / Hashtags to include: {combined_keywords_str}
Tone Style: {tone}
Word Count: ~{word_count} words

Platform Style Guidelines:
{platform_rules}

Tone Instructions:
{tone_rule}

Additional Requirements:
- Must feel natural and human-like.
- Include emojis where appropriate (based on platform rules).
- Avoid overstuffing hashtags; keep them relevant.
- Make the opening strong and scroll-stopping.
- Ensure high readability and clarity.

Call to Action:
{cta_text}

Now generate the final post:
"""
    return prompt.strip()


# ----------------------------------------
# 5. TEST RUN
# ----------------------------------------

if __name__ == "__main__":
    prompt = generate_engaging_prompt(
        topic="AI in Digital Marketing",
        platform="Twitter",
        keywords=["#AI", "#Marketing"],
        audience="Marketers & Founders",
        tone="persuasive",
        trends=["#GenAI", "#Automation"],
        word_count=40
    )

    print("\n--- GENERATED PROMPT ---\n")
    print(prompt)