"""
Improved dynamic_prompt3.py

Upgrades added:
- Platform-specific content rules (Twitter, LinkedIn, Instagram, YouTube, Facebook)
- Tone variations (positive, emotional, educational, humorous, persuasive)
- CTA (Call-To-Action) injection system
- Trend keyword integration inside the prompt
- Clear, teacher-friendly examples for students
- Highly modular prompt builder so anyone can extend it easily

This version helps create highly personalized, platform-aware, trend-aware
prompts for Generative AI content generation.
"""

from typing import List


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
    """Choose a CTA based on audience context."""
    if "marketers" in audience.lower():
        return "Follow for more marketing insights!"
    if "students" in audience.lower():
        return "Save this tip for your next project!"
    if "founders" in audience.lower():
        return "Try this strategy today and scale faster!"
    return "Share your thoughts below!"


# ----------------------------------------
# 4. MAIN PROMPT GENERATOR
# ----------------------------------------

def generate_engaging_prompt(
    topic: str,
    platform: str,
    keywords: List[str],
    audience: str,
    tone: str = "positive",
    word_count: int = 50,
    trends: List[str] = None,
    add_cta: bool = True
) -> str:
    """
    Build a highly adaptive AI prompt for content generation.
    - platform → controls style
    - tone → emotional, positive, etc.
    - keywords → user-defined
    - trends → real-time trending hashtags or keywords
    """
    platform = platform.lower()
    platform_rules = PLATFORM_GUIDELINES.get(platform, PLATFORM_GUIDELINES["twitter"])

    tone_rule = TONE_STYLES.get(tone.lower(), TONE_STYLES["positive"])

    # Combine keywords + trends
    combined_keywords = keywords + (trends or [])
    combined_keywords_str = ", ".join(set(combined_keywords))

    # CTA
    cta_text = choose_cta(audience) if add_cta else ""

    prompt = f"""
You are a top-tier social media content creator.

Create an engaging, viral-ready post based on the following details:

Topic: {topic}
Platform: {platform.title()}
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
