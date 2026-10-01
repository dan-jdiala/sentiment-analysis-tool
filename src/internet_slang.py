"""
Internet Slang Detection Module
Maps common internet slang and abbreviations to sentiment scores
"""

# ========== POSITIVE SLANG ==========
POSITIVE_SLANG = {
    # Laughter/amusement
    "lol": 1,  # laugh out loud
    "lmao": 1,  # laughing my ass off
    "rofl": 1,  # rolling on floor laughing
    "haha": 1,  # laughter
    "hehe": 1,  # light laughter
    "lulz": 1,  # internet laughter

    # Agreement/enthusiasm
    "yay": 2,  # excitement
    "woo": 2,  # excitement
    "woohoo": 2,  # excitement
    "yes": 1,  # agreement
    "yeah": 1,  # agreement
    "yep": 1,  # agreement
    "yup": 1,  # agreement

    # Affection/positivity
    "omg": 1,  # oh my god - surprise/excitement
    "aww": 1,  # cuteness
    "aw": 1,  # cuteness
    "hmu": 0,  # hit me up - neutral
    "fyi": 0,  # for your information
    "tbh": 0,  # to be honest - neutral, adds emphasis
    "ngl": 0,  # not gonna lie - neutral, adds honesty
    "imo": 0,  # in my opinion - neutral
    "imho": 0,  # in my humble opinion - neutral

    # Casual positive
    "dope": 1,  # cool
    "sick": 1,  # awesome
    "lit": 1,  # awesome/exciting
    "fire": 1,  # awesome (but may be in lexicon with 🔥)
    "baller": 1,  # impressive
    "clutch": 1,  # excellent timing
    "slay": 1,  # dominate/do well
    "goals": 1,  # aspirational/good
    "vibes": 1,  # good energy
    "yas": 1,  # yes with enthusiasm

    # Support/encouragement
    "ftw": 1,  # for the win
    "gonna": 0,  # going to - neutral
    "wanna": 0,  # want to - neutral
}

# ========== NEGATIVE SLANG ==========
NEGATIVE_SLANG = {
    # Disappointment/disapproval
    "smh": -1,  # shaking my head
    "smdh": -1,  # shaking my damn head
    "smfh": -1,  # shaking my f***ing head
    "ugh": -1,  # disgust
    "eww": -1,  # disgust
    "blah": -1,  # boring/unimpressed
    "meh": -1,  # unimpressed
    "boo": -1,  # disapproval
    "booo": -1,  # strong disapproval
    "ew": -1,  # disgust
    "yikes": -1,  # worry/concern
    "yikes!": -1,  # worry/concern

    # Anger/frustration
    "wtf": -2,  # what the f***
    "wth": -1,  # what the hell
    "omfg": -2,  # oh my f***ing god
    "fml": -2,  # f*** my life
    "ugh": -1,  # frustration
    "argh": -1,  # frustration
    "ugh!!": -1,  # strong frustration

    # Sarcasm/mockery
    "yeah right": -2,  # sarcasm
    "sure": -1,  # sarcasm/doubt (context dependent)
    "right": -1,  # sarcasm/doubt
    "lol ok": -1,  # dismissive
    "ok lol": -1,  # dismissive

    # Tiredness/boredom
    "smh": -1,  # tired/unimpressed
    "done": -1,  # finished/frustrated
    "over it": -1,  # frustrated/tired
    "whatever": -1,  # dismissive
    "nope": -1,  # rejection

    # Disbelief/skepticism
    "seriously?": -1,  # doubt
    "really?": -1,  # skepticism
    "really??": -1,  # strong skepticism
}

# ========== NEUTRAL SLANG ==========
# These don't add sentiment but modify the tone
NEUTRAL_SLANG = {
    "btw": 0,  # by the way
    "fyi": 0,  # for your information
    "imho": 0,  # in my humble opinion
    "imo": 0,  # in my opinion
    "jk": 0,  # just kidding
    "tbh": 0,  # to be honest
    "ngl": 0,  # not gonna lie
    "imo": 0,  # in my opinion
    "idk": 0,  # i don't know
    "idc": 0,  # i don't care
    "nvm": 0,  # nevermind
    "asap": 0,  # as soon as possible
    "atm": 0,  # at the moment
    "tl;dr": 0,  # too long; didn't read
    "fyi": 0,  # for your information
    "afaik": 0,  # as far as i know
}

# ========== COMBINE ALL ==========
ALL_SLANG = {}
ALL_SLANG.update(POSITIVE_SLANG)
ALL_SLANG.update(NEGATIVE_SLANG)
ALL_SLANG.update(NEUTRAL_SLANG)


def detect_internet_slang(text: str) -> dict:
    """
    Detect internet slang in text and calculate sentiment adjustment.

    Args:
        text: Review text to analyze

    Returns:
        Dictionary with:
        - detected_slang: List of (slang, score) tuples found
        - total_adjustment: Total sentiment adjustment
        - positive_count: Number of positive slang terms
        - negative_count: Number of negative slang terms
        - neutral_count: Number of neutral slang terms
    """
    text_lower = text.lower()
    detected = []
    total_adjustment = 0
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    # Split by spaces and punctuation to get individual words
    import re
    words = re.findall(r'\b\w+\b', text_lower)

    for word in words:
        if word in ALL_SLANG:
            score = ALL_SLANG[word]
            detected.append((word, score))
            total_adjustment += score

            if score > 0:
                positive_count += 1
            elif score < 0:
                negative_count += 1
            else:
                neutral_count += 1

    return {
        "detected_slang": detected,
        "total_adjustment": total_adjustment,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
    }


def apply_slang_adjustments(text: str, total_pos: int, total_neg: int, debug: bool = False) -> tuple:
    """
    Apply internet slang sentiment adjustments to overall scores.

    Args:
        text: Review text
        total_pos: Current positive score
        total_neg: Current negative score
        debug: Enable debug output

    Returns:
        Tuple of (total_pos, total_neg) with slang adjustments applied
    """
    slang_analysis = detect_internet_slang(text)

    for slang, score in slang_analysis["detected_slang"]:
        if score > 0:
            total_pos += score
            if debug:
                print(f"[Internet Slang] '{slang}' → +{score} positive")
        elif score < 0:
            total_neg += abs(score)
            if debug:
                print(f"[Internet Slang] '{slang}' → -{abs(score)} negative")
        else:
            if debug:
                print(f"[Internet Slang] '{slang}' → neutral marker")

    return total_pos, total_neg


# ========== TEST EXAMPLES ==========
if __name__ == "__main__":
    test_reviews = [
        "lol this product is amazing!",
        "smh this is terrible",
        "honestly? ngl it's pretty good",
        "wtf is this garbage??? smh",
        "omg I love it!!! 10/10",
        "meh, it's okay I guess lol",
        "this is fire tbh",
        "ugh just awful. worst purchase ever smh",
        "yay it arrived! so excited!!!",
        "whatever, doesn't matter. done with this",
    ]

    print("=" * 80)
    print("INTERNET SLANG DETECTION TEST")
    print("=" * 80)

    for review in test_reviews:
        analysis = detect_internet_slang(review)
        print(f"\nReview: {review}")

        if analysis["detected_slang"]:
            print(f"  Detected slang:")
            for slang, score in analysis["detected_slang"]:
                direction = "+" if score > 0 else ("-" if score < 0 else "0")
                print(f"    - '{slang}': {direction}{abs(score)}")
            print(f"  Total adjustment: {analysis['total_adjustment']:+d}")
            print(f"  Breakdown: {analysis['positive_count']} positive, "
                  f"{analysis['negative_count']} negative, "
                  f"{analysis['neutral_count']} neutral")
        else:
            print("  No internet slang detected")