"""
Improved phrase detection for sentiment analysis
Handles positive negations like "can't complain", "won't complain", etc.
"""

import re


def apply_phrase_adjustments(text: str, total_pos: int, total_neg: int, debug: bool = False) -> tuple:
    """
    Apply special handling for positive negation phrases and idioms.
    These phrases express satisfaction despite containing negative words.

    Examples:
    - "Can't complain about the price" → +2 POSITIVE (not -2 negative)
    - "Won't complain about the service" → +2 POSITIVE
    - "No regrets" → +2 POSITIVE
    - "Can't fault the quality" → +2 POSITIVE

    Args:
        text: Review text to analyze
        total_pos: Current positive score
        total_neg: Current negative score
        debug: Enable debug output

    Returns:
        Tuple of (total_pos, total_neg)
    """
    text_lower = text.lower()

    # ========== EXACT PHRASE MATCHES ==========
    # These should match exactly as written
    exact_positive_phrases = {
        "no regrets",
        "don't regret",
        "didn't regret",
        "won't regret",
        "not sorry",
        "not a regret",
        "wouldn't change",
        "wouldn't change a thing",
        "can't ask for more",
        "couldn't ask for more",
    }

    for phrase in exact_positive_phrases:
        if phrase in text_lower:
            total_pos += 2
            if debug:
                print(f"[Phrase Adjustment] Found '{phrase}' → +2 positive")

    # ========== FLEXIBLE PATTERN MATCHES ==========
    # These use regex to match variations like "can't complain about X"
    flexible_patterns = [
        # Can't/won't complain variations
        (r"can't complain", "can't complain"),
        (r"cannot complain", "cannot complain"),
        (r"won't complain", "won't complain"),
        (r"will not complain", "will not complain"),
        (r"couldn't complain", "couldn't complain"),
        (r"not complaining", "not complaining"),

        # Can't fault variations
        (r"can't fault", "can't fault"),
        (r"cannot fault", "cannot fault"),
        (r"won't fault", "won't fault"),
        (r"couldn't fault", "couldn't fault"),
        (r"no fault", "no fault"),
        (r"can't find any fault", "can't find any fault"),

        # No issues variations
        (r"no problems? with", "no problem(s) with"),
        (r"no issues? with", "no issue(s) with"),
        (r"nothing wrong with", "nothing wrong with"),
        (r"no complaints?", "no complaint(s)"),

        # Satisfied variations
        (r"can't be happier", "can't be happier"),
        (r"couldn't be happier", "couldn't be happier"),
        (r"can't ask for better", "can't ask for better"),
        (r"couldn't ask for better", "couldn't ask for better"),

        # Perfect/flawless variations
        (r"nothing to complain about", "nothing to complain about"),
        (r"nothing bad to say", "nothing bad to say"),
    ]

    for pattern, phrase_name in flexible_patterns:
        if re.search(pattern, text_lower):
            total_pos += 1
            if debug:
                print(f"[Phrase Adjustment] Found pattern '{phrase_name}' → +1 positive")

    # ========== ASPECT-SPECIFIC POSITIVE PHRASES ==========
    # These mention specific aspects positively despite negative words
    aspect_patterns = {
        "price": [
            r"can't complain about.*price",
            r"cannot complain about.*price",
            r"price.*can't complain",
            r"price.*cannot complain",
            r"no complaints? about.*price",
            r"nothing wrong with.*price",
            r"fair.*price",
            r"reasonable.*price",
        ],
        "service": [
            r"can't complain about.*service",
            r"cannot complain about.*service",
            r"service.*can't complain",
            r"service.*cannot complain",
            r"no complaints? about.*service",
        ],
        "quality": [
            r"can't fault.*quality",
            r"cannot fault.*quality",
            r"quality.*can't complain",
            r"quality.*cannot complain",
            r"no fault.*quality",
        ],
        "food": [
            r"can't complain about.*food",
            r"cannot complain about.*food",
            r"food.*can't complain",
            r"food.*cannot complain",
        ],
    }

    for aspect, patterns in aspect_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                total_pos += 2
                if debug:
                    print(f"[Phrase Adjustment] Found '{aspect}' positive phrase → +2 positive")
                break  # Only count once per aspect

    # ========== NEGATION OF NEGATION PHRASES ==========
    # "Can't say I regret" type phrases - double negation = positive
    double_negation_patterns = [
        r"can't say i regret",
        r"cannot say i regret",
        r"won't say i regret",
        r"can't say it.*bad",
        r"cannot say it.*bad",
        r"can't say.*disappointed",
        r"cannot say.*disappointed",
    ]

    for pattern in double_negation_patterns:
        if re.search(pattern, text_lower):
            total_pos += 2
            if debug:
                print(f"[Phrase Adjustment] Found double negation pattern '{pattern}' → +2 positive")
            break

    return total_pos, total_neg


# ========== HELPER FUNCTION ==========
def detect_complaint_phrases(text: str) -> list:
    """
    Detect all complaint phrases in text and return them.
    Useful for debugging what phrases were detected.

    Args:
        text: Review text to analyze

    Returns:
        List of detected complaint phrases
    """
    detected = []

    exact_phrases = {
        "no regrets", "don't regret", "didn't regret", "won't regret",
        "not sorry", "not a regret", "wouldn't change", "can't ask for more",
    }

    flexible_patterns = [
        r"can't complain", r"cannot complain", r"won't complain",
        r"couldn't complain", r"not complaining", r"can't fault",
        r"no problems? with", r"no issues? with", r"nothing wrong with",
        r"can't be happier", r"couldn't be happier", r"can't ask for better",
        r"nothing to complain about", r"nothing bad to say",
    ]

    text_lower = text.lower()

    for phrase in exact_phrases:
        if phrase in text_lower:
            detected.append(phrase)

    for pattern in flexible_patterns:
        if re.search(pattern, text_lower):
            detected.append(pattern)

    return detected


# ========== TEST EXAMPLES ==========
if __name__ == "__main__":
    test_reviews = [
        "Not bad at all. It works fine and does what it's supposed to do. Nothing special but reliable. Can't complain about the price.",
        "The service is great. Won't complain about anything.",
        "Could be better but I can't fault the quality.",
        "No regrets buying this. Can't complain.",
        "Nothing wrong with the service, no complaints there.",
        "Can't ask for better quality at this price.",
        "The food was decent. Nothing to complain about.",
        "Couldn't be happier with my purchase!",
        "Can't say I regret this decision.",
    ]

    print("=" * 80)
    print("TESTING COMPLAINT PHRASE DETECTION")
    print("=" * 80)

    for review in test_reviews:
        detected = detect_complaint_phrases(review)
        print(f"\nReview: {review}")
        if detected:
            print(f"  Detected phrases: {detected}")
            total_pos = 0
            total_neg = 0
            total_pos, total_neg = apply_phrase_adjustments(review, total_pos, total_neg, debug=False)
            print(f"  Adjustment: +{total_pos}/-{total_neg}")
        else:
            print("  No complaint phrases detected")