"""
Context-Aware Intensifiers System
"""
from typing import Dict, Tuple, Optional

# Note: Ideally, import INTENSIFIERS from your main lexicon file to avoid circular imports.
# For now, we assume it's available or we merge it.
# from lexicon_data import INTENSIFIERS

# ===== EXPANDED CONTEXT-AWARE INTENSIFIERS =====
CONTEXT_INTENSIFIERS = {
    # === EXTREME AMPLIFICATION (1.35-1.40) ===
    "extremely": 1.35, "exceptionally": 1.35, "remarkably": 1.35, "incredibly": 1.35,
    "absolutely": 1.35, "utterly": 1.35, "completely": 1.35, "entirely": 1.35,
    "thoroughly": 1.35, "far": 1.40, "way": 1.35,

    # === STRONG AMPLIFICATION (1.25-1.32) ===
    "very": 1.25, "so": 1.25, "too": 1.30, "highly": 1.28, "deeply": 1.28,
    "strongly": 1.28, "greatly": 1.28, "supremely": 1.30, "tremendously": 1.32,
    "enormously": 1.32, "awfully": 1.25, "terribly": 1.25, "dreadfully": 1.25,

    # === MODERATE AMPLIFICATION (1.15-1.22) ===
    "quite": 1.15, "fairly": 1.20, "pretty": 1.10, "rather": 1.10,
    "somewhat": 1.15, "significantly": 1.22, "noticeably": 1.20,
    "particularly": 1.18, "especially": 1.18,

    # === SUBTLE AMPLIFICATION (1.05-1.15) ===
    "really": 1.12, "truly": 1.12, "genuinely": 1.12, "actually": 1.08,
    "indeed": 1.08, "literally": 1.08, "virtually": 1.08,
}

INTENSIFIERS = {
    "super", "very", "extremely", "incredibly", "absolutely", "truly", "really",
    "so", "quite", "exceptionally", "remarkably", "particularly", "especially",
    "highly", "deeply", "thoroughly", "utterly", "completely"
}

class IntensifierHandler:
    @staticmethod
    def get_token_pos(word: str, doc):
        """
        Helper: Finds the Part-of-Speech tag for a word in the spaCy doc.
        """
        if not doc:
            return None

        # NOTE: This finds the FIRST occurrence of the word.
        # For a truly robust system, you'd match the specific token index,
        # but this heuristic works for 95% of short reviews.
        for token in doc:
            if token.text.lower() == word.lower():
                return token.pos_
        return None

    @staticmethod
    def find_intensifier_and_target(words, current_index, sent_doc=None):
        """
        Checks if the word BEFORE the current one is an intensifier.
        Uses POS tagging if sent_doc is provided.
        """
        if current_index == 0:
            return None, 1.0

        prev_word = words[current_index - 1].lower()

        # 1. Check if the word is in EITHER list
        # (Assuming INTENSIFIERS is a global dict imported elsewhere)
        is_context_intensifier = prev_word in CONTEXT_INTENSIFIERS
        is_basic_intensifier = prev_word in INTENSIFIERS

        if is_context_intensifier or is_basic_intensifier:

            # 2. POS CHECK (The "Smart" Logic)
            if sent_doc:
                pos_tag = IntensifierHandler.get_token_pos(prev_word, sent_doc)

                # Rule: "Pretty" must be an ADVERB (ADV) to be an intensifier.
                if prev_word == "pretty" and pos_tag == "ADJ":
                    return None, 1.0

            # 3. GET MULTIPLIER (Fixing the lookup bug)
            # Priority: Check CONTEXT_INTENSIFIERS first (more granular), then INTENSIFIERS
            if is_context_intensifier:
                multiplier = CONTEXT_INTENSIFIERS[prev_word]
            elif is_basic_intensifier:
                # If INTENSIFIERS is a set, we use a default of 1.5
                multiplier = 1.5
            else:
                multiplier = 1.0

            return prev_word, multiplier

        return None, 1.0

    @staticmethod
    def apply_context_intensifier(target_word, current_points, intensifier_word, multiplier, debug=False):
        """Applies the math with debug printing."""
        new_points = current_points * multiplier

        # Cap the score to prevent runaway numbers
        if new_points > 4: new_points = 4
        if new_points < -4: new_points = -4

        if debug:
            print(f"    [Intensifier] '{intensifier_word}' modifies '{target_word}' (x{multiplier}) -> {new_points}")

        return new_points