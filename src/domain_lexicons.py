"""
Domain-Specific Lexicon System
Allows different sentiment weights for the same word in different domains

Example: "heavy" in restaurant context (bad = too much food) vs
         "heavy" in software context (good = powerful/substantial)
"""

from typing import Dict, Set, Optional
from enum import Enum


class Domain(Enum):
    """Available domains for sentiment analysis"""
    GENERAL = "general"
    RESTAURANT = "restaurant"
    SOFTWARE = "software"
    HOTEL = "hotel"
    RETAIL = "retail"


class DomainLexicon:
    """
    Manages domain-specific sentiment lexicons.

    Each domain can have custom weights for words that have different
    sentiment values in different contexts.
    """

    def __init__(self):
        """Initialize domain lexicons"""
        self.lexicons = {
            Domain.GENERAL: self._load_general_lexicon(),
            Domain.RESTAURANT: self._load_restaurant_lexicon(),
            Domain.SOFTWARE: self._load_software_lexicon(),
            Domain.HOTEL: self._load_hotel_lexicon(),
            Domain.RETAIL: self._load_retail_lexicon(),
        }
        self.current_domain = Domain.GENERAL

    def set_domain(self, domain: Domain):
        """Switch to a different domain"""
        if domain in self.lexicons:
            self.current_domain = domain
            print(f"✓ Switched to {domain.value} domain")
        else:
            raise ValueError(f"Domain {domain} not found")

    def get_word_weight(self, word: str, domain: Optional[Domain] = None) -> int:
        """
        Get sentiment weight for a word in current or specified domain.

        Returns:
            int: -3 to 3 sentiment score (0 if word not found)
        """
        if domain is None:
            domain = self.current_domain

        lexicon = self.lexicons.get(domain, self.lexicons[Domain.GENERAL])

        # Check all sentiment levels
        for level, words in lexicon.items():
            if word.lower() in words:
                return level

        return 0

    def _load_general_lexicon(self) -> Dict[int, Set[str]]:
        """Base lexicon - neutral/general sentiment"""
        return {
            3: {"amazing", "excellent", "wonderful", "fantastic", "great", "love", "brilliant"},
            2: {"good", "nice", "like", "enjoy", "pleasant", "beautiful"},
            1: {"okay", "decent", "fine", "alright"},
            -1: {"bad", "poor", "disappointing", "dislike", "ugly"},
            -2: {"terrible", "awful", "horrible", "nasty"},
            -3: {"awful", "disgusting", "hate", "worst", "disaster"},
        }

    def _load_restaurant_lexicon(self) -> Dict[int, Set[str]]:
        """
        Restaurant-specific lexicon.

        Context matters:
        - "heavy" = negative (too much, too rich)
        - "light" = positive (fresh, not heavy)
        - "thick" = positive (good sauce/substance)
        - "thin" = negative (watery, weak flavor)
        """
        return {
            3: {"amazing", "excellent", "wonderful", "fantastic", "delicious", "fresh",
                "crispy", "flavorful", "tender", "perfectly cooked", "aromatic"},
            2: {"good", "nice", "tasty", "enjoyable", "appetizing", "fresh",
                "tender", "moist", "savory"},
            1: {"okay", "decent", "acceptable", "edible", "mild"},
            -1: {"bland", "dry", "overcooked", "undercooked", "stale", "heavy",
                 "greasy", "cold", "slow service"},
            -2: {"terrible", "awful", "inedible", "burnt", "salty", "rancid"},
            -3: {"disgusting", "poisoned", "worst meal", "food poisoning"},
        }

    def _load_software_lexicon(self) -> Dict[int, Set[str]]:
        """
        Software/gaming lexicon.

        Context matters:
        - "heavy" = positive (powerful, robust)
        - "light" = positive (fast, responsive)
        - "clunky" = negative (awkward, unresponsive)
        - "smooth" = positive (polished, seamless)
        - "lag" = negative (performance issue)
        - "crash" = very negative (critical failure)
        """
        return {
            3: {"amazing", "excellent", "smooth", "responsive", "intuitive", "fast",
                "optimized", "powerful", "robust", "stable", "feature-rich"},
            2: {"good", "nice", "functional", "reliable", "quick", "clean",
                "efficient", "polished", "well-designed"},
            1: {"okay", "usable", "decent", "acceptable", "adequate", "basic"},
            -1: {"slow", "buggy", "glitchy", "clunky", "cumbersome", "outdated",
                 "limited", "confusing"},
            -2: {"broken", "unplayable", "laggy", "crashes", "freezes", "awful ui"},
            -3: {"completely broken", "game breaking", "uninstall", "worst update"},
        }

    def _load_hotel_lexicon(self) -> Dict[int, Set[str]]:
        """
        Hotel/accommodation lexicon.

        Context matters:
        - "clean" = very positive (critical)
        - "noisy" = negative (sleep quality)
        - "spacious" = positive (room quality)
        - "cramped" = negative (poor design)
        """
        return {
            3: {"pristine", "immaculate", "luxurious", "spacious", "comfortable",
                "peaceful", "beautiful", "welcoming", "impeccable"},
            2: {"clean", "nice", "cozy", "friendly", "helpful", "good service", "quiet"},
            1: {"okay", "decent", "acceptable", "basic", "average"},
            -1: {"dirty", "noisy", "cramped", "dated", "uncomfortable", "mediocre",
                 "cold", "unfriendly"},
            -2: {"filthy", "disgusting", "unbearable noise", "broken", "bedbugs"},
            -3: {"infested", "unsafe", "health hazard", "worst stay"},
        }

    def _load_retail_lexicon(self) -> Dict[int, Set[str]]:
        """
        Retail/product lexicon.

        Context matters:
        - "cheap" = positive (good price) or negative (low quality)
        - "expensive" = negative (poor value)
        - "durable" = positive (quality)
        - "fragile" = negative (poor quality)
        """
        return {
            3: {"excellent", "amazing", "quality", "durable", "perfect fit",
                "great value", "worth it", "beautiful", "high-quality"},
            2: {"good", "nice", "well made", "reliable", "good price", "satisfied"},
            1: {"okay", "decent", "acceptable", "adequate", "satisfactory"},
            -1: {"cheap", "fragile", "uncomfortable", "poor quality", "not as described",
                 "overpriced", "damaged"},
            -2: {"broken", "defective", "waste of money", "scam", "falling apart"},
            -3: {"completely broken", "fraud", "worst purchase", "dangerous"},
        }

    def adjust_result_for_domain(self, domain: str, raw_text: str, result: Dict) -> Dict:
        """
        Adjust final sentiment scores based on domain-specific lexicon weights.
        This modifies pos_score, neg_score, and possibly the final sentiment.
        """

        # Convert domain string to Domain enum
        try:
            domain_enum = Domain(domain.lower())
        except Exception:
            domain_enum = Domain.GENERAL

        # Switch lexicon
        self.set_domain(domain_enum)

        words = raw_text.lower().split()

        domain_pos = 0
        domain_neg = 0

        for word in words:
            weight = self.get_word_weight(word, domain_enum)
            if weight > 0:
                domain_pos += weight
            elif weight < 0:
                domain_neg += abs(weight)

        # Apply domain adjustments
        result["pos_score"] += domain_pos
        result["neg_score"] += domain_neg

        # Recalculate sentiment
        if result["pos_score"] > result["neg_score"]:
            result["sentiment"] = "POSITIVE"
        elif result["neg_score"] > result["pos_score"]:
            result["sentiment"] = "NEGATIVE"
        else:
            result["sentiment"] = "NEUTRAL"

        return result

    def display_domain_weights(self, domain: Optional[Domain] = None):
        """Display all words and weights for a domain"""
        if domain is None:
            domain = self.current_domain

        lexicon = self.lexicons.get(domain, self.lexicons[Domain.GENERAL])

        print(f"\n{'='*60}")
        print(f"Domain-Specific Lexicon: {domain.value.upper()}")
        print(f"{'='*60}\n")

        for weight in [3, 2, 1, -1, -2, -3]:
            words = lexicon.get(weight, set())
            sentiment = "POSITIVE" if weight > 0 else "NEGATIVE" if weight < 0 else "NEUTRAL"
            strength = "STRONG" if abs(weight) == 3 else "MEDIUM" if abs(weight) == 2 else "MILD"

            print(f"[{sentiment} - {strength}] (weight: {weight:+d})")
            print(f"  {', '.join(sorted(words))}\n")


# Example usage and integration
if __name__ == "__main__":
    # Initialize domain lexicon system
    domain_lex = DomainLexicon()

    # Show general domain
    domain_lex.display_domain_weights(Domain.GENERAL)

    # Test restaurant domain
    print("\n" + "="*60)
    print("RESTAURANT DOMAIN TEST")
    print("="*60 + "\n")

    domain_lex.set_domain(Domain.RESTAURANT)

    test_words_restaurant = ["delicious", "heavy", "fresh", "bland", "crispy"]
    for word in test_words_restaurant:
        weight = domain_lex.get_word_weight(word)
        print(f"'{word}' in RESTAURANT: {weight:+d}")

    # Test software domain
    print("\n" + "="*60)
    print("SOFTWARE DOMAIN TEST")
    print("="*60 + "\n")

    domain_lex.set_domain(Domain.SOFTWARE)

    test_words_software = ["smooth", "lag", "responsive", "clunky", "crash"]
    for word in test_words_software:
        weight = domain_lex.get_word_weight(word)
        print(f"'{word}' in SOFTWARE: {weight:+d}")

    # Compare same word in different domains
    print("\n" + "="*60)
    print("DOMAIN COMPARISON TEST")
    print("="*60 + "\n")

    test_word = "light"
    print(f"Word: '{test_word}'")
    for domain in Domain:
        domain_lex.set_domain(domain)
        weight = domain_lex.get_word_weight(test_word)
        print(f"  {domain.value:15s}: {weight:+d}")