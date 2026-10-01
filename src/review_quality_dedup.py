"""
Review Quality Scoring & Deduplication System

Provides:
1. Review Quality Scoring - Identifies detailed vs shallow reviews
2. Duplicate Detection - Finds potentially duplicate or spam reviews
3. Quality Metrics - Word count, aspect count, detail level
"""

from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import csv


class ReviewQualityScorer:
    """Score review quality based on length, detail, and completeness"""

    # Aspect keywords from ASPECTS dictionary
    ASPECT_KEYWORDS = {
        "food": {"food", "dish", "meal", "appetizer", "entree", "main", "dessert",
                 "sauce", "taste", "flavor", "recipe"},
        "service": {"service", "server", "waiter", "staff", "host", "friendly",
                    "attentive", "helpful", "rude"},
        "atmosphere": {"atmosphere", "ambiance", "decor", "clean", "noise", "quiet",
                       "music", "lighting"},
        "value": {"price", "cost", "expensive", "affordable", "worth", "value", "deal"},
        "quality": {"quality", "excellent", "great", "good", "bad", "amazing",
                    "terrible", "wonderful"},
        "experience": {"experience", "visit", "stay", "trip", "overall", "fun", "boring"},
        "performance": {"performance", "speed", "fast", "slow", "lag", "crash", "battery"},
        "design": {"design", "aesthetic", "visual", "layout", "interface", "ui"},
    }

    @staticmethod
    def score_review_quality(text: str) -> Dict:
        """
        Score review quality on multiple dimensions.

        Returns:
            {
                "quality_level": "DETAILED", "STANDARD", or "SHALLOW",
                "quality_score": 0-100,
                "word_count": int,
                "aspect_count": int,
                "detail_ratio": 0-1.0,
                "recommendation": str,
            }
        """
        text_lower = text.lower()
        word_count = len(text.split())

        # Count aspects mentioned
        aspects_found = set()
        for aspect, keywords in ReviewQualityScorer.ASPECT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    aspects_found.add(aspect)
                    break

        aspect_count = len(aspects_found)

        # Calculate scores
        # More words = more detail
        word_score = min(word_count / 150 * 40, 40)  # Max 40 points

        # More aspects mentioned = more comprehensive
        aspect_score = min(aspect_count / 4 * 30, 30)  # Max 30 points

        # Specific details (numbers, quotes, examples) = higher quality
        has_numbers = any(char.isdigit() for char in text)
        has_quotes = '"' in text or "'" in text
        has_specifics = has_numbers or has_quotes
        specific_score = 20 if has_specifics else 10  # Max 20 points

        total_score = int(word_score + aspect_score + specific_score)

        # Determine quality level
        if total_score >= 70:
            quality_level = "DETAILED"
            recommendation = "Highly Reliable - Multiple aspects covered with specifics"
        elif total_score >= 40:
            quality_level = "STANDARD"
            recommendation = "Adequate - Covers main points"
        else:
            quality_level = "SHALLOW"
            recommendation = "Limited detail - Consider requesting more info"

        return {
            "quality_level": quality_level,
            "quality_score": total_score,
            "word_count": word_count,
            "aspect_count": aspect_count,
            "detail_ratio": total_score / 100,
            "recommendation": recommendation,
            "has_numbers": has_numbers,
            "has_quotes": has_quotes,
        }


class ReviewDeduplicator:
    """Detect duplicate and potentially spam reviews"""

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two reviews (0.0 to 1.0)
        Uses SequenceMatcher for fuzzy matching
        """
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return 1.0

        ratio = SequenceMatcher(None, t1, t2).ratio()
        return ratio

    @staticmethod
    def find_duplicates(reviews: List[Dict], threshold: float = 0.85) -> List[Tuple]:
        """
        Find potential duplicate reviews.

        Args:
            reviews: List of review dicts with 'text' or 'review' key
            threshold: Similarity threshold (0.85 = 85% similar)

        Returns:
            List of (review1_idx, review2_idx, similarity, text_preview)
        """
        duplicates = []

        # Extract review texts
        review_texts = []
        for review in reviews:
            text = review.get("review") or review.get("text") or ""
            review_texts.append(text)

        # Compare all pairs
        for i in range(len(review_texts)):
            for j in range(i + 1, len(review_texts)):
                similarity = ReviewDeduplicator.calculate_similarity(
                    review_texts[i],
                    review_texts[j]
                )

                if similarity >= threshold:
                    duplicates.append({
                        "review1_idx": i,
                        "review2_idx": j,
                        "similarity": similarity,
                        "text1": review_texts[i][:50],
                        "text2": review_texts[j][:50],
                        "flag": "DUPLICATE" if similarity > 0.95 else "SIMILAR",
                    })

        return duplicates

    @staticmethod
    def detect_spam_patterns(text: str) -> Dict:
        """
        Detect potential spam patterns in review text.

        Returns flags for:
        - SUSPICIOUS_LENGTH: Extremely short or long
        - EXCESSIVE_CAPS: Too many capital letters
        - EXCESSIVE_PUNCTUATION: Too many exclamation/question marks
        - PROMOTIONAL: Contains promotional language
        - GENERIC: Generic copy-paste language
        """
        text_lower = text.lower()
        word_count = len(text.split())

        flags = []
        confidence = 0.0

        # Check length
        if word_count < 3:
            flags.append("SUSPICIOUS_LENGTH")
            confidence += 0.3
        elif word_count > 500:
            flags.append("SUSPICIOUS_LENGTH")
            confidence += 0.1

        # Check capitalization
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / len(text) if text else 0
        if caps_ratio > 0.3:
            flags.append("EXCESSIVE_CAPS")
            confidence += 0.2

        # Check punctuation
        punct_count = text.count("!") + text.count("?")
        punct_ratio = punct_count / word_count if word_count > 0 else 0
        if punct_ratio > 0.2:
            flags.append("EXCESSIVE_PUNCTUATION")
            confidence += 0.15

        # Check for promotional language
        promo_words = {"buy", "click", "link", "visit", "check out", "subscribe",
                       "follow", "sign up", "download", "install"}
        if any(word in text_lower for word in promo_words):
            flags.append("PROMOTIONAL")
            confidence += 0.25

        # Check for generic language
        generic_phrases = {"great product", "highly recommend", "best ever",
                           "terrible experience", "waste of money"}
        if any(phrase in text_lower for phrase in generic_phrases):
            flags.append("GENERIC")
            confidence += 0.1

        is_spam = confidence > 0.5

        return {
            "is_spam": is_spam,
            "confidence": confidence,
            "flags": flags,
            "word_count": word_count,
        }


def analyze_reviews_quality(csv_file: str) -> Dict:
    """
    Comprehensive analysis of review quality and duplicates.

    Args:
        csv_file: Path to CSV file with reviews

    Returns:
        Analysis results dictionary
    """

    print("\n" + "=" * 80)
    print("REVIEW QUALITY & DEDUPLICATION ANALYSIS")
    print("=" * 80 + "\n")

    reviews = []

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Find review column
            review_col = None
            for col in reader.fieldnames:
                if 'review' in col.lower() or 'text' in col.lower():
                    review_col = col
                    break

            if not review_col:
                print(f"Error: Could not find review column")
                return {}

            for row in reader:
                text = row.get(review_col, "").strip()
                if text:
                    reviews.append({"review": text, "original_row": row})

    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        return {}

    # Quality scoring
    print(f"{'=' * 80}\nQUALITY ANALYSIS\n{'=' * 80}\n")

    quality_results = []
    quality_distribution = {"DETAILED": 0, "STANDARD": 0, "SHALLOW": 0}

    for idx, review in enumerate(reviews):
        quality = ReviewQualityScorer.score_review_quality(review["review"])
        quality_results.append({
            "idx": idx,
            "review": review["review"][:50],
            **quality
        })
        quality_distribution[quality["quality_level"]] += 1

    # Print quality distribution
    print(f"Quality Distribution:")
    for level, count in quality_distribution.items():
        pct = (count / len(reviews) * 100) if reviews else 0
        print(f"  {level:10} {count:3} ({pct:5.1f}%)")

    # Show detailed reviews
    detailed_reviews = [r for r in quality_results if r["quality_level"] == "DETAILED"]
    print(f"\nHighly Detailed Reviews ({len(detailed_reviews)}):")
    for review in detailed_reviews[:5]:
        print(f"  Score: {review['quality_score']:3}/100 | Aspects: {review['aspect_count']} | "
              f"'{review['review']}...'")

    # Deduplication
    print(f"\n{'=' * 80}\nDUPLICATION ANALYSIS\n{'=' * 80}\n")

    duplicates = ReviewDeduplicator.find_duplicates(reviews, threshold=0.85)

    if duplicates:
        print(f"Found {len(duplicates)} potential duplicates:\n")
        for dup in duplicates[:10]:
            print(f"  Review #{dup['review1_idx']} vs #{dup['review2_idx']}")
            print(f"    Similarity: {dup['similarity']:.1%} [{dup['flag']}]")
            print(f"    Text 1: '{dup['text1']}...'")
            print(f"    Text 2: '{dup['text2']}...'")
            print()
    else:
        print("✓ No significant duplicates found\n")

    # Spam detection
    print(f"{'=' * 80}\nSPAM DETECTION\n{'=' * 80}\n")

    spam_reviews = []
    for idx, review in enumerate(reviews):
        spam_check = ReviewDeduplicator.detect_spam_patterns(review["review"])
        if spam_check["is_spam"]:
            spam_reviews.append({
                "idx": idx,
                "review": review["review"][:50],
                **spam_check
            })

    if spam_reviews:
        print(f"Flagged {len(spam_reviews)} potential spam reviews:\n")
        for spam in spam_reviews[:10]:
            print(f"  Review #{spam['idx']}")
            print(f"    Confidence: {spam['confidence']:.0%}")
            print(f"    Flags: {', '.join(spam['flags'])}")
            print(f"    Text: '{spam['review']}...'")
            print()
    else:
        print("✓ No spam detected\n")

    return {
        "total_reviews": len(reviews),
        "quality_results": quality_results,
        "quality_distribution": quality_distribution,
        "duplicates": duplicates,
        "spam_reviews": spam_reviews,
    }


if __name__ == "__main__":
    # Example usage
    csv_file = "reviews.csv"  # Change to your file
    results = analyze_reviews_quality(csv_file)