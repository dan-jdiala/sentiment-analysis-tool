"""
Batch Confidence Scoring System
Identifies reliable reviews and flags uncertain ones for manual review
"""

from typing import List, Dict
from collections import defaultdict
import statistics


class ConfidenceScorer:
    """
    Scores and analyzes confidence levels across a batch of reviews.
    Helps identify which reviews are most trustworthy.
    """

    def __init__(self):
        self.reviews_data = []

    def add_review(self, username: str, text: str, sentiment: str,
                   confidence: float, aspect_results: Dict):
        """Add a review to the batch"""
        self.reviews_data.append({
            "username": username,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "aspects": aspect_results,
            "word_count": len(text.split()),
            "aspect_count": sum(1 for asp, (sent, _) in aspect_results.items()
                                if sent != "NOT_MENTIONED"),
        })

    def calculate_batch_confidence(self) -> Dict:
        """
        Calculate overall confidence metrics for the batch.

        Returns:
            Dictionary with confidence statistics
        """
        if not self.reviews_data:
            return {"error": "No reviews to analyze"}

        confidences = [r["confidence"] for r in self.reviews_data]

        high_confidence = [r for r in self.reviews_data if r["confidence"] > 0.7]
        medium_confidence = [r for r in self.reviews_data
                             if 0.3 <= r["confidence"] <= 0.7]
        low_confidence = [r for r in self.reviews_data if r["confidence"] < 0.3]

        avg_confidence = statistics.mean(confidences)
        median_confidence = statistics.median(confidences)

        return {
            "total_reviews": len(self.reviews_data),
            "average_confidence": round(avg_confidence, 3),
            "median_confidence": round(median_confidence, 3),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "high_confidence_count": len(high_confidence),
            "medium_confidence_count": len(medium_confidence),
            "low_confidence_count": len(low_confidence),
            "high_confidence_percentage": round((len(high_confidence) / len(self.reviews_data)) * 100, 1),
            "low_confidence_percentage": round((len(low_confidence) / len(self.reviews_data)) * 100, 1),
        }

    def get_trustworthiness_tier(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.8:
            return "HIGHLY TRUSTWORTHY ⭐⭐⭐"
        elif confidence >= 0.6:
            return "TRUSTWORTHY ⭐⭐"
        elif confidence >= 0.4:
            return "MODERATE ⭐"
        elif confidence >= 0.2:
            return "LOW TRUST ⚠️"
        else:
            return "UNCERTAIN - REVIEW MANUALLY ❌"

    def identify_flagged_reviews(self) -> Dict:
        """
        Identify reviews that need manual review.

        Returns:
            Dictionary with flagged reviews by category
        """
        flagged = {
            "very_low_confidence": [],
            "mixed_sentiment_low_confidence": [],
            "contradictory": [],
        }

        for review in self.reviews_data:
            # Very low confidence
            if review["confidence"] < 0.15:
                flagged["very_low_confidence"].append({
                    "username": review["username"],
                    "text": review["text"][:100] + "...",
                    "sentiment": review["sentiment"],
                    "confidence": review["confidence"],
                    "reason": "Confidence below 15% - review is uncertain"
                })

            # Mixed/Neutral with low confidence
            if review["sentiment"] in ["MIXED", "NEUTRAL"] and review["confidence"] < 0.3:
                flagged["mixed_sentiment_low_confidence"].append({
                    "username": review["username"],
                    "text": review["text"][:100] + "...",
                    "sentiment": review["sentiment"],
                    "confidence": review["confidence"],
                    "reason": f"{review['sentiment']} sentiment with low confidence"
                })

            # Contradictory aspects (e.g., says NEGATIVE but all aspects positive)
            positive_aspects = sum(1 for asp, (sent, _) in review["aspects"].items()
                                   if sent == "POSITIVE")
            negative_aspects = sum(1 for asp, (sent, _) in review["aspects"].items()
                                   if sent == "NEGATIVE")

            if review["sentiment"] == "NEGATIVE" and positive_aspects > negative_aspects:
                flagged["contradictory"].append({
                    "username": review["username"],
                    "text": review["text"][:100] + "...",
                    "sentiment": review["sentiment"],
                    "confidence": review["confidence"],
                    "reason": "Says NEGATIVE but has more positive aspects"
                })
            elif review["sentiment"] == "POSITIVE" and negative_aspects > positive_aspects:
                flagged["contradictory"].append({
                    "username": review["username"],
                    "text": review["text"][:100] + "...",
                    "sentiment": review["sentiment"],
                    "confidence": review["confidence"],
                    "reason": "Says POSITIVE but has more negative aspects"
                })

        return flagged

    def get_top_reviews(self, n: int = 5) -> List[Dict]:
        """Get most trustworthy reviews"""
        sorted_reviews = sorted(
            self.reviews_data,
            key=lambda x: x["confidence"],
            reverse=True
        )
        return sorted_reviews[:n]

    def get_review_quality_score(self, review: Dict) -> Dict:
        """
        Score overall review quality based on multiple factors.

        Factors:
        - Confidence score (weight: 40%)
        - Word count (weight: 30%)
        - Aspect count (weight: 30%)
        """
        confidence_score = review["confidence"] * 100  # 0-100

        # Word count score (ideal: 50-300 words)
        word_count = review["word_count"]
        if word_count < 10:
            word_score = 10  # Too short
        elif word_count < 50:
            word_score = 60  # Short but acceptable
        elif word_count <= 300:
            word_score = 100  # Ideal
        elif word_count <= 500:
            word_score = 90  # Long but acceptable
        else:
            word_score = 70  # Too long

        # Aspect count score (ideal: 2+ aspects)
        aspect_count = review["aspect_count"]
        if aspect_count == 0:
            aspect_score = 20  # No aspects mentioned
        elif aspect_count == 1:
            aspect_score = 60  # Only one aspect
        elif aspect_count <= 5:
            aspect_score = 100  # Multiple aspects
        else:
            aspect_score = 90  # Many aspects

        # Calculate weighted quality score
        quality_score = (
                (confidence_score * 0.4) +
                (word_score * 0.3) +
                (aspect_score * 0.3)
        )

        return {
            "overall_quality": round(quality_score, 1),
            "confidence_component": round(confidence_score, 1),
            "word_count_component": word_score,
            "aspect_count_component": aspect_score,
            "word_count": word_count,
            "aspect_count": aspect_count,
        }

    def display_batch_report(self):
        """Display comprehensive batch confidence report"""
        if not self.reviews_data:
            print("No reviews to analyze")
            return

        batch_stats = self.calculate_batch_confidence()
        flagged = self.identify_flagged_reviews()

        print("\n" + "=" * 80)
        print("BATCH CONFIDENCE ANALYSIS REPORT")
        print("=" * 80 + "\n")

        # Overall Statistics
        print("OVERALL CONFIDENCE METRICS:")
        print(f"  Total Reviews: {batch_stats['total_reviews']}")
        print(f"  Average Confidence: {batch_stats['average_confidence']:.0%}")
        print(f"  Median Confidence: {batch_stats['median_confidence']:.0%}")
        print(f"  Range: {batch_stats['min_confidence']:.0%} - {batch_stats['max_confidence']:.0%}\n")

        # Confidence Distribution
        print("CONFIDENCE DISTRIBUTION:")
        print(
            f"  🟢 High Confidence (>70%): {batch_stats['high_confidence_count']} reviews ({batch_stats['high_confidence_percentage']}%)")
        print(f"  🟡 Medium Confidence (30-70%): {batch_stats['medium_confidence_count']} reviews")
        print(
            f"  🔴 Low Confidence (<30%): {batch_stats['low_confidence_count']} reviews ({batch_stats['low_confidence_percentage']}%)\n")

        # Trust Score
        trust_score = batch_stats['average_confidence'] * 100
        if trust_score >= 70:
            trust_level = "HIGHLY TRUSTWORTHY"
        elif trust_score >= 50:
            trust_level = "TRUSTWORTHY"
        elif trust_score >= 30:
            trust_level = "MODERATE"
        else:
            trust_level = "UNCERTAIN - REVIEW MANUALLY"

        print(f"OVERALL TRUST SCORE: {trust_score:.0f}/100 - {trust_level}\n")

        # Flagged Reviews
        print("=" * 80)
        print("FLAGGED REVIEWS FOR MANUAL REVIEW:")
        print("=" * 80 + "\n")

        if flagged["very_low_confidence"]:
            print(f"❌ VERY LOW CONFIDENCE ({len(flagged['very_low_confidence'])} reviews):")
            for review in flagged["very_low_confidence"]:
                print(f"  • {review['username']}: {review['sentiment']} ({review['confidence']:.0%})")
                print(f"    Reason: {review['reason']}\n")

        if flagged["mixed_sentiment_low_confidence"]:
            print(f"⚠️  MIXED/NEUTRAL WITH LOW CONFIDENCE ({len(flagged['mixed_sentiment_low_confidence'])} reviews):")
            for review in flagged["mixed_sentiment_low_confidence"]:
                print(f"  • {review['username']}: {review['sentiment']} ({review['confidence']:.0%})")
                print(f"    Reason: {review['reason']}\n")

        if flagged["contradictory"]:
            print(f"🔀 CONTRADICTORY ASPECT ANALYSIS ({len(flagged['contradictory'])} reviews):")
            for review in flagged["contradictory"]:
                print(f"  • {review['username']}: {review['sentiment']}")
                print(f"    Reason: {review['reason']}\n")

        if not any(flagged.values()):
            print("✅ No reviews flagged for manual review - all reviews appear reliable!\n")

        # Top Reviews
        print("=" * 80)
        print("MOST TRUSTWORTHY REVIEWS:")
        print("=" * 80 + "\n")

        top_reviews = self.get_top_reviews(5)
        for i, review in enumerate(top_reviews, 1):
            quality = self.get_review_quality_score(review)
            tier = self.get_trustworthiness_tier(review["confidence"])
            print(f"{i}. {review['username']}")
            print(f"   Sentiment: {review['sentiment']}")
            print(f"   Confidence: {review['confidence']:.0%}")
            print(f"   Trustworthiness: {tier}")
            print(f"   Quality Score: {quality['overall_quality']}/100")
            print(f"   Details: {review['word_count']} words, {review['aspect_count']} aspects\n")


# Example usage
if __name__ == "__main__":
    scorer = ConfidenceScorer()

    # Example reviews (you would populate this from your review processing)
    scorer.add_review(
        "Alice",
        "Great product, excellent quality, highly recommend!",
        "POSITIVE",
        0.92,
        {"quality": ("POSITIVE", 3), "experience": ("POSITIVE", 2)}
    )

    scorer.add_review(
        "Bob",
        "It's okay.",
        "NEUTRAL",
        0.10,
        {"quality": ("NEUTRAL", 0)}
    )

    scorer.add_review(
        "Charlie",
        "Mixed feelings. Good but bad.",
        "MIXED",
        0.20,
        {"quality": ("POSITIVE", 1), "performance": ("NEGATIVE", 1)}
    )

    # Display report
    scorer.display_batch_report()