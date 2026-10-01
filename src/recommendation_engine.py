"""
Recommendation Generator and Sentiment Trends Analysis
For use with the main sentiment analysis tool

Provides:
1. Actionable recommendations based on aspect analysis
2. Sentiment trend tracking over time
3. Critical issue identification
"""

from collections import defaultdict
from datetime import datetime


class RecommendationEngine:
    """Generates recommendations based on aspect sentiments across reviews."""

    def __init__(self):
        self.aspect_history = defaultdict(list)
        self.review_count = 0

    def add_review(self, aspect_results):
        """Add a review's aspect results to history."""
        for aspect, (sentiment, score) in aspect_results.items():
            if sentiment != "NOT_MENTIONED":
                self.aspect_history[aspect].append({
                    "sentiment": sentiment,
                    "score": score,
                    "timestamp": datetime.now()
                })
        self.review_count += 1

    def get_aspect_summary(self):
        """Summarize sentiment for each aspect across all reviews."""
        summary = {}

        for aspect, history in self.aspect_history.items():
            if not history:
                summary[aspect] = {"status": "NOT_MENTIONED", "count": 0, "positive": 0, "negative": 0, "neutral": 0}
                continue

            positive_count = sum(1 for h in history if h["sentiment"] == "POSITIVE")
            negative_count = sum(1 for h in history if h["sentiment"] == "NEGATIVE")
            neutral_count = sum(1 for h in history if h["sentiment"] == "NEUTRAL")

            if positive_count > negative_count:
                status = "STRONG" if positive_count >= len(history) * 0.8 else "POSITIVE"
            elif negative_count > positive_count:
                status = "CRITICAL" if negative_count >= len(history) * 0.6 else "NEGATIVE"
            else:
                status = "MIXED"

            summary[aspect] = {
                "status": status,
                "count": len(history),
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "percentage_positive": (positive_count / len(history) * 100) if history else 0
            }

        return summary

    def generate_recommendations(self):
        """Generate actionable recommendations."""
        summary = self.get_aspect_summary()
        recommendations = {
            "critical": [],
            "improve": [],
            "maintain": [],
            "strengths": []
        }

        for aspect, data in summary.items():
            if data["count"] == 0:
                continue

            if data["status"] == "CRITICAL":
                recommendations["critical"].append(
                    f"🔴 CRITICAL - {aspect.capitalize()}: {data['negative']}/{data['count']} reviews negative. "
                    f"Action: Investigate and fix {aspect} issues urgently."
                )
            elif data["status"] == "NEGATIVE":
                recommendations["improve"].append(
                    f"🟡 NEEDS ATTENTION - {aspect.capitalize()}: {data['percentage_positive']:.0f}% positive. "
                    f"Action: Develop strategy to improve {aspect}."
                )
            elif data["status"] == "POSITIVE":
                recommendations["maintain"].append(
                    f"🟢 GOOD - {aspect.capitalize()}: {data['percentage_positive']:.0f}% positive. "
                    f"Action: Maintain current {aspect} quality."
                )
            elif data["status"] == "STRONG":
                recommendations["strengths"].append(
                    f"⭐ STRENGTH - {aspect.capitalize()}: {data['percentage_positive']:.0f}% positive across {data['count']} reviews. "
                    f"Action: Keep excelling in {aspect}!"
                )

        return recommendations

    def get_sentiment_trends(self):
        """Calculate sentiment trends for each aspect."""
        trends = {}

        for aspect, history in self.aspect_history.items():
            if len(history) < 2:
                trends[aspect] = "STABLE"
                continue

            first_half = history[:len(history) // 2]
            second_half = history[len(history) // 2:]

            first_positive = sum(1 for h in first_half if h["sentiment"] == "POSITIVE")
            second_positive = sum(1 for h in second_half if h["sentiment"] == "POSITIVE")

            first_rate = first_positive / len(first_half) if first_half else 0
            second_rate = second_positive / len(second_half) if second_half else 0

            diff = second_rate - first_rate

            if diff > 0.15:
                trends[aspect] = "IMPROVING ↗"
            elif diff < -0.15:
                trends[aspect] = "DECLINING ↘"
            else:
                trends[aspect] = "STABLE →"

        return trends


def display_recommendations(recommendations, aspect_summary, trends):
    """Display recommendations in a formatted way."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION REPORT")
    print("=" * 60)

    if recommendations["critical"]:
        print("\n🔴 CRITICAL ISSUES:")
        for rec in recommendations["critical"]:
            print(f"  {rec}")

    if recommendations["improve"]:
        print("\n🟡 AREAS TO IMPROVE:")
        for rec in recommendations["improve"]:
            print(f"  {rec}")

    if recommendations["maintain"]:
        print("\n🟢 AREAS TO MAINTAIN:")
        for rec in recommendations["maintain"]:
            print(f"  {rec}")

    if recommendations["strengths"]:
        print("\n⭐ STRENGTHS:")
        for rec in recommendations["strengths"]:
            print(f"  {rec}")

    print("\n" + "-" * 60)
    print("SENTIMENT TRENDS:")
    print("-" * 60)

    for aspect, trend in trends.items():
        summary = aspect_summary.get(aspect, {})
        if summary.get("count", 0) > 0:
            print(f"  {aspect.capitalize()}: {trend} (Positive: {summary['percentage_positive']:.0f}%)")

    print("\n" + "=" * 60)