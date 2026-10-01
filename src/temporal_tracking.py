"""
Temporal Sentiment Tracking
Tracks sentiment over time to detect trends and improvements.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import statistics


class TemporalSentimentTracker:
    """
    Tracks sentiment scores over time for trend analysis.
    Stores sentiment history and calculates improvement/decline trends.
    """

    def __init__(self):
        """Initialize the sentiment tracker"""
        self.sentiment_history: List[Dict] = []
        self.aspect_history: Dict[str, List[Dict]] = defaultdict(list)

    def add_sentiment(self, sentiment: str, pos_score: int, neg_score: int,
                      confidence: float, aspect_results: Dict = None,
                      timestamp: Optional[datetime] = None):
        """
        Add a sentiment entry with timestamp.

        Args:
            sentiment: "POSITIVE", "NEGATIVE", "NEUTRAL", "MIXED"
            pos_score: Positive sentiment points
            neg_score: Negative sentiment points
            confidence: Confidence score (0.0-1.0)
            aspect_results: Dict of aspect sentiment results
            timestamp: When this review was made (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        entry = {
            "timestamp": timestamp,
            "sentiment": sentiment,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "confidence": confidence,
            "net_score": pos_score - neg_score,  # For trend calculations
        }

        self.sentiment_history.append(entry)

        # Track aspect history
        if aspect_results:
            for aspect, (aspect_sentiment, aspect_score) in aspect_results.items():
                self.aspect_history[aspect].append({
                    "timestamp": timestamp,
                    "sentiment": aspect_sentiment,
                    "score": aspect_score,
                })

    def get_trend(self, days: int = 30) -> Dict:
        """
        Calculate sentiment trend over the last N days.

        Returns:
            {
                "direction": "IMPROVING", "DECLINING", or "STABLE",
                "change_percentage": float (positive = improving),
                "period_days": int,
                "starting_sentiment": float,
                "ending_sentiment": float,
                "confidence": float,
            }
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = [e for e in self.sentiment_history if e["timestamp"] >= cutoff_date]

        if len(recent) < 2:
            return {
                "direction": "INSUFFICIENT_DATA",
                "change_percentage": 0.0,
                "period_days": days,
                "entries_found": len(recent),
            }

        # Split into first and second half
        midpoint = len(recent) // 2
        first_half = recent[:midpoint]
        second_half = recent[midpoint:]

        # Calculate average net score for each half
        first_avg = statistics.mean([e["net_score"] for e in first_half])
        second_avg = statistics.mean([e["net_score"] for e in second_half])

        # Calculate percentage change
        if first_avg == 0:
            change_percentage = 100.0 if second_avg > 0 else -100.0
        else:
            change_percentage = ((second_avg - first_avg) / abs(first_avg)) * 100

        # Determine direction with threshold
        THRESHOLD = 10  # 10% change threshold
        if change_percentage > THRESHOLD:
            direction = "IMPROVING ↗"
        elif change_percentage < -THRESHOLD:
            direction = "DECLINING ↘"
        else:
            direction = "STABLE →"

        return {
            "direction": direction,
            "change_percentage": round(change_percentage, 2),
            "period_days": days,
            "entries_analyzed": len(recent),
            "starting_avg_score": round(first_avg, 2),
            "ending_avg_score": round(second_avg, 2),
        }

    def get_aspect_trend(self, aspect: str, days: int = 30) -> Dict:
        """
        Get trend for a specific aspect over time.

        Example: How is "service" sentiment trending?
        """
        if aspect not in self.aspect_history:
            return {"error": f"Aspect '{aspect}' not found"}

        cutoff_date = datetime.now() - timedelta(days=days)
        recent = [e for e in self.aspect_history[aspect]
                  if e["timestamp"] >= cutoff_date]

        if len(recent) < 2:
            return {
                "aspect": aspect,
                "direction": "INSUFFICIENT_DATA",
                "entries_found": len(recent),
            }

        # Count sentiment distribution
        positive = sum(1 for e in recent if e["sentiment"] == "POSITIVE")
        negative = sum(1 for e in recent if e["sentiment"] == "NEGATIVE")
        neutral = sum(1 for e in recent if e["sentiment"] == "NEUTRAL")

        total = len(recent)
        pos_percentage = (positive / total) * 100

        # Determine trend
        first_half = recent[:len(recent) // 2]
        second_half = recent[len(recent) // 2:]

        first_pos_rate = sum(1 for e in first_half if e["sentiment"] == "POSITIVE") / len(first_half)
        second_pos_rate = sum(1 for e in second_half if e["sentiment"] == "POSITIVE") / len(second_half)

        change = second_pos_rate - first_pos_rate

        if change > 0.15:
            direction = "IMPROVING ↗"
        elif change < -0.15:
            direction = "DECLINING ↘"
        else:
            direction = "STABLE →"

        return {
            "aspect": aspect,
            "direction": direction,
            "positive_percentage": round(pos_percentage, 1),
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
            "period_days": days,
        }

    def get_statistics(self, days: int = 30) -> Dict:
        """
        Get comprehensive sentiment statistics for a time period.
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = [e for e in self.sentiment_history if e["timestamp"] >= cutoff_date]

        if not recent:
            return {"error": "No data available for period"}

        sentiments = [e["sentiment"] for e in recent]
        scores = [e["net_score"] for e in recent]

        positive_count = sentiments.count("POSITIVE")
        negative_count = sentiments.count("NEGATIVE")
        neutral_count = sentiments.count("NEUTRAL")
        mixed_count = sentiments.count("MIXED")

        return {
            "period_days": days,
            "total_reviews": len(recent),
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "mixed": mixed_count,
            },
            "percentages": {
                "positive": round((positive_count / len(recent)) * 100, 1),
                "negative": round((negative_count / len(recent)) * 100, 1),
                "neutral": round((neutral_count / len(recent)) * 100, 1),
                "mixed": round((mixed_count / len(recent)) * 100, 1),
            },
            "average_net_score": round(statistics.mean(scores), 2),
            "median_net_score": round(statistics.median(scores), 2),
            "score_range": {
                "min": min(scores),
                "max": max(scores),
            }
        }

    def display_trend_report(self, days: int = 30):
        """Display a formatted trend report"""
        print("\n" + "=" * 60)
        print(f"SENTIMENT TREND REPORT (Last {days} Days)")
        print("=" * 60 + "\n")

        # Overall trend
        trend = self.get_trend(days)
        print("OVERALL SENTIMENT TREND:")
        print(f"  Direction: {trend.get('direction', 'N/A')}")
        print(f"  Change: {trend.get('change_percentage', 0):+.1f}%")
        print(f"  Starting Average: {trend.get('starting_avg_score', 0):+.1f}")
        print(f"  Ending Average: {trend.get('ending_avg_score', 0):+.1f}")
        print(f"  Reviews Analyzed: {trend.get('entries_analyzed', 0)}\n")

        # Statistics
        stats = self.get_statistics(days)
        if "error" not in stats:
            print("SENTIMENT DISTRIBUTION:")
            for sentiment, count in stats["sentiment_distribution"].items():
                percentage = stats["percentages"][sentiment]
                print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
            print(f"\n  Average Net Score: {stats['average_net_score']:+.2f}")
            print(f"  Median Net Score: {stats['median_net_score']:+.2f}\n")

        # Aspect trends
        if self.aspect_history:
            print("ASPECT TRENDS:")
            for aspect in sorted(self.aspect_history.keys()):
                aspect_trend = self.get_aspect_trend(aspect, days)
                if "error" not in aspect_trend:
                    print(f"  {aspect.capitalize()}: {aspect_trend['direction']} "
                          f"({aspect_trend['positive_percentage']:.0f}% positive)")

        print("\n" + "=" * 60)


# Example usage
if __name__ == "__main__":
    tracker = TemporalSentimentTracker()

    # Simulate reviews over 30 days
    base_date = datetime.now() - timedelta(days=29)

    # First 15 days - mostly negative
    for i in range(15):
        date = base_date + timedelta(days=i)
        if i % 3 == 0:
            tracker.add_sentiment("NEGATIVE", 1, 5, 0.7,
                                  {"food": ("NEGATIVE", -2), "service": ("POSITIVE", 2)},
                                  timestamp=date)
        else:
            tracker.add_sentiment("NEUTRAL", 2, 2, 0.4,
                                  {"food": ("NEUTRAL", 0), "service": ("NEGATIVE", -1)},
                                  timestamp=date)

    # Last 15 days - improving to positive
    for i in range(15, 30):
        date = base_date + timedelta(days=i)
        if i % 2 == 0:
            tracker.add_sentiment("POSITIVE", 5, 1, 0.8,
                                  {"food": ("POSITIVE", 2), "service": ("POSITIVE", 2)},
                                  timestamp=date)
        else:
            tracker.add_sentiment("NEUTRAL", 3, 1, 0.5,
                                  {"food": ("POSITIVE", 1), "service": ("POSITIVE", 1)},
                                  timestamp=date)

    # Display report
    tracker.display_trend_report(30)