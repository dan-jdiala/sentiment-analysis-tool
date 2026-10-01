"""
services.py - Service Layer that wraps your existing code

This module provides a clean, unified interface to all your sentiment analysis components.
It sits between your UI (Streamlit, Flask) and your core analyzer.

No changes needed to your existing code - this just organizes it!
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# Import your EXISTING modules - NO CHANGES NEEDED
from sentiment_analyzer_improved import (
    sentiment_analysis,
    batch_sentiment_analysis,
    get_timing_stats_summary,
    reset_timing_stats,
    detect_sarcasm_emojis
)
from database_manager import SentimentDatabase
from domain_lexicons import DomainLexicon, Domain

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentService:
    """
    Unified service layer for sentiment analysis.

    This wraps all your existing code into one clean API.
    Use this in your Flask API, Streamlit, or any other interface.

    Example:
        service = SentimentService()
        result = service.analyze_single("john", "Great product!")
        print(result['sentiment'])  # 'POSITIVE'
    """

    def __init__(self):
        """Initialize the service with all components"""
        logger.info("Initializing SentimentService...")

        try:
            self.db = SentimentDatabase()
            logger.info("✅ Database initialized (SQLite)")
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            self.db = None

        try:
            self.domain_lex = DomainLexicon()
            logger.info("✅ Domain lexicons initialized")
        except Exception as e:
            logger.error(f"❌ Domain lexicons error: {e}")
            self.domain_lex = None

        logger.info("✅ SentimentService ready")

    # ====================================================================
    # SINGLE REVIEW ANALYSIS
    # ====================================================================

    def analyze_single(self,
                       username: str,
                       text: str,
                       domain: str = "general",
                       save_to_db: bool = True,
                       debug: bool = False) -> Dict:
        """
        Analyze a single review.

        This is your main method - use it everywhere!

        Args:
            username: Username of reviewer
            text: Review text to analyze
            domain: Domain (general, restaurant, software, hotel, retail)
            save_to_db: Save result to SQLite database
            debug: Enable debug output

        Returns:
            Complete analysis result with sentiment, confidence, aspects, etc.

        Example:
            result = service.analyze_single("john", "Amazing product!")
            print(result['sentiment'])  # 'POSITIVE'
            print(result['confidence'])  # 0.92
            print(result['review_id'])  # 123 (if saved to DB)
        """

        try:
            logger.info(f"Analyzing review from {username}: {text[:50]}...")

            # Step 1: Run your existing sentiment analysis
            result = sentiment_analysis(text, debug=debug, domain=domain)

            # Step 2: Add metadata
            result['username'] = username
            result['text'] = text
            result['domain'] = domain
            result['analyzed_at'] = datetime.now().isoformat()

            # Step 3: Save to database if requested
            review_id = None
            if save_to_db and self.db:
                try:
                    review_id = self.db.save_review(
                        username=username,
                        text=text,
                        sentiment=result['sentiment'],
                        pos_score=result['pos_score'],
                        neg_score=result['neg_score'],
                        confidence=result['confidence'],
                        aspect_results=result.get('aspects', {}),
                        sarcasm_data=(
                            result.get('is_sarcastic', False),
                            result.get('sarcasm_confidence', 0)
                        ),
                        domain=domain
                    )
                    result['review_id'] = review_id
                    logger.info(f"✅ Saved to database with ID: {review_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to save to database: {e}")
                    result['review_id'] = None
                    result['db_error'] = str(e)

            logger.info(f"✅ Analysis complete: {result['sentiment']}")
            return result

        except Exception as e:
            logger.error(f"❌ Error analyzing review: {e}")
            raise

    # ====================================================================
    # BATCH ANALYSIS
    # ====================================================================

    def analyze_batch(self,
                      username: str,
                      texts: List[str],
                      domain: str = "general",
                      save_to_db: bool = True) -> Dict:
        """
        Analyze multiple reviews at once.

        Args:
            username: Username (applies to all reviews)
            texts: List of review texts
            domain: Domain for analysis
            save_to_db: Save all results to database

        Returns:
            Dictionary with batch results and statistics

        Example:
            results = service.analyze_batch(
                username="john",
                texts=["Great!", "Terrible", "Average"],
            )
            print(f"Analyzed {results['total']} reviews")
            print(f"Positive: {results['positive']}")
        """

        try:
            logger.info(f"Batch analyzing {len(texts)} reviews for {username}...")

            # Reset timing stats for fresh batch
            reset_timing_stats()

            # Step 1: Run your existing batch analysis
            results = batch_sentiment_analysis(texts, domain=domain)

            # Step 2: Add metadata to each result
            for i, result in enumerate(results):
                result['username'] = username
                result['text'] = texts[i]
                result['domain'] = domain

            # Step 3: Save all to database if requested
            if save_to_db and self.db:
                try:
                    saved_count = self.db.save_reviews_bulk(results)
                    logger.info(f"✅ Saved {saved_count} reviews to database")
                except Exception as e:
                    logger.error(f"❌ Failed to save batch: {e}")

            # Step 4: Calculate summary statistics
            positive = sum(1 for r in results if r['sentiment'] == 'POSITIVE')
            negative = sum(1 for r in results if r['sentiment'] == 'NEGATIVE')
            neutral = sum(1 for r in results if r['sentiment'] == 'NEUTRAL')
            mixed = sum(1 for r in results if r['sentiment'] == 'MIXED')
            avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0

            # Get performance stats
            performance = get_timing_stats_summary()

            logger.info(f"✅ Batch complete: {positive} positive, {negative} negative")

            return {
                'success': True,
                'total': len(results),
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'mixed': mixed,
                'avg_confidence': avg_confidence,
                'results': results,
                'performance': performance
            }

        except Exception as e:
            logger.error(f"❌ Error in batch analysis: {e}")
            raise

    # ====================================================================
    # DATABASE QUERIES
    # ====================================================================

    def get_all_reviews(self, limit: int = 1000) -> List[Dict]:
        """Get all reviews from database"""
        if not self.db:
            return []
        return self.db.get_all_reviews(limit=limit)

    def get_reviews_by_sentiment(self, sentiment: str) -> List[Dict]:
        """Get reviews by sentiment (POSITIVE, NEGATIVE, NEUTRAL, MIXED)"""
        if not self.db:
            return []
        return self.db.get_reviews_by_sentiment(sentiment.upper())

    def get_reviews_by_username(self, username: str) -> List[Dict]:
        """Get all reviews by a specific username"""
        if not self.db:
            return []
        return self.db.get_reviews_by_username(username)

    def search_reviews(self, keyword: str, limit: int = 50) -> List[Dict]:
        """Search reviews by keyword"""
        if not self.db:
            return []
        return self.db.search_reviews(keyword, limit=limit)

    def get_review(self, review_id: int) -> Optional[Dict]:
        """Get a specific review by ID"""
        if not self.db:
            return None
        return self.db.get_review(review_id)

    # ====================================================================
    # ANALYTICS
    # ====================================================================

    def get_analytics(self, days: int = 30) -> Dict:
        """
        Get complete analytics for specified period.

        Example:
            analytics = service.get_analytics(days=30)
            print(f"Total: {analytics['total_reviews']}")
            print(f"Positive: {analytics['positive']}")
        """
        if not self.db:
            return {}
        return self.db.get_statistics(days=days)

    def get_trends(self, days: int = 30) -> List[Dict]:
        """Get sentiment trends over time"""
        if not self.db:
            return []
        return self.db.get_trend(days=days)

    def get_aspect_overview(self) -> List[Dict]:
        """Get aspect-based sentiment overview"""
        if not self.db:
            return []
        return self.db.get_aspect_overview()

    def get_database_stats(self) -> Dict:
        """Get overall database statistics"""
        if not self.db:
            return {}
        return self.db.get_database_stats()

    # ====================================================================
    # DATA MANAGEMENT
    # ====================================================================

    def delete_review(self, review_id: int) -> bool:
        """Delete a specific review"""
        if not self.db:
            return False
        return self.db.delete_review(review_id)

    def delete_all_reviews(self) -> bool:
        """Delete all reviews (WARNING: PERMANENT)"""
        if not self.db:
            return False
        self.db.delete_all_reviews()
        logger.warning("⚠️  All reviews deleted!")
        return True

    def export_to_json(self, filename: str) -> bool:
        """Export all reviews to JSON file"""
        if not self.db:
            return False
        try:
            self.db.export_to_json(filename)
            logger.info(f"✅ Exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return False

    # ====================================================================
    # HEALTH CHECK & STATUS
    # ====================================================================

    def health_check(self) -> Dict:
        """Check if service is healthy"""
        return {
            'status': 'healthy' if self.db else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected' if self.db else 'disconnected',
            'sentiment_analyzer': 'loaded',
            'domain_lexicons': 'loaded' if self.domain_lex else 'not loaded'
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global service instance (singleton pattern)
_service_instance = None


def get_service() -> SentimentService:
    """
    Get or create the global service instance.

    Use this to avoid creating multiple service instances.

    Example:
        service = get_service()
        result = service.analyze_single("john", "Great!")
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = SentimentService()
    return _service_instance


def analyze(username: str, text: str, domain: str = "general", save: bool = True) -> Dict:
    """Quick analyze function - shorthand for common usage"""
    service = get_service()
    return service.analyze_single(username, text, domain, save)


def batch_analyze(username: str, texts: List[str], domain: str = "general", save: bool = True) -> Dict:
    """Quick batch analyze function"""
    service = get_service()
    return service.analyze_batch(username, texts, domain, save)


def get_all() -> List[Dict]:
    """Quick get all reviews function"""
    service = get_service()
    return service.get_all_reviews()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SENTIMENT SERVICE - BASIC USAGE EXAMPLES")
    print("=" * 70)

    # Initialize service
    service = SentimentService()

    # Check health
    print("\n📊 Service Health:")
    health = service.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")

    # Example 1: Single review
    print("\n\n📝 EXAMPLE 1: Single Review")
    print("-" * 70)

    result = service.analyze_single(
        username="john_doe",
        text="This product is absolutely amazing! Best purchase ever!",
        domain="general",
        save_to_db=True
    )

    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Score: +{result['pos_score']}/-{result['neg_score']}")
    if 'review_id' in result:
        print(f"Saved with ID: {result['review_id']}")

    # Example 2: Batch analysis
    print("\n\n📦 EXAMPLE 2: Batch Analysis")
    print("-" * 70)

    reviews = [
        "Great quality and fast shipping!",
        "Terrible product, waste of money",
        "It's okay, nothing special",
    ]

    batch_result = service.analyze_batch(
        username="batch_user",
        texts=reviews,
        domain="general"
    )

    print(f"Total: {batch_result['total']}")
    print(f"Positive: {batch_result['positive']}")
    print(f"Negative: {batch_result['negative']}")
    print(f"Avg Confidence: {batch_result['avg_confidence']:.1%}")

    # Example 3: Analytics
    print("\n\n📈 EXAMPLE 3: Analytics")
    print("-" * 70)

    analytics = service.get_analytics(days=30)
    print(f"Total reviews (30 days): {analytics.get('total_reviews', 0)}")
    print(f"Positive: {analytics.get('positive', 0)}")
    print(f"Negative: {analytics.get('negative', 0)}")

    print("\n" + "=" * 70)
    print("✅ Examples complete!")
    print("=" * 70)