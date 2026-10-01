"""
main_example.py - Complete Integration Example

Shows how to use the SentimentService with your existing code.
This demonstrates all major features and patterns.
"""

import sys
import logging
from typing import List, Dict
from services import SentimentService
from app_config import config

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: SINGLE REVIEW ANALYSIS
# ============================================================================

def example_single_review():
    """Analyze a single review"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Review Analysis")
    print("=" * 70)

    service = SentimentService()

    # Analyze one review
    result = service.analyze_single(
        username="alice",
        text="This product is amazing! Excellent quality and fast shipping!",
        domain="general",
        save_to_db=True
    )

    # Display results
    print(f"\n📊 Analysis Result:")
    print(f"  Sentiment:     {result['sentiment']}")
    print(f"  Confidence:    {result['confidence']:.1%}")
    print(f"  Score:         +{result['pos_score']}/-{result['neg_score']}")
    print(f"  Positive words: {result['pos_count']}")
    print(f"  Negative words: {result['neg_count']}")

    if 'review_id' in result:
        print(f"  Saved to DB:   ID {result['review_id']}")

    if result.get('aspects'):
        print(f"  Aspects:       {list(result['aspects'].keys())}")

    return result


# ============================================================================
# EXAMPLE 2: BATCH ANALYSIS
# ============================================================================

def example_batch_analysis():
    """Analyze multiple reviews at once"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Batch Analysis")
    print("=" * 70)

    service = SentimentService()

    reviews = [
        "This is fantastic! Best purchase ever!",
        "Terrible quality, broke in one week",
        "Good value for money, nothing special",
        "Excellent customer service, very helpful",
        "Awful product, total waste of time",
        "Pretty good, would recommend",
        "Not bad, could be better",
        "Amazing design and functionality!",
    ]

    # Batch analyze
    result = service.analyze_batch(
        username="batch_user",
        texts=reviews,
        domain="general",
        save_to_db=True
    )

    # Display summary
    print(f"\n📊 Batch Summary:")
    print(f"  Total reviews:   {result['total']}")
    print(f"  ✅ Positive:     {result['positive']} ({result['positive'] / result['total'] * 100:.0f}%)")
    print(f"  ❌ Negative:     {result['negative']} ({result['negative'] / result['total'] * 100:.0f}%)")
    print(f"  ⚪ Neutral:      {result['neutral']} ({result['neutral'] / result['total'] * 100:.0f}%)")
    print(f"  🔀 Mixed:        {result['mixed']} ({result['mixed'] / result['total'] * 100:.0f}%)")
    print(f"  📊 Avg Confidence: {result['avg_confidence']:.1%}")

    # Show performance
    if result.get('performance'):
        perf = result['performance']
        print(f"\n⚡ Performance:")
        print(f"  Total time:    {perf.get('total_time_ms', 0):.0f}ms")
        print(f"  Avg per review: {perf.get('avg_time_per_review_ms', 0):.0f}ms")
        print(f"  Performance:   {perf.get('performance', 'unknown')}")

    return result


# ============================================================================
# EXAMPLE 3: RETRIEVE AND QUERY DATA
# ============================================================================

def example_data_retrieval():
    """Retrieve data from database"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Data Retrieval")
    print("=" * 70)

    service = SentimentService()

    # Get all reviews
    print("\n📚 All Reviews:")
    all_reviews = service.get_all_reviews(limit=100)
    print(f"  Total reviews: {len(all_reviews)}")
    if all_reviews:
        print(f"  First review: {all_reviews[0]['text'][:50]}...")

    # Get by sentiment
    print("\n✅ Positive Reviews:")
    positive = service.get_reviews_by_sentiment("POSITIVE")
    print(f"  Count: {len(positive)}")

    print("\n❌ Negative Reviews:")
    negative = service.get_reviews_by_sentiment("NEGATIVE")
    print(f"  Count: {len(negative)}")

    # Search
    print("\n🔍 Search Results:")
    search = service.search_reviews("amazing", limit=10)
    print(f"  Found {len(search)} reviews containing 'amazing'")

    return {
        'all': all_reviews,
        'positive': positive,
        'negative': negative,
        'search': search
    }


# ============================================================================
# EXAMPLE 4: ANALYTICS
# ============================================================================

def example_analytics():
    """Get analytics and insights"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Analytics")
    print("=" * 70)

    service = SentimentService()

    # Get analytics
    analytics = service.get_analytics(days=30)

    print(f"\n📈 Analytics (Last 30 Days):")
    print(f"  Total reviews:     {analytics.get('total_reviews', 0)}")
    print(f"  Positive:          {analytics.get('positive', 0)}")
    print(f"  Negative:          {analytics.get('negative', 0)}")
    print(f"  Neutral:           {analytics.get('neutral', 0)}")
    print(f"  Mixed:             {analytics.get('mixed', 0)}")
    print(f"  Avg confidence:    {analytics.get('avg_confidence', 0):.1%}")

    # Get trends
    trends = service.get_trends(days=30)
    print(f"\n📊 Trends:")
    print(f"  Data points: {len(trends)}")
    if trends:
        print(f"  Latest: {trends[-1]}")

    # Get aspects
    aspects = service.get_aspect_overview()
    print(f"\n🎯 Aspect Summary:")
    print(f"  Aspects mentioned: {len(aspects)}")

    return analytics


# ============================================================================
# EXAMPLE 5: DOMAIN-SPECIFIC ANALYSIS
# ============================================================================

def example_domain_analysis():
    """Analyze reviews for different domains"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Domain-Specific Analysis")
    print("=" * 70)

    service = SentimentService()

    domains = ["general", "restaurant", "software", "hotel"]
    domain_reviews = {
        "general": ["Great product overall!"],
        "restaurant": ["Delicious food, amazing service!"],
        "software": ["Fast performance, intuitive interface"],
        "hotel": ["Clean rooms, excellent location!"]
    }

    for domain in domains:
        text = domain_reviews.get(domain, ["Good product"])[0]

        result = service.analyze_single(
            username=f"user_{domain}",
            text=text,
            domain=domain,
            save_to_db=False  # Don't save for this example
        )

        print(f"\n🌐 {domain.upper()}:")
        print(f"  Text:      {text}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score:     +{result['pos_score']}/-{result['neg_score']}")


# ============================================================================
# EXAMPLE 6: DATABASE OPERATIONS
# ============================================================================

def example_database_operations():
    """Perform database operations"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Database Operations")
    print("=" * 70)

    service = SentimentService()

    # Get database stats
    stats = service.get_database_stats()
    print(f"\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Health check
    health = service.health_check()
    print(f"\n🏥 Service Health:")
    for key, value in health.items():
        print(f"  {key}: {value}")


# ============================================================================
# EXAMPLE 7: INTERACTIVE MODE
# ============================================================================

def example_interactive():
    """Interactive review input"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Interactive Mode")
    print("=" * 70)

    service = SentimentService()

    while True:
        print("\n📝 Interactive Review Analysis:")
        print("  1 - Analyze a review")
        print("  2 - View analytics")
        print("  3 - Exit")

        choice = input("\nChoose (1-3): ").strip()

        if choice == "1":
            username = input("Username: ").strip()
            review = input("Review: ").strip()

            if username and review:
                result = service.analyze_single(username, review)
                print(f"\n✅ {result['sentiment']} ({result['confidence']:.0%})")
            else:
                print("❌ Both fields required")

        elif choice == "2":
            analytics = service.get_analytics()
            print(f"\n📊 Total reviews: {analytics.get('total_reviews', 0)}")
            print(f"✅ Positive: {analytics.get('positive', 0)}")
            print(f"❌ Negative: {analytics.get('negative', 0)}")

        elif choice == "3":
            break

        else:
            print("❌ Invalid choice")


# ============================================================================
# MAIN MENU
# ============================================================================

def main_menu():
    """Main menu for examples"""
    while True:
        print("\n" + "=" * 70)
        print("SENTIMENT ANALYSIS SYSTEM - EXAMPLES")
        print("=" * 70)
        print("\n1 - Single Review Analysis")
        print("2 - Batch Analysis")
        print("3 - Data Retrieval")
        print("4 - Analytics")
        print("5 - Domain Analysis")
        print("6 - Database Operations")
        print("7 - Interactive Mode")
        print("8 - Run All Examples")
        print("0 - Exit")

        choice = input("\nChoose (0-8): ").strip()

        try:
            if choice == "1":
                example_single_review()
            elif choice == "2":
                example_batch_analysis()
            elif choice == "3":
                example_data_retrieval()
            elif choice == "4":
                example_analytics()
            elif choice == "5":
                example_domain_analysis()
            elif choice == "6":
                example_database_operations()
            elif choice == "7":
                example_interactive()
            elif choice == "8":
                example_single_review()
                example_batch_analysis()
                example_data_retrieval()
                example_analytics()
                example_domain_analysis()
                example_database_operations()
            elif choice == "0":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"❌ Error: {e}")

        input("\nPress Enter to continue...")


# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test():
    """Quick test without menu"""
    print("\n" + "=" * 70)
    print("QUICK TEST")
    print("=" * 70)

    service = SentimentService()

    # Quick test
    result = service.analyze_single(
        username="test",
        text="This is amazing!",
        save_to_db=True
    )

    print(f"\n✅ Test successful!")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.0%}")

    if 'review_id' in result:
        print(f"Saved to database: ID {result['review_id']}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"Environment: {config.to_dict()}")

    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_test()
        else:
            print(f"Usage: python {sys.argv[0]} [--quick]")
    else:
        # Show menu
        main_menu()
