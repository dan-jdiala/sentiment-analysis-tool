"""
Complete Sentiment Analysis Main Script - CORRECTED
Integrates sentiment analyzer, aspects, recommendations, domain lexicons,
temporal tracking, PDF report generation, and confidence scoring.
"""

import csv
import re
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from src.reviews import Reviews
from src.recommendation_engine import RecommendationEngine, display_recommendations
from sentiment_analyzer_improved import (
    sentiment_analysis, PhraseAnalyzer, expand_emojis_to_text,
    clean_input, get_word_points_with_pos, load_words, ASPECTS, debug_print,
    detect_sarcasm_emojis
)
from batch_confidence_scorer import ConfidenceScorer
from domain_lexicons import DomainLexicon, Domain
from temporal_tracking import TemporalSentimentTracker
from pdf_report_generator import SentimentReportGenerator
from review_quality_dedup import analyze_reviews_quality
from database_manager import SentimentDatabase

ENABLE_REASONING_TRACE = False

# Load sentiment lexicons
pos_mild = load_words("../lexicons/general/pos_mild")
pos_medium = load_words("../lexicons/general/pos_medium")
pos_strong = load_words("../lexicons/general/pos_strong")
neg_mild = load_words("../lexicons/general/neg_mild")
neg_medium = load_words("../lexicons/general/neg_medium")
neg_strong = load_words("../lexicons/general/neg_strong")
neutral = load_words("neutral_words")

# Store all processed reviews
saved_reviews = []


# === DOMAIN-SPECIFIC ANALYSIS ===
class SentimentAnalyzerWithDomain:
    """Wrapper that combines sentiment analysis with domain-specific lexicons"""

    def __init__(self, domain: Domain = Domain.GENERAL):
        self.domain_lex = DomainLexicon()
        self.domain = domain
        self.domain_lex.set_domain(domain)

    def set_domain(self, domain: Domain):
        """Switch to a different domain"""
        self.domain = domain
        self.domain_lex.set_domain(domain)

    def analyze(self, text: str, debug: bool = False) -> Dict:
        """Analyze sentiment using domain-specific lexicon"""
        result = sentiment_analysis(text, debug=debug)
        debug_print(debug, f"\n[Domain] Analysis using {self.domain.value} domain")
        return result


# === ASPECT ANALYSIS ===
def analyze_aspects(raw_text: str, debug: bool = False) -> Dict:
    """Perform aspect-based sentiment analysis."""
    # Reset aspects for this review
    for aspect in ASPECTS:
        ASPECTS[aspect]["sentiment"] = []

    text_lower = raw_text.lower()
    sentences = re.split(r'[.!?]+', text_lower)

    debug_print(debug, "\n=== ASPECT ANALYSIS ===\n")

    for sentence in sentences:
        if not sentence.strip():
            continue

        cleaned_words = clean_input(sentence)
        mentioned_aspects = {}
        sentence_lower = sentence.lower()

        # Find which aspects are mentioned
        for aspect, data in ASPECTS.items():
            matched_keywords = []
            for keyword in data["keywords"]:
                if re.search(r'\b' + re.escape(keyword) + r'\b', sentence_lower):
                    matched_keywords.append(keyword)

            if matched_keywords:
                mentioned_aspects[aspect] = matched_keywords
                debug_print(debug, f"  ✓ {aspect.upper()}: matched {matched_keywords}")

        # Calculate sentiment for this sentence
        sentence_sentiment = 0
        sentiment_words_found = []

        for word in cleaned_words:
            points = get_word_points_with_pos(word)
            if points != 0:
                sentence_sentiment += points
                sentiment_words_found.append((word, points))

        debug_print(debug, f"  Sentiment words: {sentiment_words_found}, Total: {sentence_sentiment}")

        # Assign sentiment to mentioned aspects
        if sentiment_words_found:
            for aspect in mentioned_aspects:
                ASPECTS[aspect]["sentiment"].append(sentence_sentiment)

    # Generate aspect results
    aspect_results = {}
    debug_print(debug, "\n=== ASPECT RESULTS ===\n")

    for aspect, data in ASPECTS.items():
        if data["sentiment"]:
            avg_sentiment = sum(data["sentiment"]) / len(data["sentiment"])
            SENTIMENT_THRESHOLD = 0.3

            if avg_sentiment > SENTIMENT_THRESHOLD:
                aspect_sentiment = "POSITIVE"
                aspect_score = int(avg_sentiment)
            elif avg_sentiment < -SENTIMENT_THRESHOLD:
                aspect_sentiment = "NEGATIVE"
                aspect_score = int(abs(avg_sentiment))
            else:
                aspect_sentiment = "NEUTRAL"
                aspect_score = 0

            aspect_results[aspect] = (aspect_sentiment, aspect_score)
            debug_print(debug, f"{aspect.upper()}: {aspect_sentiment} (raw: {avg_sentiment:.2f}, score: {aspect_score})")
        else:
            aspect_results[aspect] = ("NOT_MENTIONED", 0)

    debug_print(debug, "\n")
    return aspect_results


def get_aspect_weighted_summary(aspect_results: Dict, debug: bool = False) -> Tuple[str, float, float, List]:
    """Calculate weighted summary of aspects."""
    pos_weight = 0
    neg_weight = 0
    neutral_weight = 0
    aspects_mentioned = []

    aspect_weights = {
        "quality": 1.2, "design": 1.2, "performance": 1.3,
        "experience": 1.1, "atmosphere": 0.8, "value": 0.9,
        "service": 1.0, "food": 1.0
    }

    for aspect, (sentiment, score) in aspect_results.items():
        weight = aspect_weights.get(aspect, 1.0)

        if sentiment == "POSITIVE":
            pos_weight += weight
            aspects_mentioned.append((aspect, "POSITIVE", score))
        elif sentiment == "NEGATIVE":
            neg_weight += weight
            aspects_mentioned.append((aspect, "NEGATIVE", score))
        elif sentiment == "NEUTRAL":
            neutral_weight += weight

    total_weight = pos_weight + neg_weight + neutral_weight

    if total_weight == 0:
        return "NO_DATA", 0, 0, []

    pos_percentage = (pos_weight / total_weight) * 100
    neg_percentage = (neg_weight / total_weight) * 100

    if pos_weight > neg_weight:
        dominant = "POSITIVE"
    elif neg_weight > pos_weight:
        dominant = "NEGATIVE"
    else:
        dominant = "BALANCED"

    return dominant, pos_percentage, neg_percentage, aspects_mentioned


# === SARCASM DETECTION ===
def detect_sarcasm(text: str, debug: bool = False) -> Tuple[bool, float, str]:
    """Detect sarcasm in text."""
    text_lower = text.lower()
    sarcasm_score = 0
    sarcasm_type = None

    debug_print(debug, "\n=== SARCASM DETECTION ===\n")

    # Check for sarcasm emojis first (highest priority)
    has_emoji_sarcasm, emoji_confidence = detect_sarcasm_emojis(text, debug=debug)
    if has_emoji_sarcasm:
        return True, emoji_confidence, "sarcasm_emoji"

    # Pattern 1: Opening qualifiers with positive words
    opening_patterns = {"yeah", "sure", "right", "oh", "oh wow"}
    sentences = re.split(r'[.!?]+', text_lower)

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        first_words = sentence_clean.split()[:2]
        for pattern in opening_patterns:
            if any(pattern in word for word in first_words):
                if any(pos in sentence_clean for pos in ["great", "amazing", "wonderful", "excellent"]):
                    sarcasm_score += 2
                    sarcasm_type = "opening_qualifier"
                    debug_print(debug, f"  [Sarcasm] Opening qualifier detected")

    # Pattern 2: Excessive exclamation marks with positive sentiment
    exclamation_count = text.count("!")
    if exclamation_count >= 2:
        has_positive = any(word in text_lower for word in pos_strong)
        if has_positive:
            sarcasm_score += 0.5
            debug_print(debug, f"  [Sarcasm] Excessive exclamation marks")

    # Pattern 3: Closing with question markers
    if text_lower.rstrip().endswith(("right?", "sure?", "yeah?")):
        sarcasm_score += 1.5
        sarcasm_type = "closing_marker"
        debug_print(debug, f"  [Sarcasm] Closing sarcasm marker")

    # Determine sarcasm confidence
    if sarcasm_score >= 2:
        is_sarcastic = True
        sarcasm_confidence = min(sarcasm_score / 3.0, 1.0)
    elif sarcasm_score >= 1:
        is_sarcastic = True
        sarcasm_confidence = sarcasm_score / 2.0
    else:
        is_sarcastic = False
        sarcasm_confidence = 0.0

    debug_print(debug, f"  [Sarcasm] Score: {sarcasm_score}, Confidence: {sarcasm_confidence:.2f}\n")
    return is_sarcastic, sarcasm_confidence, sarcasm_type


# === MAIN PROCESSING FUNCTIONS ===
def get_username() -> str:
    """Get username from user."""
    return input("Enter the username: ").strip()


def get_user_review() -> str:
    """Get review text from user."""
    return input("Enter the review: ").strip()


def read_reviews_from_file(file_name: str) -> List[Reviews]:
    """Read reviews from CSV file."""
    reviews_list = []
    try:
        with open(file_name, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if it exists
            try:
                next(reader)
            except StopIteration:
                pass

            for row in reader:
                if len(row) >= 2:
                    username_val = row[0].strip()
                    review_text = row[1].strip()
                    review_obj = Reviews(username_val, review_text)
                    reviews_list.append(review_obj)

        if reviews_list:
            print(f"✓ Loaded {len(reviews_list)} reviews from {file_name}")
        return reviews_list
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return []


def process_review(username_val: str, review_text: str, engine: RecommendationEngine,
                   tracker: TemporalSentimentTracker, domain_analyzer: SentimentAnalyzerWithDomain,
                   debug: bool = False) -> Tuple[str, str, str, Dict, float, int, int, Tuple[bool, float, str]]:
    """
    Process and analyze a single review.
    Returns (username, review_text, sentiment, aspect_results, confidence, pos_score, neg_score, sarcasm_data)
    """
    # Run improved sentiment analysis
    result = domain_analyzer.analyze(review_text, debug=debug)
    sentiment = result["sentiment"]
    pos_score = result["pos_score"]
    neg_score = result["neg_score"]
    pos_count = result["pos_count"]
    neg_count = result["neg_count"]
    confidence = result["confidence"]

    # Perform aspect analysis
    aspect_results = analyze_aspects(review_text, debug=debug)

    # Get aspect-weighted summary
    dominant_aspect, pos_aspect_pct, neg_aspect_pct, aspects_mentioned = \
        get_aspect_weighted_summary(aspect_results, debug=debug)

    # Detect sarcasm
    sarcasm_detected, sarcasm_conf, sarcasm_type = detect_sarcasm(review_text, debug=debug)

    # Display results
    print("\n--- Sentiment Results ---")
    print(f"Total Positive Points: {pos_score}")
    print(f"Total Negative Points: {neg_score}")
    print(f"Positive words found: {pos_count}")
    print(f"Negative words found: {neg_count}")
    confidence_percent = int(confidence * 100)
    print(f"Sentiment: {sentiment} ({confidence_percent}% confidence)")

    if sarcasm_detected:
        sarcasm_percent = int(sarcasm_conf * 100)
        print(f"⚠️  Sarcasm Detected: {sarcasm_percent}% confidence ({sarcasm_type})\n")
    else:
        print()

    # Display aspect summary
    print("--- Aspect-Weighted Summary ---")
    if dominant_aspect == "NO_DATA":
        print("No specific aspects mentioned in this review.")
    else:
        print(f"Dominant aspect sentiment: {dominant_aspect}")
        print(f"Positive aspects: {pos_aspect_pct:.0f}%")
        print(f"Negative aspects: {neg_aspect_pct:.0f}%")
        if aspects_mentioned:
            print("Mentioned aspects:")
            for aspect, sent, score in aspects_mentioned:
                print(f"  - {aspect.capitalize()}: {sent} (score: {score})")
    print()

    # Add to recommendation engine
    engine.add_review(aspect_results)

    # Add to temporal tracker
    tracker.add_sentiment(sentiment, pos_score, neg_score, confidence, aspect_results)

    return username_val, review_text, sentiment, aspect_results, confidence, pos_score, neg_score, (sarcasm_detected, sarcasm_conf, sarcasm_type)


def save_reviews_to_file(reviews_list: List[Tuple], filename: str = "saved_reviews.csv",
                        should_analyze_quality: bool = False) -> None:
    """Append reviews to CSV file with timestamps."""
    if not reviews_list:
        print("No reviews to save.")
        return

    try:
        # Check if file exists to determine if we need to write header
        file_exists = False
        try:
            with open(filename, "r") as f:
                file_exists = True
        except FileNotFoundError:
            file_exists = False

        with open(filename, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                writer.writerow(["Username", "Review", "Sentiment", "Timestamp"])

            for username, review, sentiment in reviews_list:
                timestamp = datetime.now().strftime("%m/%d/%Y %I:%M:%S %p")
                writer.writerow([username, review, sentiment, timestamp])

        print(f"✓ {len(reviews_list)} review(s) appended to {filename}")

        if should_analyze_quality:
            print("\nAnalyzing review quality and duplicates...")
            quality_results = analyze_reviews_quality(filename)

    except IOError as e:
        print(f"Error saving reviews: {e}")


def display_saved_reviews(saved_reviews_list: List[Tuple]) -> None:
    """Display all saved reviews."""
    if not saved_reviews_list:
        print("\nSaved Reviews: None")
    else:
        print("\n" + "="*60)
        print("SAVED REVIEWS")
        print("="*60)
        for i, (username, review, sentiment) in enumerate(saved_reviews_list, 1):
            print(f"\n--- Review {i} ---")
            print(f"Username: {username}")
            print(f"Review: {review[:100]}..." if len(review) > 100 else f"Review: {review}")
            print(f"Sentiment: {sentiment}")


def select_domain() -> Domain:
    """Let user select domain"""
    print("\nSelect analysis domain:")
    print("1 - General (default)")
    print("2 - Restaurant")
    print("3 - Software")
    print("4 - Hotel")
    print("5 - Retail")

    choice = input("Enter 1-5: ").strip()

    domain_map = {
        "1": Domain.GENERAL,
        "2": Domain.RESTAURANT,
        "3": Domain.SOFTWARE,
        "4": Domain.HOTEL,
        "5": Domain.RETAIL,
    }

    return domain_map.get(choice, Domain.GENERAL)


# === MAIN PROGRAM ===
if __name__ == "__main__":
    print("=" * 60)
    print("SENTIMENT ANALYSIS SYSTEM")
    print("With Domain-Specific Lexicons, Temporal Tracking & Reports")
    print("=" * 60)

    # Initialize components
    engine = RecommendationEngine()
    tracker = TemporalSentimentTracker()
    domain = select_domain()
    domain_analyzer = SentimentAnalyzerWithDomain(domain)
    domain_lex = DomainLexicon()
    confidence_scorer = ConfidenceScorer()
    db = SentimentDatabase()
    should_analyze_quality = False

    while True:
        print("\n--- Input Method ---")
        print("1 - Enter reviews manually")
        print("2 - Load reviews from CSV file")
        print("3 - Exit")

        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "3":
            break

        debug_choice = input("Enable debug mode? (y/n): ").strip().lower()
        debug_mode = (debug_choice == "y")

        if choice == "1":
            # Manual entry mode
            while True:
                user = get_username()
                review_text = get_user_review()

                username_val, review_val, sentiment, aspect_results, confidence, pos_score, neg_score, sarcasm_data = \
                    process_review(user, review_text, engine, tracker, domain_analyzer, debug=debug_mode)

                saved_reviews.append((username_val, review_val, sentiment))

                confidence_scorer.add_review(
                    username=username_val,
                    text=review_val,
                    sentiment=sentiment,
                    confidence=confidence,
                    aspect_results=aspect_results
                )

                review_id = db.save_review(
                    username=username_val,
                    text=review_val,
                    sentiment=sentiment,
                    pos_score=pos_score,
                    neg_score=neg_score,
                    confidence=confidence,
                    aspect_results=aspect_results,
                    sarcasm_data=sarcasm_data,
                    domain=domain.value
                )

                print(f"✓ Saved to database with ID: {review_id}")

                continue_reviewing = input("Add another review? (y/n): ").strip().lower()
                if continue_reviewing != "y":
                    break

        elif choice == "2":
            # File input mode
            file_name = input("Enter CSV file name: ").strip()
            reviews_list = read_reviews_from_file(file_name)

            if reviews_list:
                for review_obj in reviews_list:
                    print(f"\nAnalyzing review from {review_obj.username}...")
                    username_val, review_val, sentiment, aspect_results, confidence, pos_score, neg_score, sarcasm_data = \
                        process_review(review_obj.username, review_obj.review, engine,
                                     tracker, domain_analyzer, debug=debug_mode)

                    saved_reviews.append((username_val, review_val, sentiment))

                    confidence_scorer.add_review(
                        username=username_val,
                        text=review_val,
                        sentiment=sentiment,
                        confidence=confidence,
                        aspect_results=aspect_results
                    )

                    review_id = db.save_review(
                        username=username_val,
                        text=review_val,
                        sentiment=sentiment,
                        pos_score=pos_score,
                        neg_score=neg_score,
                        confidence=confidence,
                        aspect_results=aspect_results,
                        sarcasm_data=sarcasm_data,
                        domain=domain.value
                    )

                    print(f"✓ Saved to database with ID: {review_id}")

            else:
                print("No reviews loaded from file.")
                continue

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            continue

    # Final summary
    print("\n" + "="*60)
    display_saved_reviews(saved_reviews)
    save_reviews_to_file(saved_reviews, should_analyze_quality=should_analyze_quality)
    print("="*60)

    # Batch confidence analysis
    if saved_reviews:
        print("\n" + "="*60)
        print("BATCH CONFIDENCE ANALYSIS")
        print("="*60)
        confidence_scorer.display_batch_report()

        # Generate recommendations
        print("\n" + "="*60)
        print("GENERATING RECOMMENDATIONS")
        print("="*60 + "\n")

        try:
            if not engine.review_count:
                print("[WARNING] No reviews were processed.")
            else:
                recommendations = engine.generate_recommendations()
                aspect_summary = engine.get_aspect_summary()
                trends = engine.get_sentiment_trends()
                display_recommendations(recommendations, aspect_summary, trends)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

        # Display temporal trends
        print("\n" + "="*60)
        print("TEMPORAL SENTIMENT ANALYSIS")
        print("="*60 + "\n")
        tracker.display_trend_report(days=30)

        # Generate PDF report
        print("\n" + "="*60)
        print("GENERATING PDF REPORT")
        print("="*60 + "\n")

        try:
            stats = tracker.get_statistics(days=30)
            trend = tracker.get_trend(days=30)

            summary_data = {
                "total_reviews": engine.review_count,
                "primary_sentiment": "positive" if engine.review_count > 0 else "neutral",
                "positive_count": stats.get("sentiment_distribution", {}).get("positive", 0),
                "negative_count": stats.get("sentiment_distribution", {}).get("negative", 0),
                "neutral_count": stats.get("sentiment_distribution", {}).get("neutral", 0),
                "average_confidence": confidence_scorer.calculate_batch_confidence().get("average_confidence", 0.75),
            }

            aspect_summary_for_pdf = {}
            aspect_summary = engine.get_aspect_summary()
            for aspect, data in aspect_summary.items():
                aspect_summary_for_pdf[aspect] = {
                    "status": data.get("status", "NOT_MENTIONED"),
                    "positive": data.get("positive", 0),
                    "negative": data.get("negative", 0),
                    "neutral": data.get("neutral", 0),
                    "count": data.get("count", 0),
                    "percentage_positive": data.get("percentage_positive", 0),
                }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"sentiment_report_{timestamp}.pdf"

            generator = SentimentReportGenerator(pdf_filename)
            generator.create_full_report(
                title=f"Sentiment Analysis Report - {domain.value.upper()}",
                summary=summary_data,
                aspect_data=aspect_summary_for_pdf,
                recommendations=engine.generate_recommendations(),
                trend_data=trend,
            )

            print(f"✓ PDF Report generated: {pdf_filename}")

        except Exception as e:
            print(f"[ERROR] Failed to generate PDF: {type(e).__name__}: {e}")

    print("\n" + "="*60)
    print("Thank you for using Sentiment Analysis System!")
    print("="*60)