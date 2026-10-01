"""
REST API for Sentiment Analysis System
Built with Flask - allows external applications to use the analyzer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
from sentiment_analyzer_improved import sentiment_analysis, batch_sentiment_analysis, get_timing_stats_summary, reset_timing_stats
from database_manager import SentimentDatabase
from domain_lexicons import Domain, DomainLexicon
import time

# In memory tracker for batch jobs
_batch_jobs = {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize database
db = SentimentDatabase()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Domain analyzer
domain_lex = DomainLexicon()


# === ERROR HANDLERS ===
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


# === HEALTH CHECK ===
@app.route('/health', methods=['GET'])
def health_check():
    """Check if API is running."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }), 200

@app.route('/api/v1/performance', methods=['GET'])
def get_performance():
    """Get the latest performance/timing stats."""
    try:
        timing_stats = get_timing_stats_summary()
        return jsonify({
            'success': True,
            'performance': timing_stats
        }), 200
    except Exception as e:
        logger.error(f"Error in get_performance: {str(e)}")
        return jsonify({'error': str(e)}), 500

# === SENTIMENT ANALYSIS ENDPOINTS ===
@app.route('/api/v1/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400

        text = data['text']
        username = data.get('username', 'anonymous')
        domain = data.get('domain', 'general')
        save_to_db = data.get('save_to_db', False)

        # Pass domain into analyzer
        result = sentiment_analysis(text, debug=False, domain=domain)

        # ✅ ADD THIS LINE - Set username in result
        result['username'] = username

        review_id = None
        if save_to_db:
            try:
                review_id = db.save_review(
                    username=username,
                    text=text,
                    sentiment=result['sentiment'],
                    pos_score=result['pos_score'],
                    neg_score=result['neg_score'],
                    confidence=result['confidence'],
                    aspect_results = result.get('aspects') or {},
                    sarcasm_data=None,
                    domain=domain
                )
                logger.info(f"Saved review {review_id} for {username}")
            except Exception as e:
                logger.error(f"Error saving review: {str(e)}")
                return jsonify({'error': f'Failed to save review: {str(e)}'}), 500

        return jsonify({
            'success': True,
            'review_id': review_id,
            'analysis': {
                'text': text,
                'sentiment': result['sentiment'],
                'confidence': round(result['confidence'], 2),
                'pos_score': result['pos_score'],
                'neg_score': result['neg_score'],
                'pos_count': result.get('pos_count', 0),
                'neg_count': result.get('neg_count', 0),
                'is_sarcastic': result.get('is_sarcastic', False),
                'sarcasm_confidence': round(result.get('sarcasm_confidence', 0), 2),
                'aspects': result.get('aspects', {})
            },
            'timestamp': datetime.now().isoformat()
        }), 201
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple reviews with performance metrics."""
    try:
        from sentiment_analyzer_improved import COLLECT_TIMING_STATS

        # Disable timing collection for speed
        import sentiment_analyzer_improved
        sentiment_analyzer_improved.COLLECT_TIMING_STATS = False

        data = request.get_json()
        reviews = data.get('reviews', [])
        texts = [r['text'] for r in reviews]
        domain = data.get('domain', 'general')

        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400

        # Reset timing stats BEFORE processing
        reset_timing_stats()

        # Process all sentiments at once
        results = batch_sentiment_analysis(texts, domain=domain)

        print(f"\n[BATCH ANALYZE DEBUG]")
        print(f"  Results count: {len(results)}")
        if results:
            first_result = results[0]
            print(f"  First result keys: {list(first_result.keys())}")
            print(f"  Has username? {'username' in first_result}")
            print(f"  Username value: {first_result.get('username')}")

        # Get timing stats AFTER processing
        timing_stats = get_timing_stats_summary()

        # Add metadata back to results (IMPORTANT: add 'text' field for database)
        for i, res in enumerate(results):
            res['username'] = reviews[i].get('username', 'anonymous')
            res['text'] = texts[i]  # <-- ADD THIS: database needs the original text
            res['domain'] = domain

        # Bulk save to database

        if data.get('save_to_db'):
            try:
                saved_count = db.save_reviews_bulk(results)  # ✅ Capture return value
                if saved_count > 0:
                    logger.info(f"✅ Saved {saved_count}/{len(results)} reviews to database")
                else:
                    logger.error(f"❌ Database save returned 0 reviews saved!")
                    # Consider failing here or returning warning
            except Exception as e:
                logger.error(f"❌ Error saving reviews: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Database save failed: {str(e)}',
                    'results': results  # Return results anyway so user doesn't lose them
                }), 500

        return jsonify({
            'success': True,
            'total_reviews': len(results),
            'results': results,
            'performance': timing_stats
        }), 201

        sentiment_analyzer_improved.COLLECT_TIMING_STATS = True

    except Exception as e:
        logger.error(f"Error in batch_analyze: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/batch-status/<job_id>', methods=['GET'])
def get_batch_status(job_id):
    """Get status of a batch job."""
    if job_id in _batch_jobs:
        job = _batch_jobs[job_id]
        return jsonify({
            'success': True,
            'status': job['status'],
            'processed': job.get('processed', 0),
            'total': job.get('total', 0),
            'percent': (job.get('processed', 0) / job.get('total', 1) * 100)
        }), 200
    else:
        return jsonify({'error': 'Job not found'}), 404


@app.route('/api/v1/batch-analyze-debug', methods=['POST'])
def batch_analyze_debug():
    """Debug version - shows timing for each step."""
    try:
        t_start = time.time()

        data = request.get_json()
        reviews = data.get('reviews', [])
        texts = [r['text'] for r in reviews]
        domain = data.get('domain', 'general')

        logger.info(f"[BATCH] Starting with {len(reviews)} reviews")
        t_parse = time.time()
        logger.info(f"[BATCH] Parse JSON: {(t_parse - t_start) * 1000:.0f}ms")

        if not reviews:
            return jsonify({'error': 'No reviews provided'}), 400

        # Reset timing stats BEFORE processing
        reset_timing_stats()
        t_reset = time.time()
        logger.info(f"[BATCH] Reset stats: {(t_reset - t_parse) * 1000:.0f}ms")

        # Process all sentiments at once
        logger.info(f"[BATCH] Starting sentiment analysis...")
        t_analysis_start = time.time()

        results = batch_sentiment_analysis(texts, domain=domain)

        t_analysis_end = time.time()
        analysis_time = (t_analysis_end - t_analysis_start)
        logger.info(
            f"[BATCH] Sentiment analysis TOTAL: {analysis_time:.1f}s ({analysis_time / len(reviews) * 1000:.0f}ms per review)")

        # Get timing stats AFTER processing
        timing_stats = get_timing_stats_summary()
        t_stats = time.time()
        logger.info(f"[BATCH] Get timing stats: {(t_stats - t_analysis_end) * 1000:.0f}ms")

        # Add metadata back to results
        for i, res in enumerate(results):
            res['username'] = reviews[i].get('username', 'anonymous')
        t_metadata = time.time()
        logger.info(f"[BATCH] Add metadata: {(t_metadata - t_stats) * 1000:.0f}ms")

        # Bulk save to database
        if data.get('save_to_db'):
            logger.info(f"[BATCH] Starting database save...")
            try:
                db.save_reviews_bulk(results)
                logger.info(f"Saved {len(results)} reviews to database")
            except Exception as e:
                logger.error(f"Error saving reviews: {str(e)}")

        t_save = time.time()
        logger.info(f"[BATCH] Database save: {(t_save - t_metadata) * 1000:.0f}ms")

        # Build response
        response_data = {
            'success': True,
            'total_reviews': len(results),
            'results': results,
            'performance': timing_stats
        }
        t_response = time.time()
        logger.info(f"[BATCH] Build response JSON: {(t_response - t_save) * 1000:.0f}ms")

        total_time = (t_response - t_start)
        logger.info(f"[BATCH] ===== TOTAL TIME: {total_time:.1f}s =====")

        logger.info(f"[BATCH] Breakdown:")
        logger.info(f"  - Parse JSON:        {(t_parse - t_start) * 1000:>6.0f}ms")
        logger.info(f"  - Reset stats:       {(t_reset - t_parse) * 1000:>6.0f}ms")
        logger.info(f"  - Sentiment analysis:{analysis_time * 1000:>6.0f}ms")
        logger.info(f"  - Get stats:         {(t_stats - t_analysis_end) * 1000:>6.0f}ms")
        logger.info(f"  - Add metadata:      {(t_metadata - t_stats) * 1000:>6.0f}ms")
        logger.info(f"  - Save to DB:        {(t_save - t_metadata) * 1000:>6.0f}ms")
        logger.info(f"  - Build response:    {(t_response - t_save) * 1000:>6.0f}ms")

        return jsonify(response_data), 201

    except Exception as e:
        logger.error(f"Error in batch_analyze_debug: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/aspects', methods=['GET'])
def get_aspects():
    """Get aspect analysis overview."""
    try:
        aspects = db.get_aspect_overview()

        # Debug: log what we're returning
        logger.info(f"[ASPECTS] Returning {len(aspects)} aspect records")
        if aspects:
            logger.info(f"[ASPECTS] Sample: {aspects[0]}")

        return jsonify({
            'success': True,
            'aspects': aspects,
            'total_records': len(aspects)
        }), 200
    except Exception as e:
        logger.error(f"Error in get_aspects: {str(e)}", exc_info=True)
        return jsonify({'error': str(e), 'success': False}), 500


# === DATABASE RETRIEVAL ENDPOINTS ===
@app.route('/api/v1/reviews/<int:review_id>', methods=['GET'])
def get_review(review_id):
    """Get a specific review by ID."""
    try:
        review = db.get_review(review_id)
        if not review:
            return jsonify({'error': 'Review not found'}), 404

        return jsonify({
            'success': True,
            'review': review
        }), 200

    except Exception as e:
        logger.error(f"Error in get_review: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/reviews', methods=['GET'])
def list_reviews():
    """
    Get all reviews with pagination.

    GET /api/v1/reviews?limit=50&offset=0
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        limit = min(limit, 500)  # Cap at 500

        reviews = db.get_all_reviews(limit=limit, offset=offset)

        return jsonify({
            'success': True,
            'total': len(reviews),
            'limit': limit,
            'offset': offset,
            'reviews': reviews
        }), 200

    except Exception as e:
        logger.error(f"Error in list_reviews: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/reviews/sentiment/<sentiment>', methods=['GET'])
def get_by_sentiment(sentiment):
    """Get all reviews with specific sentiment."""
    try:
        valid_sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED']
        if sentiment.upper() not in valid_sentiments:
            return jsonify({'error': 'Invalid sentiment'}), 400

        reviews = db.get_reviews_by_sentiment(sentiment.upper())

        return jsonify({
            'success': True,
            'sentiment': sentiment.upper(),
            'total': len(reviews),
            'reviews': reviews
        }), 200

    except Exception as e:
        logger.error(f"Error in get_by_sentiment: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/reviews/user/<username>', methods=['GET'])
def get_by_username(username):
    """Get all reviews from a specific user."""
    try:
        reviews = db.get_reviews_by_username(username)

        return jsonify({
            'success': True,
            'username': username,
            'total': len(reviews),
            'reviews': reviews
        }), 200

    except Exception as e:
        logger.error(f"Error in get_by_username: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/search', methods=['GET'])
def search():
    """
    Search reviews by keyword.

    GET /api/v1/search?q=amazing&limit=50
    """
    try:
        keyword = request.args.get('q', '', type=str)
        limit = request.args.get('limit', 50, type=int)

        if not keyword:
            return jsonify({'error': 'Missing search query parameter: q'}), 400

        results = db.search_reviews(keyword, limit=limit)

        return jsonify({
            'success': True,
            'query': keyword,
            'total': len(results),
            'results': results
        }), 200

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === STATISTICS ENDPOINTS ===
@app.route('/api/v1/statistics', methods=['GET'])
def get_statistics():
    """
    Get sentiment statistics.

    GET /api/v1/statistics?days=30
    """
    try:
        days = request.args.get('days', 30, type=int)
        stats = db.get_statistics(days=days)

        return jsonify({
            'success': True,
            'period_days': days,
            'statistics': stats
        }), 200

    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/trends', methods=['GET'])
def get_trends():
    """
    Get sentiment trends over time.

    GET /api/v1/trends?days=30
    """
    try:
        days = request.args.get('days', 30, type=int)
        trends = db.get_trend(days=days)

        return jsonify({
            'success': True,
            'period_days': days,
            'trends': trends
        }), 200

    except Exception as e:
        logger.error(f"Error in get_trends: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/info', methods=['GET'])
def get_database_info():
    """Get overall database statistics."""
    try:
        stats = db.get_database_stats()

        return jsonify({
            'success': True,
            'database_info': stats
        }), 200

    except Exception as e:
        logger.error(f"Error in get_database_info: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === DATA MANAGEMENT ENDPOINTS ===
@app.route('/api/v1/reviews/<int:review_id>', methods=['DELETE'])
def delete_review(review_id):
    """Delete a review."""
    try:
        success = db.delete_review(review_id)

        if not success:
            return jsonify({'error': 'Review not found'}), 404

        return jsonify({
            'success': True,
            'message': f'Review {review_id} deleted'
        }), 200

    except Exception as e:
        logger.error(f"Error in delete_review: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/reviews', methods=['DELETE'])
def delete_all_reviews():
    try:
        db.delete_all_reviews()
        return jsonify({
            'success': True,
            'message': 'All reviews deleted'
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/export', methods=['GET'])
def export_data():
    """Export all reviews to JSON file."""
    try:
        db.export_to_json('sentiment_export.json')

        return jsonify({
            'success': True,
            'message': 'Data exported to sentiment_export.json'
        }), 200

    except Exception as e:
        logger.error(f"Error in export_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


# === ROOT ENDPOINT ===
@app.route('/', methods=['GET'])
def root():
    """API documentation."""
    return jsonify({
        'api_name': 'Sentiment Analysis API',
        'version': '1.0.0',
        'endpoints': {
            'health': 'GET /health',
            'analyze': 'POST /api/v1/analyze',
            'batch_analyze': 'POST /api/v1/batch-analyze',
            'get_review': 'GET /api/v1/reviews/<id>',
            'list_reviews': 'GET /api/v1/reviews?limit=50&offset=0',
            'get_by_sentiment': 'GET /api/v1/reviews/sentiment/<sentiment>',
            'get_by_username': 'GET /api/v1/reviews/user/<username>',
            'search': 'GET /api/v1/search?q=keyword',
            'statistics': 'GET /api/v1/statistics?days=30',
            'trends': 'GET /api/v1/trends?days=30',
            'database_info': 'GET /api/v1/info',
            'export': 'GET /api/v1/export'
        }
    }), 200

@app.route('/openapi.json', methods=['GET'])
def openapi_spec():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Sentiment Analysis API",
            "version": "1.0.0"
        },
        "paths": {
            "/api/v1/analyze": {
                "post": {
                    "summary": "Analyze a single review",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "username": {"type": "string"},
                                        "domain": {"type": "string"},
                                        "save_to_db": {"type": "boolean"}
                                    },
                                    "required": ["text"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Analysis result"},
                        "400": {"description": "Bad request"}
                    }
                }
            },
            "/api/v1/batch-analyze": {
                "post": {
                    "summary": "Analyze multiple reviews",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "reviews": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "text": {"type": "string"},
                                                    "username": {"type": "string"},
                                                    "domain": {"type": "string"}
                                                },
                                                "required": ["text"]
                                            }
                                        },
                                        "save_to_db": {"type": "boolean"}
                                    },
                                    "required": ["reviews"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {"description": "Batch analysis result"},
                        "400": {"description": "Bad request"}
                    }
                }
            },
            "/api/v1/reviews": {
                "get": {
                    "summary": "List reviews with pagination",
                    "parameters": [
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                        {"name": "offset", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {"description": "List of reviews"}
                    }
                }
            },
            "/api/v1/search": {
                "get": {
                    "summary": "Search reviews by keyword",
                    "parameters": [
                        {"name": "q", "in": "query", "required": True, "schema": {"type": "string"}},
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}}
                    ],
                    "responses": {
                        "200": {"description": "Search results"},
                        "400": {"description": "Missing query"}
                    }
                }
            }
        }
    }
    return jsonify(spec), 200

@app.route('/docs', methods=['GET'])
def docs():
    return jsonify({
        "message": "See /openapi.json for OpenAPI spec. You can load it into Swagger UI, Postman, or any client generator."
    }), 200

if __name__ == '__main__':
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )