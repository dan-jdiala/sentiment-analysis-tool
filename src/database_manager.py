"""
Database Manager for Sentiment Analysis System
Handles persistent storage of reviews and analysis results in SQLite
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager


class SentimentDatabase:
    """Manages all database operations for sentiment analysis."""

    def __init__(self, db_path: str = "sentiment_analysis.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Create tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Reviews table
            cursor.execute('''
                    CREATE TABLE IF NOT EXISTS reviews (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        text TEXT NOT NULL,
                        sentiment TEXT NOT NULL,
                        pos_score INTEGER NOT NULL,
                        neg_score INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        domain TEXT DEFAULT 'general',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                           )
                           ''')

            # Aspects table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS aspects
                           (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               review_id INTEGER NOT NULL,
                               aspect_name TEXT NOT NULL,
                               sentiment TEXT NOT NULL,
                               score INTEGER NOT NULL,
                               FOREIGN KEY (review_id) REFERENCES reviews (id) ON DELETE CASCADE
                           )
                           ''')

            # Sarcasm detection table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS sarcasm_flags
                           (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               review_id INTEGER NOT NULL,
                               is_sarcastic INTEGER NOT NULL,
                               confidence REAL NOT NULL,
                               sarcasm_type TEXT,
                               FOREIGN KEY (review_id) REFERENCES reviews (id) ON DELETE CASCADE
                           )
                           ''')

            # Temporal tracking table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS temporal_data
                           (
                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                               review_id INTEGER NOT NULL,
                               date_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                               sentiment_trend TEXT,
                               FOREIGN KEY (review_id) REFERENCES reviews (id) ON DELETE CASCADE
                           )
                           ''')

            # Create indices for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_username ON reviews(username)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON reviews(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_aspect_review ON aspects(review_id)')

    def save_review(self, username: str, text: str, sentiment: str,
                    pos_score: int, neg_score: int, confidence: float,
                    aspect_results: Dict = None, sarcasm_data: Tuple = None,
                    domain: str = 'general') -> int:
        """
        Save a review and its analysis to the database.
        Returns the review ID.
        """

        aspect_results = aspect_results or {}

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Insert review - 7 columns, 7 values
            cursor.execute('''
                           INSERT INTO reviews (username, text, sentiment, pos_score, neg_score, confidence, domain)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ''', (username, text, sentiment, pos_score, neg_score, confidence, domain))

            review_id = cursor.lastrowid

            # Insert aspects - FIX: Handle both dict and None properly
            if isinstance(aspect_results, dict):
                for aspect, aspect_data in (aspect_results or {}).items():
                    try:
                        # Handle different formats
                        if isinstance(aspect_data, dict):
                            # Format: {"sentiment": "POSITIVE", "score": 2}
                            asp_sentiment = aspect_data.get('sentiment', 'NEUTRAL')
                            asp_score = aspect_data.get('score', 0)
                        elif isinstance(aspect_data, (list, tuple)) and len(aspect_data) >= 2:
                            # Format: ("POSITIVE", 2)
                            asp_sentiment = aspect_data[0]
                            asp_score = aspect_data[1]
                        else:
                            # Skip invalid format
                            continue

                        cursor.execute('''
                                        INSERT INTO aspects (review_id, aspect_name, sentiment, score)
                                        VALUES (?, ?, ?, ?)
                                        ''', (review_id, aspect, asp_sentiment, int(asp_score)))
                    except Exception as e:
                        print(f"Error inserting aspect {aspect}: {e}")
                        continue

            # Insert sarcasm data if present
            if sarcasm_data:
                try:
                    is_sarcastic, sarc_confidence, sarc_type = sarcasm_data
                    cursor.execute('''
                                   INSERT INTO sarcasm_flags (review_id, is_sarcastic, confidence, sarcasm_type)
                                   VALUES (?, ?, ?, ?)
                                   ''', (review_id, int(is_sarcastic), sarc_confidence, sarc_type))
                except Exception as e:
                    print(f"Error inserting sarcasm data: {e}")

            return review_id

    def save_reviews_bulk(self, results):
        """
        Save multiple reviews AND their aspects in a single batch operation.

        - Validates data before saving
        - Saves reviews first, gets their IDs
        - Then saves aspect data for each review
        - Returns number of reviews actually saved
        """
        if not results:
            print("⚠️  No results to save")
            return 0

        # Validate data BEFORE saving
        valid_results = []
        invalid_count = 0

        for i, res in enumerate(results):
            # Check for required fields
            if not res.get('text') or not res.get('text').strip():
                print(f"⚠️  Result {i}: Missing or empty text")
                invalid_count += 1
                continue

            if not res.get('username'):
                print(f"⚠️  Result {i}: Missing username")
                invalid_count += 1
                continue

            if res.get('sentiment') not in ['POSITIVE', 'NEGATIVE', 'NEUTRAL', 'MIXED']:
                print(f"⚠️  Result {i}: Invalid sentiment '{res.get('sentiment')}'")
                invalid_count += 1
                continue

            # Valid result
            valid_results.append(res)

        if not valid_results:
            print(f"❌ No valid results to save (invalid: {invalid_count})")
            return 0

        # Prepare data for reviews bulk insert
        reviews_query = """INSERT INTO reviews (username, text, sentiment, pos_score, neg_score, confidence, domain)
                           VALUES (?, ?, ?, ?, ?, ?, ?)"""

        reviews_data = []

        for res in valid_results:
            try:
                reviews_data.append((
                    res.get('username', 'anonymous').strip(),
                    res.get('text', '').strip(),
                    res.get('sentiment', 'NEUTRAL'),
                    int(res.get('pos_score', 0)),
                    int(res.get('neg_score', 0)),
                    float(res.get('confidence', 0.0)),
                    res.get('domain', 'general')
                ))
            except Exception as e:
                print(f"❌ Error preparing review: {e}")
                invalid_count += 1
                continue

        if not reviews_data:
            print(f"❌ Could not prepare any reviews for saving")
            return 0

        # Now save REVIEWS + ASPECTS
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Step 1: Save all reviews in bulk
                cursor.executemany(reviews_query, reviews_data)
                saved_count = cursor.rowcount

                # Step 2: Get the IDs of the reviews we just inserted
                cursor.execute('''
                               SELECT id FROM reviews 
                               ORDER BY id DESC 
                               LIMIT ?
                               ''', (saved_count,))

                review_ids = [row[0] for row in reversed(cursor.fetchall())]

                # Step 3: Save aspects for each review
                aspects_query = """INSERT INTO aspects (review_id, aspect_name, sentiment, score)
                                   VALUES (?, ?, ?, ?)"""

                aspects_inserted = 0

                for review_idx, res in enumerate(valid_results):
                    if review_idx >= len(review_ids):
                        break

                    review_id = review_ids[review_idx]
                    aspects = res.get('aspects', {})

                    # Insert each aspect
                    for aspect_name, aspect_data in (aspects or {}).items():
                        try:
                            # Handle different aspect data formats
                            if isinstance(aspect_data, dict):
                                asp_sentiment = aspect_data.get('sentiment', 'NEUTRAL')
                                asp_score = aspect_data.get('score', 0)
                            elif isinstance(aspect_data, (list, tuple)) and len(aspect_data) >= 2:
                                asp_sentiment = aspect_data[0]
                                asp_score = aspect_data[1]
                            else:
                                continue

                            cursor.execute(aspects_query, (
                                review_id,
                                aspect_name,
                                asp_sentiment,
                                int(asp_score)
                            ))
                            aspects_inserted += 1

                        except Exception as e:
                            print(f"  ⚠️  Error inserting aspect '{aspect_name}' for review {review_id}: {e}")
                            continue

                print(f"✅ Successfully saved {saved_count} reviews ({invalid_count} invalid skipped)")
                print(f"✅ Saved {aspects_inserted} aspect records")
                return saved_count

            except Exception as e:
                print(f"❌ Database error during bulk save: {e}")
                return 0

    def get_review(self, review_id: int) -> Optional[Dict]:
        """Retrieve a single review by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM reviews WHERE id = ?', (review_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_reviews(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all reviews with pagination."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           SELECT *
                           FROM reviews
                           ORDER BY created_at DESC LIMIT ?
                           OFFSET ?
                           ''', (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def get_reviews_by_sentiment(self, sentiment: str) -> List[Dict]:
        """Get all reviews with a specific sentiment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           SELECT *
                           FROM reviews
                           WHERE sentiment = ?
                           ORDER BY created_at DESC
                           ''', (sentiment,))
            return [dict(row) for row in cursor.fetchall()]

    def get_reviews_by_username(self, username: str) -> List[Dict]:
        """Get all reviews from a specific user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           SELECT *
                           FROM reviews
                           WHERE username = ?
                           ORDER BY created_at DESC
                           ''', (username,))
            return [dict(row) for row in cursor.fetchall()]

    def get_aspect_overview(self) -> List[Dict]:
        """Get aggregated aspect sentiment stats across all reviews."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           SELECT aspect_name,
                                  sentiment,
                                  COUNT(*) AS count,
                                  AVG(score) AS avg_score
                           FROM aspects
                           GROUP BY aspect_name, sentiment
                           ORDER BY aspect_name, sentiment
                           ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self, days: int = 30) -> Dict:
        """Get sentiment statistics for the last N days."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Overall statistics
            cursor.execute('''
                           SELECT COUNT(*)                                                as total,
                                  SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive,
                                  SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative,
                                  SUM(CASE WHEN sentiment = 'NEUTRAL' THEN 1 ELSE 0 END)  as neutral,
                                  AVG(confidence)                                         as avg_confidence,
                                  AVG(pos_score)                                          as avg_pos_score,
                                  AVG(neg_score)                                          as avg_neg_score
                           FROM reviews
                           WHERE created_at >= datetime('now', '-' || ? || ' days')
                           ''', (days,))

            stats = dict(cursor.fetchone())

            # Aspect breakdown
            cursor.execute('''
                           SELECT aspect_name,
                                  sentiment,
                                  COUNT(*) as count,
                                  AVG(score) as avg_score
                           FROM aspects
                           WHERE review_id IN (
                               SELECT id FROM reviews
                               WHERE created_at >= datetime('now', '-' || ? || ' days')
                               )
                           GROUP BY aspect_name, sentiment
                           ''', (days,))

            stats['aspects'] = {row[0]: {row[1]: {'count': row[2], 'avg_score': row[3]}}
                                for row in cursor.fetchall()}

            return stats

    def get_trend(self, days: int = 30) -> List[Dict]:
        """Get daily sentiment trend."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                           SELECT DATE(created_at) as date, 
                                  sentiment, 
                                  COUNT(*) as count, 
                                  AVG(pos_score - neg_score) as net_score
                           FROM reviews
                           WHERE created_at >= datetime('now', '-' || ? || ' days')
                           GROUP BY DATE(created_at), sentiment
                           ORDER BY date DESC
                           ''', (days,))

            return [dict(row) for row in cursor.fetchall()]

    def search_reviews(self, keyword: str) -> List[Dict]:
        """Search reviews by keyword. Returns ALL matching reviews."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            search_term = f"%{keyword}%"

            cursor.execute('''
                           SELECT *
                           FROM reviews
                           WHERE text LIKE ?
                              OR username LIKE ?
                           ORDER BY created_at DESC
                           ''', (search_term, search_term))

            return [dict(row) for row in cursor.fetchall()]

    def delete_review(self, review_id: int) -> bool:
        """Delete a review and all associated data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM reviews WHERE id = ?', (review_id,))
            return cursor.rowcount > 0

    def delete_all_reviews(self) -> bool:
        """Delete ALL reviews and related data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM aspects")
            cursor.execute("DELETE FROM sarcasm_flags")
            cursor.execute("DELETE FROM temporal_data")
            cursor.execute("DELETE FROM reviews")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='reviews'")
            return True

    def update_review(self, review_id: int, **kwargs) -> bool:
        """Update review fields."""
        allowed_fields = {'sentiment', 'pos_score', 'neg_score', 'confidence', 'domain'}
        fields = {k: v for k, v in kwargs.items() if k in allowed_fields}

        if not fields:
            return False

        set_clause = ', '.join([f'{k} = ?' for k in fields.keys()])
        values = list(fields.values()) + [review_id]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE reviews 
                SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', values)
            return cursor.rowcount > 0

    def export_to_json(self, output_file: str = "sentiment_analysis_export.json"):
        """Export all reviews to JSON."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM reviews')
            reviews = [dict(row) for row in cursor.fetchall()]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(reviews, f, indent=2, default=str)

        print(f"✓ Exported {len(reviews)} reviews to {output_file}")

    def get_database_stats(self) -> Dict:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM reviews')
            total_reviews = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(DISTINCT username) FROM reviews')
            unique_users = cursor.fetchone()[0]

            cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM reviews')
            first_review, last_review = cursor.fetchone()

            return {
                'total_reviews': total_reviews,
                'unique_users': unique_users,
                'first_review': first_review,
                'last_review': last_review,
                'database_file': self.db_path
            }