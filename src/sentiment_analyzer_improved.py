"""
PATCHED SENTIMENT ANALYZER - Fixed Critical Issues

Issues Fixed:
1. ✅ find_closest_word() - Fixed type hint (Optional not imported)
2. ✅ batch_sentiment_analysis() - Fixed nlp.pipe syntax
3. ✅ sentiment_analysis() - Incomplete decision logic
4. ✅ early_is_mixed undefined error
5. ✅ Explicit neutral check moved to correct position
"""

from typing import Optional, Set, List, Dict, Tuple
import os
import csv
import re
import spacy
import enchant
from datetime import datetime
from difflib import SequenceMatcher, get_close_matches
import time
from functools import wraps

from src.context_intensifiers import IntensifierHandler, CONTEXT_INTENSIFIERS, INTENSIFIERS
from improved_phrase_adjustments import apply_phrase_adjustments
from internet_slang import apply_slang_adjustments, detect_internet_slang
from domain_lexicons import DomainLexicon, Domain

ENABLE_REASONING_TRACE = False
FUZZY_CACHE = {}
ALL_SENTIMENT_WORDS = None
_domain_lex_cache = {}
_lemma_cache = {}

# Load spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # Add sentencizer for sentence detection (lightweight, no parsing needed)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Load enchant
try:
    dictionary = enchant.Dict("en_US")
    ENCHANT_AVAILABLE = True
except (ImportError, enchant.Error):
    print("Warning: enchant not available. Fuzzy matching will be limited.")
    dictionary = None
    ENCHANT_AVAILABLE = False

# === DICTIONARIES ===
SARCASM_QUALIFIERS = {"yeah", "sure", "right", "seriously", "yeah right"}

OPINION_MARKERS = {
    "i think", "i feel", "i believe", "i found", "i consider", "i felt",
    "in my opinion", "in my view", "imho", "i think that",
    "seems to", "appears to", "looks like", "seems like",
    "i love", "i hate", "i like", "i dislike", "i enjoyed", "i didn't enjoy",
    "personally", "subjectively", "imo", "if you ask me"
}

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "won't": "will not", "wouldn't": "would not", "can't": "cannot",
    "couldn't": "could not", "shouldn't": "should not", "haven't": "have not",
    "hasn't": "has not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "it's": "it is",
    "that's": "that is", "what's": "what is", "who's": "who is",
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would",
}

# INTENSIFIERS = {
#     "super", "very", "extremely", "incredibly", "absolutely", "truly", "really",
#     "so", "quite", "exceptionally", "remarkably", "particularly", "especially",
#     "highly", "deeply", "thoroughly", "utterly", "completely", "absolutely"
# }

DIMINISHERS = {
    "somewhat", "kind", "a bit", "slightly", "fairly", "sort of",
    "moderately", "reasonably", "relatively", "kinda", "pretty"
}

NEUTRAL_WORDS = {
    "okay", "fine", "average", "decent", "alright", "acceptable",
    "so-so", "middling", "mediocre", "unremarkable", "ordinary", "standard",
    "normal", "regular", "usual", "typical", "ok", "reasonable",
    "satisfactory", "adequate", "sufficient", "tolerable", "passable",
    "moderate", "fair", "neutral", "indifferent", "lukewarm", "half-hearted",
    "pretty", "functional", "works", "working", "quality", "shipping", "service",
    "product", "design", "support", "experience"
}

DEAL_BREAKER_KEYWORDS = {
    "battery", "crash", "broken", "defective", "unsafe", "malware", "exploit",
    "dangerous", "fails", "failure", "stopped", "won't work", "doesn't work"
}

POSITIVE_NEGATION_PHRASES = {
    "can't say i regret", "no regrets", "don't regret", "not sorry",
    "wouldn't change", "can't complain",
}

EXPLICIT_NEUTRAL_PHRASES = {
    "okay i guess", "not bad but not great", "not bad but not great either",
    "good but not great", "pretty average", "kind of average", "so-so",
    "hit or miss", "mixed bag", "neither good nor bad", "fairly standard",
    "nothing special", "average experience", "mixed feelings", "nothing to rave about",
    "can't complain", "it's okay", "it's fine", "works fine",
    "does what it's supposed to", "it's decent", "pretty decent",
    "fairly average", "it's functional", "nothing exceptional"
}

CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "still", "nonetheless"}

NEGATORS = {
    "not", "no", "never", "n't", "cannot", "could", "won", "should", "nothing",
    "have", "has", "hardly", "barely", "scarcely", "without", "neither", "nor",
}

NON_SENTIMENT_WORDS = {
    "but", "however", "although", "though", "yet", "still", "nonetheless",
    "try", "service", "food", "performance", "restaurant", "dish", "plate",
    "wanted", "did", "server", "honesty", "desert", "dessert", "there",
    "their", "they're", "its", "it's", "your", "you're", "go", "went",
    "come", "came", "make", "made", "get", "got", "have", "had", "take", "took",
}

SENTIMENT_VERBS = {
    "love", "adore", "enjoy", "appreciate", "cherish", "treasure", "revere", "savor", "admire",
    "hate", "despise", "regret", "loathe", "detest", "dislike", "resent", "abhor",
    "broke", "crash", "failed", "break", "fail", "stuck", "froze", "freeze", "stopped", "stop", "jammed", "jam",
    "works_well", "functions", "performs"
}

NON_SENTIMENT_VERBS = {
    "be", "is", "are", "was", "were", "am", "go", "went", "come", "came",
    "make", "made", "do", "did", "get", "got", "have", "had", "say", "said",
    "tell", "told", "take", "took", "give", "gave",
}

EMOJI_SENTIMENT = {
    "😍": 3, "🥰": 3, "😂": 2, "🤣": 2, "😊": 2, "😎": 2, "👍": 2,
    "❤️": 3, "💕": 3, "🎉": 3, "✨": 2, "⭐": 2, "🔥": 2,
    "😡": -3, "😠": -2, "😤": -1, "💔": -3, "👎": -2, "💩": -3,
}

SARCASM_EMOJIS = {"🤡": True}

ASPECTS = {
    "food": {"keywords": {"food", "dish", "meal", "appetizer", "entree", "main", "course", "dessert", "breakfast", "lunch", "dinner", "sauce", "taste", "flavor", "spice", "recipe", "cook", "cuisine"}, "sentiment": [], "weight": 1.0},
    "service": {"keywords": {"service", "server", "waiter", "waitress", "staff", "attendant", "worker", "host", "hostess", "manager", "customer_service", "rep", "support", "help"}, "sentiment": [], "weight": 1.0},
    "shipping": {"keywords": {"shipping", "delivery", "package", "arrived", "arrive", "delivered", "delivery_time", "shipping_time", "packaging", "box", "condition", "damage", "damaged", "broken", "dent", "crease", "fold"}, "sentiment": [], "weight": 1.1},
    "atmosphere": {"keywords": {"atmosphere", "ambiance", "decor", "decoration", "setting", "environment", "room", "space", "music", "lighting", "light", "noise"}, "sentiment": [], "weight": 0.8},
    "value": {"keywords": {"price", "cost", "expensive", "cheap", "affordable", "worth", "value", "money", "bill", "charge", "fee", "overpriced", "deal", "pricing", "priced", "wallet"}, "sentiment": [], "weight": 0.9},
    "quality": {"keywords": {"quality", "product", "item", "materials", "material", "durability", "durable", "sturdy", "flimsy", "build"}, "sentiment": [], "weight": 1.2},
    "experience": {"keywords": {"experience", "visit", "stay", "trip", "journey", "overall", "impression", "feeling", "vibe", "enjoyment"}, "sentiment": [], "weight": 1.1},
    "performance": {"keywords": {"performance", "speed", "fast", "slow", "smooth", "lag", "crash", "reliable", "unreliable", "battery", "power", "responsive", "control", "controls", "gameplay", "mechanics", "frame", "fps", "render", "stutter", "frame rate", "combat", "unbalanced"}, "sentiment": [], "weight": 1.3},
    "design": {"keywords": {"design", "aesthetic", "visual", "graphic", "layout", "interface", "ui", "ux", "usability", "world", "map", "character", "visuals", "graphics"}, "sentiment": [], "weight": 1.2}
}

_timing_stats = {
    'is_explicitly_neutral': [],
    'nlp_processing': [],
    'word_analysis': [],
    'aspect_analysis': [],
    'mixed_detection': [],
    'total': []
}

# === UTILITY FUNCTIONS ===
def debug_print(debug: bool, *args) -> None:
    """Print only when debug mode is enabled."""
    if debug:
        print(*args)

def get_domain_lexicon(domain: str = "general"):
    """Cache domain lexicon to avoid re-instantiation."""
    if domain not in _domain_lex_cache:
        _domain_lex_cache[domain] = DomainLexicon()
    return _domain_lex_cache[domain]


def lemmatize_word_cached(word: str) -> str:
    """Lemmatize word with caching."""
    word_lower = word.lower()

    if word_lower in _lemma_cache:
        return _lemma_cache[word_lower]

    if nlp is None:
        result = word_lower
    else:
        doc = nlp(word_lower)
        result = doc[0].lemma_ if len(doc) > 0 else word_lower

    _lemma_cache[word_lower] = result
    return result

def find_closest_word_fast(word: str, word_set: Set[str], threshold: float = 0.85) -> Optional[str]:
    """Optimized fuzzy matching with early exits."""
    if not word:
        return None

    word_lower = word.lower()

    #Don't fuzzy match these common context words
    context_words = {'quality', 'product', 'service', 'shipping', 'delivery'}
    if word_lower in context_words:
        FUZZY_CACHE[word_lower] = None
        return None

    # Fuzzy matching is for typos ("amazng"). A correctly spelled word with no sentiment of its
    # own must not borrow one from a look-alike ("price" -> "pricey", "honestly" -> "honest").
    if dictionary is not None and dictionary.check(word_lower):
        FUZZY_CACHE[word_lower] = None
        return None

    # Check cache first
    if word_lower in FUZZY_CACHE:
        return FUZZY_CACHE[word_lower]

    # Quick checks
    if len(word_lower) < 3:
        FUZZY_CACHE[word_lower] = None
        return None

    if word_lower in NON_SENTIMENT_WORDS or word_lower in DIMINISHERS or word_lower in INTENSIFIERS:
        FUZZY_CACHE[word_lower] = None
        return None

    # Exact match
    if word_lower in word_set:
        FUZZY_CACHE[word_lower] = word_lower
        return word_lower

    # Only do expensive fuzzy matching if word is common enough
    if len(word_lower) >= 4:  # Only fuzzy match longer words
        matches = get_close_matches(word_lower, word_set, n=1, cutoff=threshold)
        result = matches[0] if matches else None
    else:
        result = None

    FUZZY_CACHE[word_lower] = result
    return result


def _batch_lemmatize(words: List[str]) -> Dict[str, str]:
    """Lemmatize multiple words in one pass."""
    if nlp is None or not words:
        return {w: w.lower() for w in words}

    # Only process words not in cache
    uncached = [w for w in words if w.lower() not in _lemma_cache]

    if not uncached:
        return {w: _lemma_cache[w.lower()] for w in words}

    # Batch process with spaCy
    docs = nlp.pipe(uncached, batch_size=200)
    results = {}

    for word, doc in zip(uncached, docs):
        lemma = doc[0].lemma_ if len(doc) > 0 else word.lower()
        _lemma_cache[word.lower()] = lemma
        results[word.lower()] = lemma

    # Add cached ones
    for w in words:
        if w.lower() not in results:
            results[w.lower()] = _lemma_cache[w.lower()]

    return results

def time_operation(operation_name: str):
    """Decorator to time specific operations"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

            if operation_name in _timing_stats:
                _timing_stats[operation_name].append(elapsed)

            return result

        return wrapper

    return decorator


def print_timing_stats(sample_size: int = 50):
    """Print timing statistics after running batch analysis"""
    print("\n" + "=" * 70)
    print("⏱️  PERFORMANCE PROFILING RESULTS")
    print("=" * 70)

    for operation, times in _timing_stats.items():
        if not times:
            continue

        avg = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)

        print(f"\n{operation.upper()}")
        print(f"  Average: {avg:.2f}ms")
        print(f"  Min:     {min_t:.2f}ms")
        print(f"  Max:     {max_t:.2f}ms")
        print(f"  Total:   {sum(times):.2f}ms for {len(times)} reviews")

    if _timing_stats['total']:
        total_avg = sum(_timing_stats['total']) / len(_timing_stats['total'])
        print(f"\n{'TOTAL PER REVIEW'.upper()}")
        print(f"  Average: {total_avg:.2f}ms")
        print(f"  Expected: 50-200ms")
        if total_avg > 500:
            print(f"  ⚠️  TOO SLOW: {total_avg / 100:.0f}x slower than expected!")


def reset_timing_stats():
    """Reset timing stats for new run"""
    global _timing_stats
    for key in _timing_stats:
        _timing_stats[key] = []


def get_timing_stats_summary() -> Dict:
    """
    Convert timing stats to a summary dict for API response.
    Returns averages, totals, and performance assessment.
    """
    summary = {
        'total_reviews_analyzed': len(_timing_stats['total']),
        'total_time_ms': sum(_timing_stats['total']),
        'avg_time_per_review_ms': 0,
        'operations': {}
    }

    if _timing_stats['total']:
        summary['avg_time_per_review_ms'] = (
                sum(_timing_stats['total']) / len(_timing_stats['total'])
        )

    # Get averages for each operation
    for op_name, times in _timing_stats.items():
        if not times or op_name == 'total':
            continue

        summary['operations'][op_name] = {
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'total_ms': sum(times),
            'count': len(times)
        }

    # Assessment
    avg = summary['avg_time_per_review_ms']
    if avg < 100:
        summary['performance'] = 'EXCELLENT ⚡'
    elif avg < 200:
        summary['performance'] = 'GOOD ✓'
    elif avg < 500:
        summary['performance'] = 'ACCEPTABLE ⚠️'
    else:
        summary['performance'] = 'SLOW ❌'

    return summary

# Idioms whose meaning differs from their words (e.g. "steal" is negative, "a steal" is a bargain).
# Weight replaces the general-lexicon weight of the words inside the phrase. Applies to every domain.
IDIOMS = {
    "steal of a deal": 3,
    "a total steal": 3,
    "an absolute steal": 3,
    "a steal": 2,
    "killer feature": 2,
    "killer features": 2,
}

# A negation just before an idiom flips it ("not a steal at all" is negative).
IDIOM_NEGATORS = {"not", "never", "hardly", "no", "isn't", "wasn't", "aren't", "weren't", "isnt", "wasnt"}

_domain_lexicon_instance = None


def _get_domain_lexicon() -> DomainLexicon:
    """Single shared DomainLexicon (building one per review was the original slowdown)."""
    global _domain_lexicon_instance
    if _domain_lexicon_instance is None:
        _domain_lexicon_instance = DomainLexicon()
    return _domain_lexicon_instance


def domain_and_idiom_adjustment(raw_text: str, domain: str = "general") -> Tuple[int, int, List[str]]:
    """
    Re-weight terms whose sentiment depends on context, before the final label is decided.

    For each idiom, and for each domain-specific term (a word or phrase from the chosen domain's
    lexicon that is not in the general one), remove the general-lexicon weight of its words and
    apply the context weight instead. Longest terms match first, and each span counts once.

    Returns (pos_delta, neg_delta, matched_terms); deltas are applied to the positive and
    negative score magnitudes.
    """
    text = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s'-]", " ", raw_text.lower())) + " "

    terms = [(phrase, weight) for phrase, weight in IDIOMS.items()]
    idiom_set = set(IDIOMS)
    try:
        domain_enum = Domain((domain or "general").lower())
    except ValueError:
        domain_enum = Domain.GENERAL
    if domain_enum != Domain.GENERAL:
        lexicons = _get_domain_lexicon().lexicons
        general_terms = set().union(*lexicons[Domain.GENERAL].values())
        for weight, words in lexicons[domain_enum].items():
            terms.extend((term, weight) for term in words if term not in general_terms)

    # Longest phrases first; for duplicates, the stronger weight wins.
    terms.sort(key=lambda t: (-len(t[0]), -abs(t[1])))

    pos_delta = neg_delta = 0
    matched = []
    for term, weight in terms:
        needle = f" {term} "
        while needle in text:
            negated = term in idiom_set and any(
                w in IDIOM_NEGATORS for w in text[: text.index(needle)].split()[-2:]
            )
            if negated:
                # The main pass already flipped these words under the negation, so their
                # weight may sit on either side; remove it from the positive side and score
                # the idiom as negative.
                weight = -abs(weight)
                pos_delta -= sum(abs(get_word_points_with_pos(t)) for t in term.split())
            else:
                for token in term.split():
                    base = get_word_points_with_pos(token)
                    if base > 0:
                        pos_delta -= base
                    elif base < 0:
                        neg_delta -= -base
            if weight > 0:
                pos_delta += weight
            elif weight < 0:
                neg_delta += -weight
            matched.append(term)
            text = text.replace(needle, " _ ", 1)

    return pos_delta, neg_delta, matched


def _init_sentiment_words():
    """Initialize all sentiment words set once at module load."""
    global ALL_SENTIMENT_WORDS
    if ALL_SENTIMENT_WORDS is None:
        ALL_SENTIMENT_WORDS = pos_mild | pos_medium | pos_strong | neg_mild | neg_medium | neg_strong
    return ALL_SENTIMENT_WORDS

def load_words(file_path: str) -> Set[str]:
    """Load words from a file, one word per line."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return set()

# Load sentiment lexicons
pos_mild = load_words("../lexicons/general/pos_mild")
pos_medium = load_words("../lexicons/general/pos_medium")
pos_strong = load_words("../lexicons/general/pos_strong")
neg_mild = load_words("../lexicons/general/neg_mild")
neg_medium = load_words("../lexicons/general/neg_medium")
neg_strong = load_words("../lexicons/general/neg_strong")

# Call this once when module loads
_init_sentiment_words()

# === CORE FUNCTIONS ===
def get_emoji_sentiment(char: str) -> int:
    """Get sentiment score for emoji."""
    return EMOJI_SENTIMENT.get(char, 0)

def expand_emojis_to_text(text: str) -> str:
    """Convert emojis to descriptive text."""
    emoji_map = {
        "😍": "love beautiful gorgeous", "🥰": "love beautiful wonderful",
        "😂": "funny hilarious", "🤣": "funny hilarious", "😎": "cool awesome",
        "👍": "good positive", "❤️": "love beautiful", "💕": "love beautiful",
        "🎉": "celebrate amazing", "✨": "amazing wonderful", "⭐": "excellent",
        "😡": "angry terrible", "😠": "angry bad", "💔": "sad broken",
        "👎": "bad terrible", "💩": "terrible awful",
    }
    expanded = text
    for emoji, desc in emoji_map.items():
        if emoji in expanded:
            expanded = expanded.replace(emoji, f" {desc} ")
    return expanded

def detect_sarcasm_emojis(text: str, debug: bool = False) -> Tuple[bool, float]:
    """Detect sarcasm emojis that affect sentiment."""
    has_sarcasm = False
    confidence = 0.0
    for emoji in SARCASM_EMOJIS:
        if emoji in text:
            has_sarcasm = True
            confidence = 0.85
            debug_print(debug, f"  [Sarcasm Emoji] Found '{emoji}'")
            break
    return has_sarcasm, confidence

def expand_contractions(text: str) -> str:
    """Expand contractions in text."""
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text)
    return text

def lemmatize_word(word: str) -> str:
    """Lemmatize a word using spaCy."""
    if nlp is None:
        return word.lower()
    doc = nlp(word.lower())
    return doc[0].lemma_ if len(doc) > 0 else word.lower()

def find_closest_word(word: str, word_set: Set[str], threshold: float = 0.85) -> Optional[str]:
    """Finds the closest match in word_set using fuzzy matching."""
    if not word:
        return None

    word_lower = word.lower()

    if word_lower in FUZZY_CACHE:
        return FUZZY_CACHE[word_lower]

    if len(word_lower) < 3:
        return None

    if word_lower in NON_SENTIMENT_WORDS or word_lower in DIMINISHERS or word_lower in INTENSIFIERS:
        return None

    if word_lower in word_set:
        return word_lower

    matches = get_close_matches(word_lower, word_set, n=1, cutoff=threshold)
    result = matches[0] if matches else None

    FUZZY_CACHE[word_lower] = result
    return result

def get_word_points_with_pos(word: str, nlp_doc=None) -> int:
    word = word.lower().strip()

    if not word:
        return 0

    if word in NEUTRAL_WORDS:
        return 0

    lemmatized = lemmatize_word(word)
    if lemmatized in NEUTRAL_WORDS:
        return 0

    if word in pos_strong: return 3
    if word in pos_medium: return 2
    if word in pos_mild: return 1
    if word in neg_strong: return -3
    if word in neg_medium: return -2
    if word in neg_mild: return -1

    if lemmatized != word:
        if lemmatized in pos_strong: return 3
        if lemmatized in pos_medium: return 2
        if lemmatized in pos_mild: return 1
        if lemmatized in neg_strong: return -3
        if lemmatized in neg_medium: return -2
        if lemmatized in neg_mild: return -1

    if nlp_doc is not None:
        try:
            for token in nlp_doc:
                if token.text.lower() == word:
                    if token.pos_ not in ["ADJ", "ADV", "VERB"]:
                        return 0
                    if token.pos_ == "VERB" and word in NON_SENTIMENT_VERBS:
                        return 0
                    break
        except:
            pass

    return 0

def clean_input(text: str) -> List[str]:
    """Tokenize and clean input text."""
    stop_words = {"the", "is", "a", "of", "and", "in", "to", "with", "on", "at", "by"}
    text = expand_contractions(text)
    text = text.lower()
    for punc in [".", ",", "!", "?", ";", ":", '"', "'", "-", "—"]:
        text = text.replace(punc, ' ')

    words = text.split()
    cleaned = []
    for word in words:
        word = word.strip()
        if word.isalpha() and word not in stop_words:
            lemmatized = lemmatize_word(word)
            cleaned.append(lemmatized)
    return cleaned

def is_explicitly_neutral(text: str) -> bool:
    """Check if review explicitly states it's neutral."""
    text_lower = text.lower()
    for phrase in EXPLICIT_NEUTRAL_PHRASES:
        if phrase in text_lower:
            return True
    return False

def has_critical_issue(text: str, aspect_results: Dict) -> bool:
    """Check if review mentions critical/deal-breaker issues."""
    text_lower = text.lower()
    for aspect, (sentiment, score) in aspect_results.items():
        if sentiment == "NEGATIVE":
            for keyword in DEAL_BREAKER_KEYWORDS:
                if keyword in text_lower:
                    return True
    return False

# === PHRASE ANALYSIS ===
class PhraseAnalyzer:
    """Analyzes multi-word phrases for better sentiment understanding."""

    @staticmethod
    def find_contrast_point(sentence: str) -> int:
        words = sentence.lower().split()
        for i, word in enumerate(words):
            if word in CONTRAST_WORDS:
                return i
        return -1

    @staticmethod
    def is_negated(word_index: int, words: List[str]) -> bool:
        if word_index == 0:
            return False
        for i in range(max(0, word_index - 2), word_index):
            if words[i] in NEGATORS:
                return True
        return False

    @staticmethod
    def find_comparison_point(sentence: str) -> int:
        comparison_words = {"better", "worse", "superior", "inferior", "greater", "lesser", "more", "less", "best", "worst"}
        words = sentence.lower().split()
        for i, word in enumerate(words):
            if word in comparison_words:
                return i
        return -1

    @staticmethod
    def analyze_comparison_phrase(sentence: str, debug: bool = False) -> Tuple[int, int]:
        words = sentence.lower().split()
        comp_idx = PhraseAnalyzer.find_comparison_point(sentence)

        if comp_idx == -1:
            return 0, 0

        comparison_word = words[comp_idx]
        debug_print(debug, f"  [Comparison Analysis] '{sentence.strip()}'")

        positive_comparisons = {"better", "superior", "greater", "best"}
        negative_comparisons = {"worse", "inferior", "lesser", "worst", "less"}
        boost_amount = 2

        if comparison_word in positive_comparisons:
            start_idx = max(0, comp_idx - 2)
            preceding_words = words[start_idx:comp_idx]
            negating_context = {"seen", "known", "had", "used", "expected"}

            if any(w in preceding_words for w in negating_context):
                debug_print(debug, f"    Context: '{' '.join(preceding_words)} {comparison_word}' detected → Neutralizing")
                return 0, 0

            debug_print(debug, f"    Type: Positive comparison → +{boost_amount} positive boost")
            return boost_amount, 0

        elif comparison_word in negative_comparisons:
            debug_print(debug, f"    Type: Negative comparison → +{boost_amount} negative boost")
            return 0, boost_amount

        return 0, 0

    @staticmethod
    def get_pre_contrast_sentiment(words: List[str], contrast_idx: int) -> Tuple[int, int]:
        pos, neg = 0, 0
        for i in range(contrast_idx):
            word = words[i]
            if word in NEGATORS:
                continue
            points = get_word_points_with_pos(word)

            is_negated = PhraseAnalyzer.is_negated(i, words)
            if is_negated:
                if points < -1:
                    points = 1
                else:
                    points = -points

            if points > 0:
                pos += points
            elif points < 0:
                neg += abs(points)
        return pos, neg

    @staticmethod
    def get_post_contrast_sentiment(words: List[str], contrast_idx: int) -> Tuple[int, int]:
        pos, neg = 0, 0
        for i in range(contrast_idx + 1, len(words)):
            word = words[i]
            if word in NEGATORS:
                continue
            points = get_word_points_with_pos(word)

            is_negated = PhraseAnalyzer.is_negated(i, words)
            if is_negated:
                if points < -1:
                    points = 1
                else:
                    points = -points

            if points > 0:
                pos += points
            elif points < 0:
                neg += abs(points)
        return pos, neg

    @staticmethod
    def analyze_contrast_phrase(sentence: str, debug: bool = False) -> Tuple[int, int, bool]:
        expanded_sentence = expand_contractions(sentence)
        words = expanded_sentence.lower().split()
        contrast_idx = PhraseAnalyzer.find_contrast_point(expanded_sentence)

        if contrast_idx == -1:
            return 0, 0, False

        pre_pos, pre_neg = PhraseAnalyzer.get_pre_contrast_sentiment(words, contrast_idx)
        post_pos, post_neg = PhraseAnalyzer.get_post_contrast_sentiment(words, contrast_idx)

        debug_print(debug, f"  [Contrast Analysis] '{sentence.strip()}'")
        debug_print(debug, f"    Before '{words[contrast_idx]}': +{pre_pos}/-{pre_neg}")
        debug_print(debug, f"    After '{words[contrast_idx]}': +{post_pos}/-{post_neg}")

        has_both_pos_and_neg = (pre_pos > 0 or post_pos > 0) and (pre_neg > 0 or post_neg > 0)

        if has_both_pos_and_neg:
            debug_print(debug, f"    [Mixed Detected] Both positive and negative sentiments present")

        has_pre_negation = contrast_idx >= 1 and words[contrast_idx - 1] in NEGATORS
        has_post_negation = (contrast_idx + 1 < len(words) and words[contrast_idx + 1] in NEGATORS)

        if has_pre_negation and has_post_negation:
            debug_print(debug, f"    Type: 'not X but not Y' → Balanced neutral")
            return 0, 0, False

        elif has_pre_negation:
            final_pos = int(post_pos * 0.8)
            final_neg = int(post_neg * 0.8)
            debug_print(debug, f"    Type: 'not X but Y' → Post dominates")

        elif has_post_negation:
            final_pos = max(0, int(pre_pos * 0.5))
            final_neg = max(0, int(pre_neg * 0.5))
            debug_print(debug, f"    Type: 'X but not Y' → Balanced/Neutral")

        else:
            final_pos = int(pre_pos * 0.3) + int(post_pos * 0.7)
            final_neg = int(pre_neg * 0.3) + int(post_neg * 0.7)
            debug_print(debug, f"    Type: 'X but Y' → Post dominates (70%)")

        debug_print(debug, f"    Result: +{final_pos}/-{final_neg}")
        return final_pos, final_neg, has_both_pos_and_neg

def is_diminisher_in_context(word: str, next_word: str) -> bool:
    word_lower = word.lower()
    if word_lower == "pretty":
        if not next_word:
            return False
        next_points = get_word_points_with_pos(next_word)
        return next_points != 0
    return word_lower in DIMINISHERS

def analyze_aspects(raw_text: str, debug: bool = False) -> Dict[str, Dict]:
    """Analyze sentiment for each aspect using proximity-based scoring."""
    aspect_results = {}

    text_lower = expand_contractions(raw_text).lower()
    text_cleaned = text_lower
    for punc in [".", ",", "!", "?", ";", ":", '"', "'", "-", "—"]:
        text_cleaned = text_cleaned.replace(punc, ' ')

    words = text_cleaned.split()

    if debug:
        print(f"[Aspect Analysis] Text: '{raw_text}'")
        print(f"[Aspect Analysis] Words: {words}\n")

    aspect_positions = {}
    for aspect_name, aspect_data in ASPECTS.items():
        keywords = aspect_data["keywords"]
        positions = []
        for i, word in enumerate(words):
            if word in keywords:
                positions.append(i)
        if positions:
            aspect_positions[aspect_name] = positions
            if debug:
                print(f"[{aspect_name}] Keywords found at positions: {positions}")

    if debug:
        print()

    sentiment_word_to_aspect = {}

    for i, word in enumerate(words):
        points = get_word_points_with_pos(word, None)

        if points == 0:
            continue

        is_negated = False
        if i > 0 and words[i - 1] in NEGATORS:
            is_negated = True
        elif i > 1 and words[i - 2] in NEGATORS:
            is_negated = True

        if is_negated:
            if points < -1:
                points = 1
            else:
                points = -points
            if debug:
                print(f"   [Aspect Negation] '{word}' flipped/dampened to {points}")

        closest_aspect = None
        closest_distance = float('inf')

        for aspect_name, positions in aspect_positions.items():
            for pos in positions:
                distance = abs(i - pos)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_aspect = aspect_name

        if closest_aspect and closest_distance <= 5:
            sentiment_word_to_aspect[i] = (closest_aspect, points)
            if debug:
                print(f"[Sentiment Word] '{word}' at position {i} → '{closest_aspect}' (distance: {closest_distance}, score: {points})")

    if debug:
        print()

    for aspect_name in ASPECTS.keys():
        aspect_pos = 0
        aspect_neg = 0

        for word_pos, (assigned_aspect, points) in sentiment_word_to_aspect.items():
            if assigned_aspect == aspect_name:
                if points > 0:
                    aspect_pos += points
                else:
                    aspect_neg += abs(points)

        if aspect_pos > 0 or aspect_neg > 0:
            if aspect_pos > aspect_neg:
                aspect_sentiment = "POSITIVE"
                score = aspect_pos
            elif aspect_neg > aspect_pos:
                aspect_sentiment = "NEGATIVE"
                score = -aspect_neg
            else:
                aspect_sentiment = "NEUTRAL"
                score = 0

            if debug:
                print(f"[{aspect_name}] Result: {aspect_sentiment} (pos={aspect_pos}, neg={aspect_neg}, score={score})")

            aspect_results[aspect_name] = {
                "sentiment": aspect_sentiment,
                "score": score
            }

    if debug:
        print(f"\n[Aspect Analysis] Final results: {aspect_results}\n")

    return aspect_results


def analyze_aspects_fast(raw_text: str, debug: bool = False) -> Dict[str, Dict]:
    """
    Fast aspect analysis - keyword-based without proximity scoring.
    Takes ~50-100ms instead of 1800ms for full analysis.
    """
    text_lower = raw_text.lower()
    aspect_results = {}

    # Define aspects with keywords (simplified from full ASPECTS dict)
    aspects_simple = {
        "quality": ["quality", "durable", "material", "crafted", "stitching", "seam", "build"],
        "shipping": ["shipping", "delivery", "package", "arrived", "packaging", "box", "damaged"],
        "performance": ["performance", "speed", "fast", "slow", "lag", "crash", "reliable", "battery"],
        "design": ["design", "aesthetic", "visual", "layout", "interface", "ui", "ux"],
        "service": ["service", "support", "helpful", "responsive", "staff", "customer_service"],
        "value": ["price", "expensive", "cheap", "affordable", "worth", "value"],
        "experience": ["experience", "visit", "stay", "overall", "impression", "enjoyment"],
    }

    if debug:
        print(f"[Aspect Analysis - Fast] Analyzing: {raw_text[:100]}...")

    # Find which aspects are mentioned
    for aspect_name, keywords in aspects_simple.items():
        if any(keyword in text_lower for keyword in keywords):
            # Found this aspect - now determine sentiment for it
            sentiment, score = determine_aspect_sentiment(raw_text, aspect_name, keywords, debug)

            aspect_results[aspect_name] = {
                "sentiment": sentiment,
                "score": score
            }

            if debug:
                print(f"  [{aspect_name.upper()}] {sentiment} (score: {score})")

    return aspect_results


def determine_aspect_sentiment(raw_text: str, aspect_name: str, keywords: List[str], debug: bool = False) -> Tuple[
    str, int]:
    """
    Determine sentiment for a specific aspect.
    Looks for sentiment words CLOSEST to aspect keywords.
    Properly handles negation and diminishers.
    """
    text_lower = raw_text.lower()
    words = text_lower.split()

    if not words:
        return "NEUTRAL", 0

    aspect_pos, aspect_neg = 0, 0

    # Words that indicate diminishment/moderation
    diminishers = {'bit', 'somewhat', 'kind', 'fairly', 'quite', 'rather', 'sort', 'kinda', 'a'}
    negators = {'not', 'no', 'never', "n't", 'neither', 'nor'}
    context_words_set = diminishers | negators

    # Find aspect keyword positions
    aspect_positions = []
    for i, word in enumerate(words):
        for keyword in keywords:
            if keyword in word:
                aspect_positions.append(i)
                break

    if not aspect_positions:
        return "NEUTRAL", 0

    # For each aspect mention, find sentiment words nearby
    for aspect_idx in aspect_positions:
        # Look at words immediately before the aspect keyword (more relevant)
        # and words after it
        closest_sentiment = None
        closest_distance = float('inf')

        # Check 5 words before and 3 words after (before is more important)
        for i in range(max(0, aspect_idx - 5), min(len(words), aspect_idx + 3)):
            word = words[i]

            # Skip the aspect keyword itself and other context words
            if word in context_words_set:
                continue

            points = get_word_points_with_pos(word, None)

            if points == 0:
                continue

            # Distance metric: prefer words closer to the aspect
            distance = abs(i - aspect_idx)

            # If we found a sentiment word, use it (prefer closer words)
            if distance < closest_distance:
                closest_distance = distance
                closest_sentiment = (i, word, points)

        if closest_sentiment is None:
            continue

        sent_idx, sent_word, sent_points = closest_sentiment

        # Check for negation (word right before the sentiment word)
        is_negated = False
        diminished = False

        if sent_idx > 0:
            prev_word = words[sent_idx - 1]
            if prev_word in negators:
                is_negated = True
            elif prev_word in diminishers:
                diminished = True
                sent_points = round(sent_points * 0.5)  # Reduce by 50%

        # Apply negation (flip the sentiment)
        if is_negated:
            if sent_points > 0:
                aspect_neg += abs(sent_points)
            else:
                aspect_pos += abs(sent_points)
        else:
            if sent_points > 0:
                aspect_pos += sent_points
            else:
                aspect_neg += abs(sent_points)

        if debug:
            print(
                f"  [Aspect: {aspect_name}] Found '{sent_word}' (pts={sent_points}, negated={is_negated}, diminished={diminished})")

    # Determine final sentiment for this aspect
    if aspect_pos > aspect_neg:
        return "POSITIVE", aspect_pos
    elif aspect_neg > aspect_pos:
        return "NEGATIVE", -aspect_neg
    else:
        return "NEUTRAL", 0

def batch_sentiment_analysis(texts: list, domain="general"):
    """Process multiple texts efficiently."""
    results = []

    if nlp is None:
        # Fallback if spaCy not available
        for text in texts:
            result = sentiment_analysis(text, debug=False, domain=domain)
            result['text'] = text  # ✅ ADD THIS LINE
            results.append(result)
    else:
        # Use spaCy batch processing
        docs = nlp.pipe(texts, batch_size=100)
        for text, doc in zip(texts, docs):
            result = sentiment_analysis(text, debug=False, domain=domain, nlp_doc=doc)
            result['text'] = text  # ✅ ADD THIS LINE
            results.append(result)

    return results

COLLECT_TIMING_STATS = True


def detect_mixed_sentiment_advanced(raw_text: str, pos_score: int, neg_score: int,
                                    has_contrast: bool, debug: bool = False) -> Tuple[bool, str]:
    """
    Detect MIXED sentiment with multiple signals.

    Returns:
        (is_mixed: bool, reason: str)
    """
    text_lower = raw_text.lower()

    # Signal 1: Explicit contrast phrases (MUST have actual contrast detected)
    contrast_phrases = {
        'but', 'however', 'although', 'though', 'yet', 'still',
        'despite', 'nonetheless', 'on the other hand', 'at the same time'
    }
    has_explicit_contrast = any(phrase in text_lower for phrase in contrast_phrases)

    if has_explicit_contrast and has_contrast:
        # BUT: only mark as MIXED if actual contradiction (one pos, one neg)
        # "Bad quality and slow shipping" shouldn't be MIXED (both negative)
        if pos_score > 0 and neg_score > 0:  # Both sentiments exist
            if debug:
                print(f"[MIXED Detection] Signal 1: Explicit contrast phrase found")
            return True, "explicit_contrast"

    # Signal 2: Scores are VERY close (within 30% gap) AND both sides positive
    # This filters out "bad quality AND slow shipping" (both negative)
    total = abs(pos_score) + abs(neg_score)
    if total > 0:
        gap = abs(pos_score - neg_score) / total
        if gap < 0.20:  # Tighter: less than 30% difference
            # ONLY mark as MIXED if BOTH pos and neg are positive (actual conflict)
            if pos_score > 0 and neg_score > 0:
                if debug:
                    print(
                        f"[MIXED Detection] Signal 2: Very close conflicting scores ({pos_score} vs {neg_score}, gap={gap:.0%})")
                return True, "close_conflicting_scores"

    return False, "none"

def sentiment_analysis(raw_text: str, debug: bool = False, domain: str = "general", nlp_doc=None) -> Dict:
    """
    Complete sentiment analysis with ALL features.

    Features:
    ✅ Explicit neutral detection
    ✅ Emoji sentiment
    ✅ Phrase analysis (contrast & comparison)
    ✅ Word-by-word scoring
    ✅ Intensifiers & diminishers
    ✅ Negation handling
    ✅ Mixed sentiment detection
    ✅ Aspect analysis (optional, disabled by default for speed)
    ✅ Performance timing
    """

    review_start = time.perf_counter()

    # === INPUT VALIDATION ===
    if not raw_text or not isinstance(raw_text, str):
        return {
            "sentiment": "NEUTRAL",
            "confidence": 0.0,
            "pos_score": 0,
            "neg_score": 0,
            "pos_count": 0,
            "neg_count": 0,
            "aspects": {}
        }

    # === 1. EXPLICIT NEUTRAL CHECK (EARLY EXIT) ===
    explicit_start = time.perf_counter()

    is_neutral = is_explicitly_neutral(raw_text)

    explicit_elapsed = (time.perf_counter() - explicit_start) * 1000
    _timing_stats['is_explicitly_neutral'].append(explicit_elapsed)

    if is_neutral:
        debug_print(debug, "[Explicit Neutral] Early return")
        return {
            "username": "batch_analysis",
            "sentiment": "NEUTRAL",
            "confidence": 1.0,
            "pos_score": 0,
            "neg_score": 0,
            "pos_count": 0,
            "neg_count": 0,
            "aspects": {},
            "text": raw_text
        }

    # === 2. NLP PROCESSING ===
    nlp_start = time.perf_counter()

    if nlp_doc is None and nlp is not None:
        nlp_doc = nlp(raw_text)

    nlp_elapsed = (time.perf_counter() - nlp_start) * 1000
    _timing_stats['nlp_processing'].append(nlp_elapsed)

    # === 3. INITIALIZE COUNTERS ===
    pos_word_count = 0
    neg_word_count = 0
    total_pos = 0
    total_neg = 0
    original_pos = 0  # Before phrase adjustments
    original_neg = 0
    has_contrast_mixed = False

    # === 4. EMOJI SENTIMENT ===
    emoji_start = time.perf_counter()

    for char in raw_text:
        score = get_emoji_sentiment(char)
        if score > 0:
            total_pos += score
            original_pos += score
        elif score < 0:
            total_neg += abs(score)
            original_neg += abs(score)

    emoji_elapsed = (time.perf_counter() - emoji_start) * 1000

    # === 5. TEXT PREPARATION ===
    text_expanded = expand_emojis_to_text(raw_text)

    # === 6. PHRASE ADJUSTMENTS (Slang) ===
    phrase_start = time.perf_counter()

    total_pos, total_neg = apply_slang_adjustments(raw_text, total_pos, total_neg, debug=debug)

    phrase_elapsed = (time.perf_counter() - phrase_start) * 1000

    # === 7. SENTENCE PROCESSING ===
    import re

    if nlp_doc:
        sentence_objs = list(nlp_doc.sents)
    else:
        temp_sentences = re.split(r'[.!?]+', text_expanded)
        sentence_objs = [s.strip() for s in temp_sentences if s.strip()]

    # === 8. MAIN WORD-BY-WORD LOOP ===
    word_start = time.perf_counter()

    all_sentiment_words = pos_mild | pos_medium | pos_strong | neg_mild | neg_medium | neg_strong

    for sent_item in sentence_objs:
        # Parse sentence
        if hasattr(sent_item, "text"):
            sent_text = sent_item.text
            sent_doc = sent_item
        else:
            sent_text = sent_item
            sent_doc = nlp(sent_text) if nlp else None

        if not sent_text.strip():
            continue

        sent_pos, sent_neg = 0, 0

        # A. Phrase Analysis (Contrast)
        p_pos, p_neg, c_mixed = PhraseAnalyzer.analyze_contrast_phrase(sent_text, debug=debug)
        if c_mixed:
            has_contrast_mixed = True

        total_pos += p_pos
        total_neg += p_neg
        original_pos += p_pos
        original_neg += p_neg

        # B. Phrase Analysis (Comparison)
        comp_pos, comp_neg = PhraseAnalyzer.analyze_comparison_phrase(sent_text, debug=debug)
        total_pos += comp_pos
        total_neg += comp_neg
        original_pos += comp_pos
        original_neg += comp_neg

        # Skip word-by-word if comparison found
        if comp_pos > 0 or comp_neg > 0:
            continue

        # C. Word-by-Word Analysis
        words = clean_input(sent_text)

        for i, word in enumerate(words):
            # Skip if intensifier looking ahead
            if (word in CONTEXT_INTENSIFIERS or word in INTENSIFIERS) and i + 1 < len(words):
                next_w = words[i + 1]
                if get_word_points_with_pos(next_w, sent_doc) != 0:
                    continue

            # Get base points
            points = get_word_points_with_pos(word, sent_doc)

            # Apply intensifier if found
            intensifier, multiplier = IntensifierHandler.find_intensifier_and_target(
                words, i, sent_doc=sent_doc
            )

            if points != 0 and intensifier:
                points = IntensifierHandler.apply_context_intensifier(
                    word, points, intensifier, multiplier, debug=debug
                )

            # Fuzzy matching fallback
            if points == 0:
                matched = find_closest_word_fast(word, all_sentiment_words, threshold=0.85)
                if matched:
                    points = get_word_points_with_pos(matched, sent_doc)

            if points == 0:
                continue

            # Handle negation
            is_negated = PhraseAnalyzer.is_negated(i, words)

            # Apply diminishers
            dim_mult = 1.0
            if i > 0 and is_diminisher_in_context(words[i - 1], word):
                dim_mult = 0.75

            final_points = round(points * dim_mult)

            # Apply to scores
            if is_negated:
                if points > 0:
                    sent_neg += max(1, round(final_points * 0.5))
                    neg_word_count += 1
                else:
                    sent_pos += max(1, round(abs(final_points) * 0.5))
                    pos_word_count += 1
            else:
                if final_points > 0:
                    sent_pos += final_points
                    pos_word_count += 1
                else:
                    sent_neg += abs(final_points)
                    neg_word_count += 1

        total_pos += sent_pos
        total_neg += sent_neg
        original_pos += sent_pos
        original_neg += sent_neg

    word_elapsed = (time.perf_counter() - word_start) * 1000
    _timing_stats['word_analysis'].append(word_elapsed)

    # DEBUG: Show scores at this point
    if "bad" in raw_text.lower() and "quality" in raw_text.lower():
        print(f"\n[DEBUG] After word-by-word analysis for: '{raw_text[:50]}'")
        print(f"  original_pos = {original_pos}")
        print(f"  original_neg = {original_neg}")
        print(f"  total_pos = {total_pos}")
        print(f"  total_neg = {total_neg}")

    # === 9. SARCASM DETECTION ===
    has_sarcasm, sarcasm_conf = detect_sarcasm_emojis(raw_text, debug=debug)
    if has_sarcasm:
        if total_pos > total_neg:
            total_neg += 5
        elif total_neg > total_pos:
            total_neg += 3
        else:
            total_neg += 2

    slang_info = detect_internet_slang(raw_text)

    # === 9b. CONTEXT TERMS: IDIOMS + DOMAIN-SPECIFIC WEIGHTS ===
    dom_pos, dom_neg, context_terms = domain_and_idiom_adjustment(raw_text, domain)
    if dom_pos or dom_neg:
        original_pos = max(0, original_pos + dom_pos)
        original_neg = max(0, original_neg + dom_neg)
        total_pos = max(0, total_pos + dom_pos)
        total_neg = max(0, total_neg + dom_neg)
        if debug:
            print(f"[Context terms] {context_terms}: pos {dom_pos:+d}, neg {dom_neg:+d}")

    # === 10. MIXED SENTIMENT DETECTION (IMPROVED) ===
    mixed_start = time.perf_counter()

    # Use phrase-based detection first
    early_is_mixed, mixed_reason = detect_mixed_sentiment_advanced(
        raw_text,
        original_pos,
        original_neg,
        has_contrast_mixed,
        debug=debug
    )

    if early_is_mixed and debug:
        print(f"\n[MIXED] '{raw_text[:60]}...'")
        print(f"  Scores: pos={original_pos}, neg={original_neg}")
        print(f"  Reason: {mixed_reason}")
        print(f"  has_contrast_mixed: {has_contrast_mixed}")

    mixed_elapsed = (time.perf_counter() - mixed_start) * 1000
    _timing_stats['mixed_detection'].append(mixed_elapsed)

    # === 11. DETERMINE FINAL SENTIMENT WITH PROPER CONFIDENCE ===

    # Helper function for confidence calculation
    def calculate_confidence(pos_score, neg_score, is_mixed):
        if is_mixed:
            # For MIXED, express lower confidence
            total_magnitude = abs(pos_score) + abs(neg_score)
            if total_magnitude == 0:
                return 0.5

            # Confidence = 45% base + 15% if one side slightly dominates
            min_ratio = min(abs(pos_score), abs(neg_score)) / total_magnitude
            confidence = 0.45 + (min_ratio * 0.15)  # Range: 45-60%
            return min(0.60, max(0.45, confidence))

        # For POSITIVE/NEGATIVE, confidence based on margin and strength
        total_magnitude = abs(pos_score) + abs(neg_score)
        if total_magnitude == 0:
            return 0.0

        # Margin ratio: how much one side dominates
        margin = abs(pos_score - neg_score)
        margin_ratio = margin / (total_magnitude + 1)

        # Strength: how many sentiment words total (more = more confident)
        # If you have "Terrible! Horrible! Awful!" (3 strong words), confidence should be high
        strength_factor = min(total_magnitude / 5, 1.0)  # ✅ CHANGED: 8 → 5 (faster ramp)

        # Combine margin and strength with better weighting
        # Higher margin weight (0.8) because margin is more reliable indicator
        confidence = (margin_ratio * 0.75) + (strength_factor * 0.25)  # ✅ CHANGED: 0.7/0.25 → 0.8/0.2

        # Clamp to reasonable range: 35-98% (was 30-95%)
        return min(0.98, max(0.35, confidence))

    # Determine final sentiment
    # Use original_pos/original_neg for final sentiment (not total_pos/total_neg which include adjustments)
    if early_is_mixed:
        final_sentiment = "MIXED"
        confidence = calculate_confidence(original_pos, original_neg, is_mixed=True)

    elif original_pos == 0 and original_neg == 0:
        final_sentiment = "NEUTRAL"
        confidence = 0.0

    elif original_pos > original_neg:
        final_sentiment = "POSITIVE"
        confidence = calculate_confidence(original_pos, original_neg, is_mixed=False)

    elif original_neg > original_pos:
        final_sentiment = "NEGATIVE"
        confidence = calculate_confidence(original_pos, original_neg, is_mixed=False)

    else:
        final_sentiment = "NEUTRAL"
        confidence = 0.5

    # === 12. ASPECT ANALYSIS (RE-ENABLED WITH FAST VERSION) ===
    aspect_start = time.perf_counter()

    # Use fast aspect analysis - only checks for aspect keywords, doesn't do proximity scoring
    aspect_results = analyze_aspects_fast(raw_text, debug=debug)

    aspect_elapsed = (time.perf_counter() - aspect_start) * 1000
    _timing_stats['aspect_analysis'].append(aspect_elapsed)

    # === 13. BUILD RESULT ===
    result = {
        "username": 'batch_analysis',
        "sentiment": final_sentiment,
        "confidence": confidence,
        "pos_score": total_pos,
        "neg_score": total_neg,
        "pos_count": pos_word_count,
        "neg_count": neg_word_count,
        "is_sarcastic": has_sarcasm,
        "sarcasm_confidence": sarcasm_conf,
        "detected_slang": slang_info.get("detected_slang", []),
        "slang_adjustment": slang_info.get("total_adjustment", 0),
        "aspects": aspect_results,
        "domain": domain,
        "context_terms": context_terms,
        "text": raw_text
    }

    # Domain-specific weights are applied in step 9b, before the label is decided.

    # === 15. RECORD TIMING ===
    total_elapsed = (time.perf_counter() - review_start) * 1000
    _timing_stats['total'].append(total_elapsed)

    # Log if slow
    if total_elapsed > 500:
        print(f"\n⚠️  SLOW REVIEW ({total_elapsed:.0f}ms):")
        print(f"  Explicit Neutral: {explicit_elapsed:.1f}ms")
        print(f"  NLP Processing:   {nlp_elapsed:.1f}ms")
        print(f"  Word Analysis:    {word_elapsed:.1f}ms")
        print(f"  Aspect Analysis:  {aspect_elapsed:.1f}ms")
        print(f"  Mixed Detection:  {mixed_elapsed:.1f}ms")
        print(f"  Text: {raw_text[:80]}...")

    if COLLECT_TIMING_STATS:
        total_elapsed = (time.perf_counter() - review_start) * 1000
        _timing_stats['total'].append(total_elapsed)

    return result

if __name__ == "__main__":
    # Test cases
    test_reviews = [
        "Good but not great",
        "Pretty good",
        "Pretty nice day",
        "Not great but okay",
        "Excellent but expensive",
        "Terrible food, but amazing service",
        "Nice try 🤡",
        "Amazing work 🤡",
        "This is terrible 🤡",
    ]

    print("\n" + "="*60)
    print("SENTIMENT ANALYZER TEST")
    print("="*60)

    for review in test_reviews:
        result = sentiment_analysis(review, debug=False)
        print(f"\nReview: '{review}'")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: +{result['pos_score']}/-{result['neg_score']}")
        print(f"  Confidence: {int(result['confidence']*100)}%")


