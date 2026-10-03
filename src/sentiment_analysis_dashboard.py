"""
Streamlit Dashboard for Sentiment Analysis System
Uses ONLY API calls - no direct database access
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
import json

API_BASE_URL = "http://localhost:5000"

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
# Warm, cheerful look: cream surfaces, rounded white cards, one color per sentiment.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600&family=Nunito:wght@400;600;700&display=swap');

    html, body, .stApp, [data-testid="stMarkdownContainer"], input, textarea, button, label {
        font-family: 'Nunito', 'Segoe UI', sans-serif;
    }
    h1, h2, h3, [data-testid="stHeading"] * {
        font-family: 'Fredoka', 'Trebuchet MS', sans-serif !important;
        font-weight: 600 !important;
        color: #33271e;
        letter-spacing: 0;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 4%, rgba(255, 194, 60, 0.22), transparent 32%),
            radial-gradient(circle at 4% 96%, rgba(20, 184, 166, 0.14), transparent 36%),
            #fff6ec;
    }
    [data-testid="stSidebar"] {
        background: #ffeedd;
        border-right: 1px solid #f4e3d2;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        background: #ffffff;
        border: 1px solid #ecd3bb;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(94, 58, 32, 0.06);
    }
    .stButton > button {
        border-radius: 999px;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(217, 68, 43, 0.18);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(217, 68, 43, 0.24);
    }
    [data-testid="stAlert"] {
        border-radius: 16px;
    }
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #f4e3d2;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 26px rgba(94, 58, 32, 0.08);
    }
    .positive { color: #0b7d70; font-weight: 700; }
    .negative { color: #c0341c; font-weight: 700; }
    .neutral  { color: #2f6fb3; font-weight: 700; }
    .mixed    { color: #b26a00; font-weight: 700; }
    .confidence-high { color: #0b7d70; }
    .confidence-low  { color: #b26a00; }
</style>
""", unsafe_allow_html=True)

def show_chart(fig, **kwargs):
    """Render a Plotly figure with the dashboard's warm theme."""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Nunito, Segoe UI, sans-serif", color="#33271e"),
        title_font=dict(family="Fredoka, Trebuchet MS, sans-serif", size=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        margin=dict(l=24, r=24, t=84, b=24),
    )
    fig.update_xaxes(gridcolor="#f4e3d2", zerolinecolor="#ecd3bb", automargin=True)
    fig.update_yaxes(gridcolor="#f4e3d2", zerolinecolor="#ecd3bb", automargin=True)
    st.plotly_chart(fig, theme=None, **kwargs)

# ========== API HELPER FUNCTIONS ==========
def api_call_cached(endpoint: str, params: dict = None):
    """Cached GET calls only (for listing/viewing data)"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Is it running on port 5000?")
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def api_call_live(endpoint: str, method: str = "GET", data: dict = None, params: dict = None):
    """Live calls - no caching (POST, DELETE, critical data)"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        timeout = 300 if endpoint == "/api/v1/batch-analyze" else 10

        if method == "GET":
            resp = requests.get(url, params=params, timeout=timeout)
        elif method == "POST":
            resp = requests.post(url, json=data, timeout=timeout)
        elif method == "DELETE":
            resp = requests.delete(url, timeout=timeout)
        else:
            return None

        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Is it running on port 5000?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"API request timed out (waited {timeout}s)")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Analyze Review", "Batch Analysis", "Dashboard", "Statistics", "Settings"]
)

# ========== PAGE: HOME ==========
if page == "Home":
    st.title("Sentiment Analysis Dashboard")
    st.markdown("""
    This dashboard lets you:
    
    - **Analyze Reviews** - Get detailed sentiment analysis with aspect breakdown
    - **View Dashboard** - See visual trends and patterns in your data
    - **Check Statistics** - Comprehensive metrics and confidence analysis
    - **Configure Settings** - Adjust analysis parameters
    
    ---
    """)

    # Quick stats
    result = api_call_cached("/api/v1/reviews", params={"limit": 10000})

    if result and result.get('success'):
        reviews = result.get('reviews', [])
        total_reviews = len(reviews)

        if total_reviews > 0:
            positive = len([r for r in reviews if r['sentiment'] == 'POSITIVE'])
            negative = len([r for r in reviews if r['sentiment'] == 'NEGATIVE'])
            neutral = len([r for r in reviews if r['sentiment'] == 'NEUTRAL'])
            mixed = len([r for r in reviews if r['sentiment'] == 'MIXED'])

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Reviews", total_reviews)
            with col2:
                st.metric("Positive", positive, f"{(positive/total_reviews*100):.0f}%")
            with col3:
                st.metric("Negative", negative, f"{(negative/total_reviews*100):.0f}%")
            with col4:
                st.metric("Neutral", neutral, f"{(neutral/total_reviews*100):.0f}%")

            col5, col6 = st.columns(2)
            with col5:
                st.metric("Mixed", mixed, f"{(mixed/total_reviews*100):.0f}%")
        else:
            st.info("No reviews yet. Analyze a review to get started.")
    else:
        st.warning("Unable to fetch reviews from API")

# ========== PAGE: ANALYZE REVIEW ==========
elif page == "Analyze Review":
    st.title("Analyze a Review")

    col1, col2 = st.columns([3, 1])

    with col1:
        username = st.text_input("Username (optional)", value="")
    with col2:
        domain = st.selectbox("Domain", ["general", "restaurant", "software", "hotel", "retail"], help="Applies domain-specific word weights")

    review_text = st.text_area(
        "Enter your review:",
        height=200,
        placeholder="Type or paste your review here..."
    )

    save_review = st.checkbox("Save to database", value=True)

    if st.button("Analyze", type="primary", use_container_width=True):
        if not review_text.strip():
            st.error("Enter a review to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = api_call_live(
                    "/api/v1/analyze",
                    method="POST",
                    data={
                        "text": review_text,
                        "username": username or "Anonymous",
                        "domain": domain,
                        "save_to_db": save_review
                    }
                )

            if result and result.get('success'):
                st.success("Analysis complete")
                analysis = result.get('analysis', {})

                # Main sentiment result
                col1, col2, col3 = st.columns(3)

                with col1:
                    sentiment = analysis.get('sentiment')
                    if sentiment == 'POSITIVE':
                        st.markdown(f"### <span class='positive'>{sentiment}</span>", unsafe_allow_html=True)
                    elif sentiment == 'NEGATIVE':
                        st.markdown(f"### <span class='negative'>{sentiment}</span>", unsafe_allow_html=True)
                    elif sentiment == 'MIXED':
                        st.markdown(f"### <span class='mixed'>{sentiment}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### <span class='neutral'>{sentiment}</span>", unsafe_allow_html=True)

                with col2:
                    confidence = analysis.get('confidence', 0) * 100
                    if confidence > 70:
                        st.markdown(f"### <span class='confidence-high'>Confidence: {confidence:.0f}%</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"### <span class='confidence-low'>Confidence: {confidence:.0f}%</span>", unsafe_allow_html=True)

                with col3:
                    pos = analysis.get('pos_score', 0)
                    neg = analysis.get('neg_score', 0)
                    st.metric("Score", f"+{pos}/-{neg}")

                # Sentiment breakdown
                st.subheader("Sentiment Breakdown")
                col1, col2 = st.columns(2)

                with col1:
                    sentiments = [analysis.get('pos_score', 0), analysis.get('neg_score', 0)]
                    fig_pie = px.pie(
                        values=sentiments,
                        names=['Positive', 'Negative'],
                        color=['Positive', 'Negative'],
                        color_discrete_map={'Positive': '#14b8a6', 'Negative': '#ff6b54'},
                        title="Positive vs Negative"
                    )
                    show_chart(fig_pie, use_container_width=True)

                with col2:
                    st.write("**Sentiment Details:**")
                    st.write(f"- Positive words found: {analysis.get('pos_count', 0)}")
                    st.write(f"- Negative words found: {analysis.get('neg_count', 0)}")
                    st.write(f"- Positive score: +{analysis.get('pos_score', 0)}")
                    st.write(f"- Negative score: -{analysis.get('neg_score', 0)}")

                # Aspects
                aspects = analysis.get('aspects', {})
                if aspects:
                    st.subheader("Aspect Analysis")
                    aspects_df = pd.DataFrame([
                        {
                            "Aspect": aspect.upper(),
                            "Sentiment": aspect_val.get('sentiment', 'NEUTRAL'),
                            "Score": aspect_val.get('score', 0)
                        }
                        for aspect, aspect_val in aspects.items()
                    ])

                    fig_aspects = px.bar(
                        aspects_df,
                        x="Aspect",
                        y="Score",
                        color="Sentiment",
                        color_discrete_map={'POSITIVE': '#14b8a6', 'NEGATIVE': '#ff6b54', 'NEUTRAL': '#4c9df0'},
                        title="Aspect Sentiment Scores"
                    )
                    show_chart(fig_aspects, use_container_width=True)

                # Sarcasm detection
                if analysis.get('is_sarcastic'):
                    st.warning(f"Possible sarcasm detected. Confidence: {analysis.get('sarcasm_confidence', 0):.0f}%")

# ========== PAGE: BATCH ANALYSIS ==========
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.markdown("Analyze multiple reviews at once and compare the results.")

    col1, col2 = st.columns([3, 1])

    with col1:
        domain = st.selectbox("Domain", ["general", "restaurant", "software", "hotel", "retail"], help="Applies domain-specific word weights")
    with col2:
        save_to_db = st.checkbox("Save all to database", value=True)

    st.subheader("Enter Reviews")

    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload CSV"],
        horizontal=True
    )

    reviews_list = []

    if input_method == "Paste Text":
        st.markdown("**Format:** `username: review` or just paste reviews (one per line, separated by blank lines)")

        batch_text = st.text_area(
            "Reviews (format: 'username: review' or just reviews, separated by blank lines):",
            height=300,
            placeholder="John: This product is amazing!\n\nSarah: Terrible quality"
        )

        if batch_text.strip():
            raw_reviews = [r.strip() for r in batch_text.split('\n\n') if r.strip()]
            for idx, review_text in enumerate(raw_reviews):
                first_line = review_text.split('\n')[0] if '\n' in review_text else review_text

                if ':' in first_line:
                    potential_username = first_line.split(':', 1)[0].strip()
                    if len(potential_username) < 20 and not any(p in potential_username for p in ['.', '!', '?']):
                        parts = review_text.split(':', 1)
                        username = parts[0].strip()
                        review = parts[1].strip()
                    else:
                        username = f"User {idx + 1}"
                        review = review_text
                else:
                    username = f"User {idx + 1}"
                    review = review_text

                if review:
                    reviews_list.append({'review': review, 'username': username})

    else:  # CSV upload
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)

                review_column = None
                for col in ['review', 'Review', 'text', 'Text', 'content', 'Content', 'comment', 'Comment']:
                    if col in df_uploaded.columns:
                        review_column = col
                        break

                if review_column is None:
                    st.error(f"Could not find review column. Available columns: {list(df_uploaded.columns)}")
                else:
                    username_column = None
                    for col in ['username', 'Username', 'user', 'User', 'name', 'Name', 'author', 'Author']:
                        if col in df_uploaded.columns:
                            username_column = col
                            break

                    for idx, row in df_uploaded.iterrows():
                        reviews_list.append({
                            'review': row[review_column],
                            'username': row[username_column] if username_column else f"User {idx + 1}"
                        })

                    st.success(f"Loaded {len(reviews_list)} reviews from CSV")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

    if reviews_list and st.button("Analyze batch", type="primary", use_container_width=True):
        st.success(f"Analyzing {len(reviews_list)} reviews...")

        with st.spinner("Processing..."):
            batch_result = api_call_live(
                "/api/v1/batch-analyze",
                method="POST",
                data={
                    "reviews": [
                        {
                            "text": r['review'],
                            "username": r['username']
                        }
                        for r in reviews_list
                    ],
                    "domain": domain,
                    "save_to_db": save_to_db
                }
            )

        if batch_result and batch_result.get('success'):
            # Store results AND performance metrics
            st.session_state.batch_results = batch_result.get('results', [])
            st.session_state.performance = batch_result.get('performance', {})  # <-- TIMING DATA
            st.success(f"Analyzed {len(st.session_state.batch_results)} reviews")
        else:
            st.error("Batch analysis failed")

    if 'batch_results' in st.session_state and st.session_state.batch_results:
        results = st.session_state.batch_results

        st.subheader("Batch Analysis Summary")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        total = len(results)
        positive = len([r for r in results if r['sentiment'] == 'POSITIVE'])
        negative = len([r for r in results if r['sentiment'] == 'NEGATIVE'])
        neutral = len([r for r in results if r['sentiment'] == 'NEUTRAL'])
        mixed = len([r for r in results if r['sentiment'] == 'MIXED'])
        avg_conf = sum(r['confidence'] for r in results) / len(results) * 100 if results else 0

        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Positive", positive, f"{(positive/total*100):.0f}%")
        with col3:
            st.metric("Negative", negative, f"{(negative/total*100):.0f}%")
        with col4:
            st.metric("Neutral", neutral, f"{(neutral/total*100):.0f}%")
        with col5:
            st.metric("Mixed", mixed, f"{(mixed/total*100):.0f}%")
        with col6:
            st.metric("Avg Confidence", f"{avg_conf:.0f}%")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            sentiments = [positive, negative, neutral, mixed]
            fig_pie = px.pie(
                values=sentiments,
                names=['Positive', 'Negative', 'Neutral', 'Mixed'],
                color=['Positive', 'Negative', 'Neutral', 'Mixed'],
                color_discrete_map={'Positive': '#14b8a6', 'Negative': '#ff6b54', 'Neutral': '#4c9df0', 'Mixed': '#f5b324'},
                title="Sentiment Distribution"
            )
            show_chart(fig_pie, use_container_width=True)

        with col2:
            scores_data = {
                'Positive Scores': [r['pos_score'] for r in results],
                'Negative Scores': [r['neg_score'] for r in results],
            }

            fig_scores = go.Figure()
            fig_scores.add_trace(go.Box(y=scores_data['Positive Scores'], name='Positive', marker_color='#14b8a6'))
            fig_scores.add_trace(go.Box(y=scores_data['Negative Scores'], name='Negative', marker_color='#ff6b54'))
            fig_scores.update_layout(title="Score Distribution", hovermode='closest')
            show_chart(fig_scores, use_container_width=True)

        # Detailed results table
        st.subheader("Detailed Results")

        results_df = pd.DataFrame([
            {
                "Review #": i + 1,
                "Username": r['username'],
                "Review": r.get('text', ''),
                "Sentiment": r['sentiment'],
                "Positive": r['pos_score'],
                "Negative": r['neg_score'],
                "Confidence": f"{r['confidence'] * 100:.1f}%",
                "Positive Words": r.get('pos_count', 0),
                "Negative Words": r.get('neg_count', 0),
                "Sarcasm": "Yes" if r.get('is_sarcastic', False) else "No"
            }
            for i, r in enumerate(results)
        ])

        results_df = results_df.reset_index(drop=True)
        st.dataframe(results_df, use_container_width=True, height=400)

        # Download results
        st.subheader("Download Results")
        col1, col2 = st.columns(2)

        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ========== PAGE: DASHBOARD ==========
elif page == "Dashboard":
    st.title("Sentiment Dashboard")

    result = api_call_cached("/api/v1/reviews", params={"limit": 10000})

    if result and result.get('success'):
        reviews = result.get('reviews', [])

        if len(reviews) > 0:
            df = pd.DataFrame(reviews)

            if 'confidence' in df.columns:
                df['confidence'] = df['confidence'].astype(float) * 100

            if 'created_at' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert('US/Eastern')
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            else:
                df['timestamp'] = datetime.now()

            df_sorted = df.sort_values('timestamp')
            df_sorted['timestamp_display'] = df_sorted['timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')

            fig_timeline = px.line(
                df_sorted,
                x='timestamp_display',
                y='pos_score',
                color='sentiment',
                title="Positive Score Over Time",
                color_discrete_map={'POSITIVE': '#14b8a6', 'NEGATIVE': '#ff6b54', 'NEUTRAL': '#4c9df0', 'MIXED': '#f5b324'}
            )
            show_chart(fig_timeline, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Distribution")
                sentiment_counts = df['sentiment'].value_counts()
                fig_dist = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    color=sentiment_counts.index,
                    color_discrete_map={'POSITIVE': '#14b8a6', 'NEGATIVE': '#ff6b54', 'NEUTRAL': '#4c9df0', 'MIXED': '#f5b324'},
                    title="Distribution of Sentiments"
                )
                show_chart(fig_dist, use_container_width=True)

            with col2:
                st.subheader("Average Scores by Domain")
                domain_scores = df.groupby('domain')[['pos_score', 'neg_score']].mean()
                fig_domain = px.bar(
                    domain_scores,
                    x=domain_scores.index,
                    y=['pos_score', 'neg_score'],
                    title="Average Scores by Domain",
                    barmode='group'
                )
                show_chart(fig_domain, use_container_width=True)

            st.subheader("Confidence Distribution")
            fig_conf = px.histogram(
                df,
                x='confidence',
                nbins=20,
                title="Confidence Level Distribution",
                labels={'confidence': 'Confidence %', 'count': 'Number of Reviews'}
            )
            show_chart(fig_conf, use_container_width=True)

            # Aspect Analysis
            st.subheader("Aspect Sentiment Overview")

            aspect_result = api_call_live("/api/v1/aspects")
            if aspect_result and aspect_result.get('success'):
                aspect_rows = aspect_result.get('aspects', [])
                if aspect_rows:
                    aspects_df = pd.DataFrame(aspect_rows)

                    fig_aspect_bar = px.bar(
                        aspects_df,
                        x="aspect_name",
                        y="count",
                        color="sentiment",
                        barmode="stack",
                        title="Aspect Frequency by Sentiment",
                        color_discrete_map={
                            'POSITIVE': '#14b8a6',
                            'NEGATIVE': '#ff6b54',
                            'NEUTRAL': '#4c9df0',
                            'MIXED': '#f5b324'
                        }
                    )
                    show_chart(fig_aspect_bar, use_container_width=True)

                    fig_aspect_score = px.bar(
                        aspects_df,
                        x="aspect_name",
                        y="avg_score",
                        color="sentiment",
                        barmode="group",
                        title="Average Aspect Score by Sentiment"
                    )
                    show_chart(fig_aspect_score, use_container_width=True)
                else:
                    st.info("No aspect data available yet.")
            else:
                st.info("No aspect data available yet.")
        else:
            st.info("No reviews yet")
    else:
        st.warning("Unable to fetch reviews from API")

# ========== PAGE: STATISTICS ==========
elif page == "Statistics":
    st.title("Detailed Statistics")

    result = api_call_cached("/api/v1/reviews", params={"limit": 10000})

    if result and result.get('success'):
        reviews = result.get('reviews', [])

        if len(reviews) > 0:
            df = pd.DataFrame(reviews)

            if 'confidence' in df.columns:
                df['confidence'] = df['confidence'].astype(float) * 100

            if 'created_at' in df.columns:
                df['timestamp'] = pd.to_datetime(df['created_at'], utc=True).dt.tz_convert('US/Eastern')
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            else:
                df['timestamp'] = datetime.now()

            df_sorted = df.sort_values('timestamp')

            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric("Total Reviews", len(df))
            with col2:
                st.metric("Positive", len(df[df['sentiment'] == 'POSITIVE']))
            with col3:
                st.metric("Negative", len(df[df['sentiment'] == 'NEGATIVE']))
            with col4:
                st.metric("Neutral", len(df[df['sentiment'] == 'NEUTRAL']))
            with col5:
                st.metric("Mixed", len(df[df['sentiment'] == 'MIXED']))
            with col6:
                avg_conf = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.0f}%")

            st.subheader("Confidence Analysis")

            col_a, col_b = st.columns(2)

            with col_a:
                high_conf = (df['confidence'] > 70).sum()
                st.metric("High Confidence (>70%)", high_conf, f"{(high_conf / len(df) * 100):.1f}%")
            with col_b:
                low_conf = (df['confidence'] < 50).sum()
                st.metric("Low Confidence (<50%)", low_conf, f"{(low_conf / len(df) * 100):.1f}%")

            st.subheader("Trends")

            df_sorted['pos_rolling'] = df_sorted['pos_score'].rolling(window=5, min_periods=1).mean()
            df_sorted['neg_rolling'] = df_sorted['neg_score'].rolling(window=5, min_periods=1).mean()
            df_sorted['timestamp_display'] = df_sorted['timestamp'].dt.strftime('%m/%d/%Y %I:%M:%S %p')

            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(
                x=df_sorted['timestamp_display'],
                y=df_sorted['pos_rolling'],
                name='Positive (5-review avg)',
                mode='lines',
                line=dict(color='#14b8a6', width=2)
            ))
            fig_trends.add_trace(go.Scatter(
                x=df_sorted['timestamp_display'],
                y=df_sorted['neg_rolling'],
                name='Negative (5-review avg)',
                mode='lines',
                line=dict(color='#ff6b54', width=2)
            ))
            fig_trends.update_layout(title="Sentiment Trends (5-Review Rolling Average)", hovermode='x unified')
            show_chart(fig_trends, use_container_width=True)

            st.subheader("Recent Reviews")
            display_df = df_sorted[['username', 'text', 'sentiment', 'pos_score', 'neg_score', 'confidence', 'timestamp']].copy()
            display_df = display_df.sort_values('timestamp', ascending=False)
            display_df.columns = ['Username', 'Review', 'Sentiment', 'Positive', 'Negative', 'Confidence', 'Timestamp']

            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1f}%")
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime("%m/%d/%Y %I:%M:%S %p")
            display_df = display_df.reset_index(drop=True)

            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No reviews yet")
    else:
        st.warning("Unable to fetch reviews from API")

# ========== PAGE: SETTINGS ==========
elif page == "Settings":
    st.title("Settings & Configuration")

    st.subheader("Database Management")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("View Database Statistics"):
            result = api_call_live("/api/v1/info")
            if result and result.get('success'):
                info = result.get('database_info', {})
                st.info(f"Total reviews: {info.get('total_reviews', 0)}")
                st.info(f"Unique users: {info.get('unique_users', 0)}")
                st.info(f"First review: {info.get('first_review', 'N/A')}")
                st.info(f"Last review: {info.get('last_review', 'N/A')}")

    with col2:
        st.warning("Database management features should be handled via API only")
        st.info("For data deletion or advanced management, use the API endpoints directly or contact an administrator.")
        st.subheader("Danger Zone")

        if st.button("Delete ALL Reviews", type="primary"):
            with st.spinner("Deleting all reviews..."):
                result = api_call_live("/api/v1/reviews", method="DELETE")

            if result and result.get("success"):
                st.success("All reviews deleted.")
                st.cache_data.clear()
            else:
                st.error("Failed to delete reviews")

    st.subheader("Import/Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Export to CSV"):
            result = api_call_cached("/api/v1/reviews", params={"limit": 10000})
            if result and result.get('success'):
                reviews = result.get('reviews', [])
                if reviews:
                    df = pd.DataFrame(reviews)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"sentiment_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No reviews to export")

    with col2:
        if st.button("Export to JSON"):
            result = api_call_cached("/api/v1/reviews", params={"limit": 10000})
            if result and result.get('success'):
                reviews = result.get('reviews', [])
                if reviews:
                    json_data = json.dumps(reviews, indent=2, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"sentiment_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No reviews to export")

    st.subheader("ℹ️ About")
    st.info("""
    **Sentiment Analysis Dashboard v2.0**
    
    API-First Architecture with:
    - REST API backend (Flask)
    - POS-aware sentiment scoring
    - Multi-aspect analysis
    - Internet slang detection
    - Sarcasm emoji detection
    - Confidence scoring
    - Temporal trend tracking
    - Professional visualizations
    
    **Tech Stack:** Streamlit, Flask, Plotly, spaCy, SQLite
    """)