"""
PDF Report Generation for Sentiment Analysis - CORRECTED
Creates professional reports with charts and summaries
"""

from datetime import datetime
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class SentimentReportGenerator:
    """Generates professional PDF reports from sentiment analysis data."""

    def __init__(self, filename: str = "sentiment_report.pdf"):
        """
        Initialize report generator.

        Args:
            filename: Output PDF filename
        """
        self.filename = filename
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='Title2',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))

        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            spaceAfter=12,
        ))

    def add_title(self, title: str, subtitle: str = ""):
        """Add title and optional subtitle"""
        self.story.append(Paragraph(
            f"<b>{title}</b>",
            self.styles['Title2']
        ))
        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['Subtitle']))
        self.story.append(Spacer(1, 0.3 * inch))

    def add_overall_sentiment_summary(self, summary: Dict):
        """Add overall sentiment summary section."""
        self.add_title("Sentiment Summary")

        # Summary table
        data = [
            ["Metric", "Value"],
            ["Total Reviews Analyzed", str(summary.get("total_reviews", 0))],
            ["Primary Sentiment", summary.get("primary_sentiment", "N/A").upper()],
            ["Positive Reviews", str(summary.get("positive_count", 0))],
            ["Negative Reviews", str(summary.get("negative_count", 0))],
            ["Neutral Reviews", str(summary.get("neutral_count", 0))],
            ["Average Confidence", f"{summary.get('average_confidence', 0):.1%}"],
        ]

        table = Table(data, colWidths=[2.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), 'CENTER'),
            ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))

    def add_aspect_analysis(self, aspect_data: Dict):
        """Add aspect-based sentiment analysis."""
        self.add_title("Aspect Analysis")

        # Prepare data
        data = [["Aspect", "Status", "Positive", "Negative", "% Positive"]]

        for aspect, info in sorted(aspect_data.items()):
            if info["count"] == 0:
                continue

            status = info["status"]
            pos = info["positive"]
            neg = info["negative"]
            pct = info["percentage_positive"]

            status_text = status.upper()

            data.append([
                aspect.capitalize(),
                status_text,
                str(pos),
                str(neg),
                f"{pct:.0f}%"
            ])

        table = Table(data, colWidths=[1.5 * inch, 1.2 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), 'CENTER'),
            ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))

    def add_recommendations(self, recommendations: Dict):
        """Add recommendations section."""
        self.add_title("Key Recommendations")

        sections = [
            ("🔴 CRITICAL ISSUES", recommendations.get("critical", []), colors.HexColor('#ff6b6b')),
            ("🟡 AREAS TO IMPROVE", recommendations.get("improve", []), colors.HexColor('#ffd93d')),
            ("🟢 AREAS TO MAINTAIN", recommendations.get("maintain", []), colors.HexColor('#6bcf7f')),
            ("⭐ STRENGTHS", recommendations.get("strengths", []), colors.HexColor('#4ecdc4')),
        ]

        for title, items, color in sections:
            if items:
                self.story.append(Paragraph(f"<b>{title}</b>", self.styles['Heading3']))

                for item in items:
                    # Remove emoji for cleaner text
                    clean_item = item.split(" - ", 1)[-1] if " - " in item else item
                    self.story.append(Paragraph(
                        f"• {clean_item}",
                        self.styles['Normal']
                    ))

                self.story.append(Spacer(1, 0.15 * inch))

    def add_trend_analysis(self, trend_data: Dict):
        """Add temporal trend analysis."""
        self.add_title("Sentiment Trends")

        direction = trend_data.get("direction", "UNKNOWN")
        change = trend_data.get("change_percentage", 0)
        period = trend_data.get("period_days", 30)

        trend_text = f"""
        <b>Overall Trend (Last {period} Days):</b><br/>
        Direction: {direction}<br/>
        Change: {change:+.1f}%<br/>
        Starting Score: {trend_data.get('starting_avg_score', 0):+.1f}<br/>
        Ending Score: {trend_data.get('ending_avg_score', 0):+.1f}<br/>
        """

        self.story.append(Paragraph(trend_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3 * inch))

    def add_confidence_analysis(self, confidence_scorer):
        """Add confidence analysis section to PDF - CORRECTED"""
        self.add_title("Review Reliability Analysis")

        batch_stats = confidence_scorer.calculate_batch_confidence()

        data = [
            ["Metric", "Value"],
            ["Total Reviews", str(batch_stats['total_reviews'])],
            ["Average Confidence", f"{batch_stats['average_confidence']:.0%}"],
            ["High Confidence (>70%)", f"{batch_stats['high_confidence_count']}"],
            ["Low Confidence (<30%)", f"{batch_stats['low_confidence_count']}"],
        ]

        table = Table(data, colWidths=[2.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), 'CENTER'),
            ("FONTNAME", (0, 0), (-1, 0), 'Helvetica-Bold'),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3 * inch))

    def add_metadata(self):
        """Add footer with generation metadata"""
        self.story.append(Spacer(1, 0.2 * inch))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = f"<i>Report generated: {timestamp}</i>"

        meta_style = ParagraphStyle(
            'Meta',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER,
        )

        self.story.append(Paragraph(metadata, meta_style))

    def generate(self):
        """Generate the PDF file"""
        self.add_metadata()
        self.doc.build(self.story)
        print(f"✓ PDF Report generated: {self.filename}")

    def create_full_report(self,
                           title: str,
                           summary: Dict,
                           aspect_data: Dict,
                           recommendations: Dict,
                           trend_data: Optional[Dict] = None,
                           confidence_scorer = None):
        """
        Create a complete report with all sections - CORRECTED.

        Args:
            title: Report title
            summary: Sentiment summary data
            aspect_data: Aspect analysis data
            recommendations: Recommendations data
            trend_data: Optional trend analysis data
            confidence_scorer: Optional ConfidenceScorer instance
        """
        self.add_title(title, f"Generated on {datetime.now().strftime('%Y-%m-%d')}")

        self.add_overall_sentiment_summary(summary)
        self.add_aspect_analysis(aspect_data)
        self.add_recommendations(recommendations)

        if trend_data:
            self.add_trend_analysis(trend_data)

        if confidence_scorer:
            self.add_confidence_analysis(confidence_scorer)

        self.generate()


# Example usage
if __name__ == "__main__":
    # Sample data
    summary = {
        "total_reviews": 25,
        "primary_sentiment": "positive",
        "positive_count": 18,
        "negative_count": 5,
        "neutral_count": 2,
        "average_confidence": 0.82,
    }

    aspect_data = {
        "food": {
            "status": "POSITIVE",
            "positive": 20,
            "negative": 3,
            "neutral": 2,
            "count": 25,
            "percentage_positive": 80,
        },
        "service": {
            "status": "NEGATIVE",
            "positive": 10,
            "negative": 10,
            "neutral": 5,
            "count": 25,
            "percentage_positive": 40,
        },
        "atmosphere": {
            "status": "POSITIVE",
            "positive": 15,
            "negative": 5,
            "neutral": 5,
            "count": 25,
            "percentage_positive": 60,
        },
    }

    recommendations = {
        "critical": ["Service quality needs immediate attention"],
        "improve": ["Train staff on customer service"],
        "maintain": ["Keep food quality consistent"],
        "strengths": ["Excellent food quality", "Good atmosphere"],
    }

    trend_data = {
        "direction": "IMPROVING ↗",
        "change_percentage": 15.5,
        "starting_avg_score": 0.5,
        "ending_avg_score": 2.1,
        "period_days": 30,
    }

    # Generate report
    generator = SentimentReportGenerator("sample_report.pdf")
    generator.create_full_report(
        title="Restaurant Sentiment Analysis Report",
        summary=summary,
        aspect_data=aspect_data,
        recommendations=recommendations,
        trend_data=trend_data,
    )