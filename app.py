import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="GenAI Sentiment Analysis", layout="wide")

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #eef2ff, #f5f0ff, #eff6ff);
        color: #1e1b4b;
    }

    /* All general text */
    .stApp p, .stApp li, .stApp span, .stApp label,
    .stApp div, .stMarkdown, .stText {
        color: #1e1b4b !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 2px solid #e0e7ff;
    }
    [data-testid="stSidebar"] * {
        color: #1e1b4b !important;
    }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] {
        background: #e0e7ff;
        border-radius: 10px;
        padding: 4px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #4338ca;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #6a11cb, #2575fc) !important;
        color: white !important;
    }

    /* Chart containers */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e0e7ff;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 2px 12px rgba(106,17,203,0.08);
    }

    /* Headings */
    h1, h2, h3 {
        color: #1e1b4b !important;
        font-weight: 700;
    }

    /* Selectbox */
    [data-testid="stSelectbox"] > div {
        background: #f5f3ff;
        border-radius: 8px;
        border: 1px solid #c7d2fe;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(106,17,203,0.4);
">
    <h1 style="color: white; margin: 0; font-size: 2rem; font-weight: 700;">
        📊 Public Sentiment Analysis of Generative AI
    </h1>
    <p style="color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 1rem;">
        Analysing public sentiment across platforms using BERT, VADER and TextBlob models.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================
# Load Data
# =============================

@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/M-FATEH/Streamlit/main/combined_df.csv")

df = load_data()

# =============================
# Sidebar Filters
# =============================

st.sidebar.header("Filter Options")

platform_option = st.sidebar.selectbox(
    "Select Platform",
    df["platform"].unique()
)

model_option = st.sidebar.selectbox(
    "Select Sentiment Model",
    ["bert_sentiment", "Vader_Sentiment_Score", "textblob_sentiment"]
)

filtered_df = df[df["platform"] == platform_option]

# =============================
# Tabs
# =============================

tab1, tab2, tab3, tab4 = st.tabs(["📍 Platform Analysis", "🌍 Model Comparison", "🧠 Subjectivity", "🥧 Pie Charts"])

# =====================================================
# TAB 1 – Single Platform Sentiment Distribution
# =====================================================

with tab1:

    st.subheader(f"{model_option.upper()} Sentiment for {platform_option}")

    sentiment_counts = (
        filtered_df[model_option]
        .value_counts(normalize=True)
        .reset_index()
    )

    sentiment_counts.columns = ["Sentiment", "Percentage"]
    sentiment_counts["Percentage"] *= 100

    fig1 = px.bar(
        sentiment_counts,
        x="Sentiment",
        y="Percentage",
        text="Percentage",
        color="Sentiment",
        title="Sentiment Distribution (%)"
    )

    fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig1.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')

    st.plotly_chart(fig1, use_container_width=True)


# =====================================================
# TAB 2 – Model Comparison Across Platforms
# =====================================================

with tab2:

    st.subheader("BERT Sentiment Across Platforms")

    bert_dist = (
        df.groupby(['platform', 'bert_sentiment'])
        .size()
        .reset_index(name="Count")
    )

    fig2 = px.bar(
        bert_dist,
        x="platform",
        y="Count",
        color="bert_sentiment",
        barmode="group",
        title="BERT Sentiment Distribution"
    )

    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Vader Sentiment Across Platforms")

    vader_dist = (
        df.groupby(['platform', 'Vader_Sentiment_Score'])
        .size()
        .reset_index(name="Count")
    )

    fig3 = px.bar(
        vader_dist,
        x="platform",
        y="Count",
        color="Vader_Sentiment_Score",
        barmode="group",
        title="Vader Sentiment Distribution"
    )

    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("TextBlob Sentiment Across Platforms")

    textblob_dist = (
        df.groupby(['platform', 'textblob_sentiment'])
        .size()
        .reset_index(name="Count")
    )

    fig4 = px.bar(
        textblob_dist,
        x="platform",
        y="Count",
        color="textblob_sentiment",
        barmode="group",
        title="TextBlob Sentiment Distribution"
    )

    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
    st.plotly_chart(fig4, use_container_width=True)


# =====================================================
# TAB 3 – Subjectivity Analysis
# =====================================================

with tab3:

    if "TextBlob_Subjectivity" in df.columns:

        st.subheader("Average Subjectivity by Platform")

        subjectivity_mean = (
            df.groupby("platform")["TextBlob_Subjectivity"]
            .mean()
            .reset_index()
        )

        fig5 = px.bar(
            subjectivity_mean,
            x="platform",
            y="TextBlob_Subjectivity",
            text="TextBlob_Subjectivity",
            title="Average Subjectivity Score (0 = Objective, 1 = Subjective)"
        )

        fig5.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')

        st.plotly_chart(fig5, use_container_width=True)
        
    else:
        st.warning("Subjectivity column not found in dataset.")


# =====================================================
# TAB 4 – Pie Charts
# =====================================================

with tab4:

    st.subheader("Sentiment Distribution – Pie Charts")

    color_map = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#3498db",
        "Positive Sentiment": "#2ecc71",
        "Negative Sentiment": "#e74c3c",
        "Neutral Sentiment": "#3498db",
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        bert_counts = df["bert_sentiment"].value_counts().reset_index()
        bert_counts.columns = ["Sentiment", "Count"]
        fig6 = px.pie(bert_counts, names="Sentiment", values="Count", title="BERT Sentiment",
                      color="Sentiment", color_discrete_map=color_map)
        fig6.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        vader_counts = df["Vader_Sentiment_Score"].value_counts().reset_index()
        vader_counts.columns = ["Sentiment", "Count"]
        fig7 = px.pie(vader_counts, names="Sentiment", values="Count", title="Vader Sentiment",
                      color="Sentiment", color_discrete_map=color_map)
        fig7.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
        st.plotly_chart(fig7, use_container_width=True)

    with col3:
        textblob_counts = df["textblob_sentiment"].value_counts().reset_index()
        textblob_counts.columns = ["Sentiment", "Count"]
        fig8 = px.pie(textblob_counts, names="Sentiment", values="Count", title="TextBlob Sentiment",
                      color="Sentiment", color_discrete_map=color_map)
        fig8.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#1e1b4b')
        st.plotly_chart(fig8, use_container_width=True)
        
