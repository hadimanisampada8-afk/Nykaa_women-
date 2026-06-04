import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Nykaa Campaign Analytics Dashboard",
    page_icon="💄",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

[data-testid="metric-container"] {
    background-color: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}

h1,h2,h3 {
    color: #d63384;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/nykaa_campaign_data.csv")

df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("💄 Nykaa Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Campaign Analysis",
        "Channel Performance",
        "Customer Segments",
        "Conversion Funnel",
        "ROI Prediction"
    ]
)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.title("📊 Nykaa Campaign Analytics Dashboard")

    total_campaigns = len(df)

    revenue_col = "Revenue"
    roi_col = "ROI"

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Campaigns",
        f"{total_campaigns:,}"
    )

    c2.metric(
        "Total Revenue",
        f"₹ {df[revenue_col].sum():,.0f}"
    )

    c3.metric(
        "Average ROI",
        round(df[roi_col].mean(),2)
    )

    c4.metric(
        "Max ROI",
        round(df[roi_col].max(),2)
    )

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            df,
            x=roi_col,
            title="ROI Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.histogram(
            df,
            x=revenue_col,
            title="Revenue Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# --------------------------------------------------
# CAMPAIGN ANALYSIS
# --------------------------------------------------

elif page == "Campaign Analysis":

    st.title("📢 Campaign Analysis")

    campaign_col = "Campaign_Type"

    campaign_stats = (
        df.groupby(campaign_col)
        .agg({
            "Revenue":"sum",
            "ROI":"mean"
        })
        .reset_index()
    )

    fig = px.bar(
        campaign_stats,
        x=campaign_col,
        y="Revenue",
        title="Revenue by Campaign Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.box(
        df,
        x=campaign_col,
        y="ROI",
        title="ROI by Campaign Type"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# CHANNEL PERFORMANCE
# --------------------------------------------------

elif page == "Channel Performance":

    st.title("📱 Channel Performance")

    channel_col = "Channel_Used"

    channel_stats = (
        df.groupby(channel_col)
        .agg({
            "Revenue":"sum",
            "ROI":"mean"
        })
        .reset_index()
    )

    fig = px.bar(
        channel_stats,
        x=channel_col,
        y="Revenue",
        title="Revenue by Channel"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig = px.treemap(
        channel_stats,
        path=[channel_col],
        values="Revenue",
        color="ROI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# CUSTOMER SEGMENTS
# --------------------------------------------------

elif page == "Customer Segments":

    st.title("👥 Customer Segment Analysis")

    segment_col = "Customer_Segment"

    fig = px.pie(
        df,
        names=segment_col,
        title="Customer Segment Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    segment_stats = (
        df.groupby(segment_col)
        .agg({
            "Revenue":"sum"
        })
        .reset_index()
    )

    fig = px.bar(
        segment_stats,
        x=segment_col,
        y="Revenue",
        title="Revenue by Segment"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# FUNNEL ANALYSIS
# --------------------------------------------------

elif page == "Conversion Funnel":

    st.title("🔄 Conversion Funnel")

    impressions = df["Impressions"].sum()
    clicks = df["Clicks"].sum()
    leads = df["Leads"].sum()
    conversions = df["Conversions"].sum()

    funnel_df = pd.DataFrame({
        "Stage":[
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions"
        ],
        "Count":[
            impressions,
            clicks,
            leads,
            conversions
        ]
    })

    fig = px.funnel(
        funnel_df,
        x="Count",
        y="Stage"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# ROI PREDICTION
# --------------------------------------------------

elif page == "ROI Prediction":

    st.title("🤖 ROI Prediction")

    model_df = df.copy()

    categorical_cols = [
        "Campaign_Type",
        "Channel_Used",
        "Customer_Segment"
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        model_df[col] = le.fit_transform(
            model_df[col]
        )

        encoders[col] = le

    features = [
        "Campaign_Type",
        "Channel_Used",
        "Customer_Segment",
        "Impressions",
        "Clicks",
        "Leads",
        "Conversions"
    ]

    X = model_df[features]
    y = model_df["ROI"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train,y_train)

    campaign = st.selectbox(
        "Campaign Type",
        df["Campaign_Type"].unique()
    )

    channel = st.selectbox(
        "Channel",
        df["Channel_Used"].unique()
    )

    segment = st.selectbox(
        "Customer Segment",
        df["Customer_Segment"].unique()
    )

    impressions = st.number_input(
        "Impressions",
        value=100000
    )

    clicks = st.number_input(
        "Clicks",
        value=5000
    )

    leads = st.number_input(
        "Leads",
        value=500
    )

    conversions = st.number_input(
        "Conversions",
        value=100
    )

    if st.button("Predict ROI"):

        input_data = pd.DataFrame({

            "Campaign_Type":[
                encoders["Campaign_Type"].transform([campaign])[0]
            ],

            "Channel_Used":[
                encoders["Channel_Used"].transform([channel])[0]
            ],

            "Customer_Segment":[
                encoders["Customer_Segment"].transform([segment])[0]
            ],

            "Impressions":[impressions],
            "Clicks":[clicks],
            "Leads":[leads],
            "Conversions":[conversions]

        })

        prediction = model.predict(input_data)[0]

        st.success(
            f"Predicted ROI: {prediction:.2f}"
        )

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.sidebar.markdown("---")

if st.sidebar.checkbox("Show Dataset"):
    st.dataframe(df.head(100))
