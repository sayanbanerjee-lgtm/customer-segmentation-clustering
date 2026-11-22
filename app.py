import plotly.express as px
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from pathlib import Path

# -----------------------------------
# 1. Page configuration
# -----------------------------------

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# 2. Project file paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

RFM_PATH = PROJECT_ROOT / "customer_rfm.csv"

MODEL_PATH = PROJECT_ROOT / "customer_segmentation_model.joblib"

# -----------------------------------
# 3. Load saved data and model
# -----------------------------------

@st.cache_data
def load_rfm_data(path):
    return pd.read_csv(path, index_col=0)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_ipt_data():

    customer_ipt = pd.read_csv(
        PROJECT_ROOT / "customer_purchase_intervals.csv"
    )

    # Exclude customers without repeat purchases
    customer_ipt = customer_ipt.dropna(
        subset=["Days_Between_Purchases"]
    )

    return customer_ipt

try:
    rfm = load_rfm_data(RFM_PATH)
    artifacts = load_model(MODEL_PATH)

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

# -----------------------------------
# 4. Recreate customer clusters
# -----------------------------------

boxcox_lambdas = artifacts["boxcox_lambdas"]

scaler = artifacts["scaler"]
pca = artifacts["pca"]
kmeans = artifacts["kmeans"]

from scipy.stats import boxcox

rfm_transformed = rfm.copy()

for column in ["Frequency", "Monetary"]:
    rfm_transformed[column] = boxcox(
        rfm_transformed[column],
        lmbda=boxcox_lambdas[column]
    )

scaled_data = pd.DataFrame(
    scaler.transform(
        rfm_transformed[["Recency", "Frequency", "Monetary"]]
    ),
    columns=["Recency", "Frequency", "Monetary"],
    index=rfm.index
)

pca_data = pd.DataFrame(
    pca.transform(scaled_data)[:, :2],
    columns=["PC1", "PC2"],
    index=rfm.index
)

rfm["Cluster"] = kmeans.predict(pca_data)

# -----------------------------------
# 5. Dashboard heading
# -----------------------------------

st.title("📊 Customer Segmentation Dashboard")

st.subheader(
    "Customer Behaviour Analysis Using RFM and K-Means"
)

st.markdown("---")

# -----------------------------------
# 6. Sidebar
# -----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Overview",
        "Segment Analysis",
        "Customer Lookup",
        "New Customer Prediction",
        "Business Recommendations"
    ]
)

# -----------------------------------
# 7. Overview page
# -----------------------------------

if page == "Overview":

    st.subheader("Dataset Used for This Project")

    st.markdown(
        """
        This dashboard is developed using the
        **UCI Online Retail II Dataset**, containing
        transaction records from a UK-based online retailer
        between December 2009 and December 2011.

        - Original records: **1,067,371**
        - Valid purchase records after cleaning: **779,425**
        - Unique customers analysed: **5,878**

        **Dataset Source:** [UCI Machine Learning Repository – Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
        """
    )

    st.divider()


    st.header("Business Overview")

    total_customers = len(rfm)

    total_revenue = rfm["Monetary"].sum()

    total_invoices = int(rfm["Frequency"].sum())

    total_segments = rfm["Cluster"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Total Invoices",
        f"{total_invoices:,}"
    )

    col3.metric(
        "Positive Sales",
        f"£{total_revenue:,.2f}"
    )

    col4.metric(
        "Customer Segments",
        total_segments
    )

    st.markdown("---")

    st.subheader("Customer Distribution by Segment")

    cluster_counts = (
        rfm["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_counts)


    # RFM Distribution Analysis
    st.divider()

    st.subheader("Customer RFM Distribution")

    st.write(
        "Explore the distribution of Recency, Frequency "
        "and Monetary values across all customers."
    )

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Recency Histogram
    with col1:
        fig_recency = px.histogram(
            rfm,
            x="Recency",
            nbins=40,
            title="Recency Distribution",
            labels={"Recency": "Recency (Days)"}
        )

        fig_recency.update_layout(
            yaxis_title="Number of Customers",
            xaxis_title="Days Since Last Purchase",
            showlegend=False
        )

        st.plotly_chart(
            fig_recency,
            use_container_width=True
        )

    # Frequency Histogram
    with col2:
        fig_frequency = px.histogram(
            rfm,
            x="Frequency",
            nbins=40,
            title="Frequency Distribution",
            labels={"Frequency": "Number of Invoices"}
        )

        fig_frequency.update_layout(
            yaxis_title="Number of Customers",
            xaxis_title="Number of Invoices",
            showlegend=False
        )

        st.plotly_chart(
            fig_frequency,
            use_container_width=True
        )

    # Monetary Histogram
    with col3:
        fig_monetary = px.histogram(
            rfm,
            x="Monetary",
            nbins=40,
            title="Monetary Distribution",
            labels={"Monetary": "Total Spending (£)"}
        )

        fig_monetary.update_layout(
            yaxis_title="Number of Customers",
            xaxis_title="Total Spending (£)",
            showlegend=False
        )

        st.plotly_chart(
            fig_monetary,
            use_container_width=True
        )


    st.subheader("Project Overview")

    st.write(
        """
        This dashboard explores customer purchasing behaviour
        using RFM analysis, Box-Cox transformation,
        StandardScaler, PCA and K-Means clustering.

        The five customer segments help identify valuable
        customers, regular customers, one-time purchasers
        and inactive customers.
        """
    )

    st.info(
        "The segmentation model was trained on 5,878 customers "
        "using five K-Means clusters."
    )

    st.caption(
        "IDEAS-TIH Advanced Internship | Customer Segmentation Project"
    )


    st.divider()

    
    st.subheader("Explore the Customer-Level Dataset")

    st.write(
        "The following table provides a preview of the "
        "customer-level RFM dataset used in this "
        "customer segmentation project."
    )

    # Prepare customer-level dataset for display
    dataset_preview = rfm.reset_index()

    # Display first 100 customers
    st.dataframe(
        dataset_preview.head(100),
        use_container_width=True
    )

    # Download the complete customer-level RFM dataset
    st.download_button(
        label="Download Customer RFM Dataset (CSV)",
        data=dataset_preview.to_csv(index=False),
        file_name="customer_rfm.csv",
        mime="text/csv"
    )




# -----------------------------------
# 8. Segment Analysis page
# -----------------------------------

elif page == "Segment Analysis":

    st.header("Customer Segment Analysis")

    cluster_names = artifacts["cluster_names"]

    # Calculate segment profiles
    segment_profile = rfm.groupby("Cluster").agg(
        Customer_Count=("Recency", "count"),
        Mean_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Mean_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),
        Mean_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median")
    )

    segment_profile["Revenue"] = (
        rfm.groupby("Cluster")["Monetary"].sum()
    )

    segment_profile["Customer_Percentage"] = (
        segment_profile["Customer_Count"]
        / len(rfm) * 100
    )

    segment_profile["Revenue_Percentage"] = (
        segment_profile["Revenue"]
        / segment_profile["Revenue"].sum() * 100
    )

    segment_profile["Segment_Name"] = (
        segment_profile.index.map(cluster_names)
    )

    # -----------------------------------
    # Segment selector
    # -----------------------------------

    st.subheader("Explore Individual Segments")

    selected_cluster = st.selectbox(
        "Select a customer segment",
        options=sorted(rfm["Cluster"].unique()),
        format_func=lambda x: (
            f"Cluster {x} - {cluster_names[x]}"
        )
    )

    selected_profile = segment_profile.loc[selected_cluster]

    st.subheader(
        f"Cluster {selected_cluster}: "
        f"{cluster_names[selected_cluster]}"
    )

    # -----------------------------------
    # Segment metrics
    # -----------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{int(selected_profile['Customer_Count']):,}"
    )

    col2.metric(
        "Customer Share",
        f"{selected_profile['Customer_Percentage']:.2f}%"
    )

    col3.metric(
        "Revenue Contribution",
        f"{selected_profile['Revenue_Percentage']:.2f}%"
    )

    st.markdown("---")

    # -----------------------------------
    # RFM profile
    # -----------------------------------

    st.subheader("RFM Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean RFM Values**")

        mean_profile = pd.DataFrame({
            "Metric": ["Recency", "Frequency", "Monetary"],
            "Mean Value": [
                selected_profile["Mean_Recency"],
                selected_profile["Mean_Frequency"],
                selected_profile["Mean_Monetary"]
            ]
        })

        st.dataframe(
            mean_profile.style.format({
                "Mean Value": "{:,.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    with col2:
        st.markdown("**Median RFM Values**")

        median_profile = pd.DataFrame({
            "Metric": ["Recency", "Frequency", "Monetary"],
            "Median Value": [
                selected_profile["Median_Recency"],
                selected_profile["Median_Frequency"],
                selected_profile["Median_Monetary"]
            ]
        })

        st.dataframe(
            median_profile.style.format({
                "Median Value": "{:,.2f}"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.markdown("---")

    # -----------------------------------
    # Revenue comparison
    # -----------------------------------

    st.subheader("Revenue Contribution by Segment")

    revenue_chart = segment_profile.reset_index()

    fig_revenue = px.bar(
        revenue_chart,
        x="Segment_Name",
        y="Revenue_Percentage",
        color="Segment_Name",
        text="Revenue_Percentage",
        labels={
            "Segment_Name": "Customer Segment",
            "Revenue_Percentage": "Revenue Contribution (%)"
        },
        title="Percentage of Total Positive Sales"
    )

    fig_revenue.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig_revenue.update_layout(
        showlegend=False,
        yaxis_title="Revenue Contribution (%)",
        xaxis_title="Customer Segment"
    )

    st.plotly_chart(
        fig_revenue,
        use_container_width=True
    )


    # RFM Radar Chart
    st.divider()

    st.subheader("Customer Segment RFM Radar Chart")

    st.write(
        "Compare the five customer segments using "
        "mean standardized RFM values."
    )

    
    # Recreate standardized RFM data using the saved scaler
    rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # Apply the saved Box-Cox transformations
    rfm_transformed = rfm_features.copy()

    rfm_transformed["Frequency"] = boxcox(
        rfm_features["Frequency"],
        lmbda=artifacts["boxcox_lambdas"]["Frequency"]
    )

    rfm_transformed["Monetary"] = boxcox(
        rfm_features["Monetary"],
        lmbda=artifacts["boxcox_lambdas"]["Monetary"]
    )

    # Apply the previously fitted scaler
    rfm_scaled_app = pd.DataFrame(
        artifacts["scaler"].transform(rfm_transformed),
        columns=["Recency", "Frequency", "Monetary"],
        index=rfm.index
    )

    # Attach cluster labels
    radar_data = rfm_scaled_app.copy()
    radar_data["Cluster"] = rfm["Cluster"]


    # Calculate mean standardized RFM values per cluster
    radar_profile = (
        radar_data.groupby("Cluster")[
            ["Recency", "Frequency", "Monetary"]
        ]
        .mean()
        .reset_index()
    )

    # Convert cluster numbers into business-friendly names
    radar_profile["Segment"] = (
        radar_profile["Cluster"].map(artifacts["cluster_names"])
    )

    
    # Reshape data into long format for Plotly
    radar_long = radar_profile.melt(
        id_vars=["Cluster", "Segment"],
        value_vars=["Recency", "Frequency", "Monetary"],
        var_name="RFM Variable",
        value_name="Mean Standardized Value"
    )

    # Create radar chart
    fig_radar = px.line_polar(
        radar_long,
        r="Mean Standardized Value",
        theta="RFM Variable",
        color="Segment",
        line_close=True,
        title="RFM Comparison Across Five Customer Segments"
    )

    fig_radar.update_traces(
        fill="toself",
        opacity=0.55
    )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Mean Standardized Value"
            )
        ),
        showlegend=True
    )

    st.plotly_chart(
        fig_radar,
        use_container_width=True
    )


    # Purchase Interval Analysis
    st.divider()

    st.subheader("Purchase Interval Analysis")

    st.write(
        "Analyze the average number of days between "
        "consecutive purchases across customer segments."
    )

    # Load customer IPT data
    customer_ipt = load_ipt_data()

    # Attach cluster information to IPT data
    ipt_analysis = customer_ipt.merge(
        rfm[["Cluster"]],
        left_on="CustomerID",
        right_index=True,
        how="inner"
    )

    # Calculate segment-level IPT coverage
    ipt_coverage = (
        ipt_analysis.groupby("Cluster")
        .size()
        .rename("Customers_With_IPT")
        .to_frame()
    )

    ipt_coverage["Total_Customers"] = (
        rfm.groupby("Cluster").size()
    )

    ipt_coverage["Coverage_Percentage"] = (
        ipt_coverage["Customers_With_IPT"]
        / ipt_coverage["Total_Customers"] * 100
    )

    st.write("### Purchase Interval Coverage")

    st.dataframe(
        ipt_coverage.round(2),
        use_container_width=True
    )


    # Interactive IPT Boxplot
    st.write("### Purchase Interval Distribution by Segment")

    # Add business-friendly segment names
    ipt_analysis["Segment"] = (
        ipt_analysis["Cluster"].map(
            artifacts["cluster_names"]
        )
    )

    # Create interactive boxplot
    fig_ipt = px.box(
        ipt_analysis,
        x="Segment",
        y="Days_Between_Purchases",
        color="Segment",
        points="outliers",
        title="Average Inter-Purchase Time Across Customer Segments",
        labels={
            "Segment": "Customer Segment",
            "Days_Between_Purchases": "Average Days Between Purchases"
        },
        hover_data=["CustomerID"]
    )

    fig_ipt.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Average Purchase Interval (Days)",
        showlegend=False,
        xaxis_tickangle=-25
    )

    st.caption(
        "IPT coverage represents the percentage of customers "
        "with at least two distinct invoices. Customers with "
        "only one invoice do not have a measurable "
        "inter-purchase interval."
    )

    st.plotly_chart(
        fig_ipt,
        use_container_width=True,
        key="ipt_segment_boxplot"
    )
    
    st.markdown("---")


    # -----------------------------------
    # Full segment comparison
    # -----------------------------------

    st.subheader("Comparison of All Customer Segments")

    comparison_table = segment_profile.reset_index()

    display_columns = [
        "Cluster",
        "Segment_Name",
        "Customer_Count",
        "Customer_Percentage",
        "Mean_Recency",
        "Mean_Frequency",
        "Mean_Monetary",
        "Revenue",
        "Revenue_Percentage"
    ]

    st.dataframe(
        comparison_table[display_columns].style.format({
            "Customer_Percentage": "{:.2f}%",
            "Mean_Recency": "{:.2f}",
            "Mean_Frequency": "{:.2f}",
            "Mean_Monetary": "£{:,.2f}",
            "Revenue": "£{:,.2f}",
            "Revenue_Percentage": "{:.2f}%"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------
    # PCA visualization
    # -----------------------------------

    st.subheader("Customer Distribution in PCA Space")

    pca_plot_data = pca_data.copy()

    pca_plot_data["Cluster"] = (
        rfm["Cluster"].astype(str)
    )

    pca_plot_data["CustomerID"] = rfm.index.astype(str)

    pca_plot_data["Segment_Name"] = (
        pca_plot_data["Cluster"].astype(int).map(cluster_names)
    )

    fig_pca = px.scatter(
        pca_plot_data,
        x="PC1",
        y="PC2",
        color="Segment_Name",
        hover_data=["CustomerID"],
        title="Five Customer Segments in Two-Dimensional PCA Space",
        opacity=0.65
    )

    fig_pca.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    st.caption(
        "PCA is used to visualize customer similarity. "
        "The clustering model was trained using these two components."
    )


# -----------------------------------
# 9. Customer Lookup page
# -----------------------------------

elif page == "Customer Lookup":

    st.header("🔎 Customer Lookup")

    st.write(
        "Enter a CustomerID to explore their purchasing "
        "behaviour and assigned customer segment."
    )

    cluster_names = artifacts["cluster_names"]

    # -----------------------------------
    # Customer ID input
    # -----------------------------------

    customer_id = st.number_input(
        "Enter Customer ID",
        min_value=0,
        value=12347,
        step=1
    )

    search_button = st.button(
        "Find Customer",
        type="primary"
    )

    if search_button:

        # Search customer in RFM dataset
        customer_match = (
    pd.to_numeric(rfm.index, errors="coerce") == int(customer_id)
)

        if customer_match.any():

            customer = rfm.loc[customer_match].iloc[0]

            cluster = int(customer["Cluster"])

            segment_name = cluster_names[cluster]

            # -----------------------------------
            # Customer information
            # -----------------------------------

            st.success("Customer found!")

            st.subheader("Customer Profile")

            st.write(f"**Customer ID:** {customer_id}")

            st.write(f"**Assigned Cluster:** {cluster}")

            st.write(f"**Customer Segment:** {segment_name}")

            st.markdown("---")

            # -----------------------------------
            # RFM metrics
            # -----------------------------------

            st.subheader("Customer RFM Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Recency",
                f"{int(customer['Recency'])} days"
            )

            col2.metric(
                "Frequency",
                f"{int(customer['Frequency'])} invoices"
            )

            col3.metric(
                "Monetary",
                f"£{customer['Monetary']:,.2f}"
            )

            st.markdown("---")

            # -----------------------------------
            # Segment comparison
            # -----------------------------------

            cluster_customers = rfm[
                rfm["Cluster"] == cluster
            ]

            customer_share = (
                len(cluster_customers) / len(rfm) * 100
            )

            revenue_share = (
                cluster_customers["Monetary"].sum()
                / rfm["Monetary"].sum() * 100
            )

            st.subheader("Customer Segment Information")

            col1, col2 = st.columns(2)

            col1.metric(
                "Segment Customer Share",
                f"{customer_share:.2f}%"
            )

            col2.metric(
                "Segment Revenue Contribution",
                f"{revenue_share:.2f}%"
            )

            st.info(
                f"This customer belongs to the "
                f"**{segment_name}** segment."
            )

            # -----------------------------------
            # Compare customer with segment average
            # -----------------------------------

            st.subheader(
                "Customer vs Segment Average"
            )

            comparison = pd.DataFrame({
                "Metric": [
                    "Recency (days)",
                    "Frequency (invoices)",
                    "Monetary (£)"
                ],
                "Customer": [
                    customer["Recency"],
                    customer["Frequency"],
                    customer["Monetary"]
                ],
                "Segment Average": [
                    cluster_customers["Recency"].mean(),
                    cluster_customers["Frequency"].mean(),
                    cluster_customers["Monetary"].mean()
                ]
            })

            st.dataframe(
                comparison.style.format({
                    "Customer": "{:,.2f}",
                    "Segment Average": "{:,.2f}"
                }),
                hide_index=True,
                use_container_width=True
            )

        else:

            st.error(
                f"Customer ID {customer_id} was not found "
                "in the customer dataset. Please check the ID."
            )

        
# 10. New Customer Prediction page
elif page == "New Customer Prediction":

    st.header("New Customer Segment Prediction")

    st.write(
        "Enter a customer's RFM values to predict "
        "which customer segment they belong to."
    )

    st.info(
        "Enter Recency in days, Frequency as the number of "
        "distinct invoices, and Monetary as total spending in GBP."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        new_recency = st.number_input(
            "Recency (days)",
            min_value=0,
            value=30,
            step=1
        )

    with col2:
        new_frequency = st.number_input(
            "Frequency (invoices)",
            min_value=1,
            value=5,
            step=1
        )

    with col3:
        new_monetary = st.number_input(
            "Monetary (£)",
            min_value=0.01,
            value=1500.0,
            step=100.0
        )

    if st.button("Predict Customer Segment", type="primary"):

        # Create input DataFrame
        new_customer = pd.DataFrame({
            "Recency": [new_recency],
            "Frequency": [new_frequency],
            "Monetary": [new_monetary]
        })

        # Apply the previously fitted Box-Cox transformations
        transformed_customer = new_customer.copy()

        transformed_customer["Frequency"] = boxcox(
            new_customer["Frequency"],
            lmbda=artifacts["boxcox_lambdas"]["Frequency"]
        )

        transformed_customer["Monetary"] = boxcox(
            new_customer["Monetary"],
            lmbda=artifacts["boxcox_lambdas"]["Monetary"]
        )

        # Apply the previously fitted scaler
        scaled_customer = pd.DataFrame(
            artifacts["scaler"].transform(transformed_customer),
            columns=["Recency", "Frequency", "Monetary"]
        )

        # Apply the previously fitted PCA model
        customer_pca = pd.DataFrame(
            artifacts["pca"].transform(scaled_customer)[:, :2],
            columns=["PC1", "PC2"]
        )

        # Predict the cluster
        predicted_cluster = int(
            artifacts["kmeans"].predict(customer_pca)[0]
        )

        # Get the business-friendly segment name
        segment_name = artifacts["cluster_names"][predicted_cluster]

        st.success("Prediction completed successfully!")

        st.subheader(f"Predicted Segment: {segment_name}")

        st.metric("Assigned Cluster", predicted_cluster)

        st.write("### Entered Customer RFM")

        st.dataframe(new_customer, use_container_width=True)

        st.write(
            "The prediction uses the same fitted "
            "transformation, scaling, PCA and K-Means "
            "pipeline as the original customer segmentation."
        )


# 11. Business Recommendations page
elif page == "Business Recommendations":

    st.title("Business Recommendations")
    st.write(
        "Explore targeted marketing strategies for each customer "
        "segment based on purchasing behaviour and revenue contribution."
    )

    # Calculate customer and revenue profiles
    cluster_summary = rfm.groupby("Cluster").agg(
        Customer_Count=("Recency", "size"),
        Mean_Recency=("Recency", "mean"),
        Mean_Frequency=("Frequency", "mean"),
        Mean_Monetary=("Monetary", "mean"),
        Revenue=("Monetary", "sum")
    )

    cluster_summary["Customer_Percentage"] = (
        cluster_summary["Customer_Count"] / len(rfm) * 100
    )

    cluster_summary["Revenue_Percentage"] = (
        cluster_summary["Revenue"] / rfm["Monetary"].sum() * 100
    )

    # Recommendations for each segment
    recommendations = {
        0: {
            "objective": "Retain valuable customers and strengthen loyalty.",
            "strategies": [
                "Offer VIP membership and exclusive loyalty benefits.",
                "Provide early access to new products and special offers.",
                "Use personalized product recommendations.",
                "Reward repeat purchases with loyalty points."
            ]
        },
        1: {
            "objective": "Convert one-time buyers into repeat customers.",
            "strategies": [
                "Send a welcome-back email after the first purchase.",
                "Offer a limited-time incentive for the second purchase.",
                "Recommend complementary products based on the first order.",
                "Use low-cost reminders to encourage another purchase."
            ]
        },
        2: {
            "objective": "Increase purchase frequency and customer value.",
            "strategies": [
                "Introduce loyalty points and repeat-purchase rewards.",
                "Use cross-selling and upselling campaigns.",
                "Offer personalized product bundles.",
                "Provide targeted promotions based on purchase history."
            ]
        },
        3: {
            "objective": "Reactivate customers with an established purchase history.",
            "strategies": [
                "Launch targeted win-back email campaigns.",
                "Offer personalized comeback incentives.",
                "Remind customers about products they previously purchased.",
                "Highlight relevant new products and seasonal offers."
            ]
        },
        4: {
            "objective": "Attempt cost-effective recovery of long-inactive customers.",
            "strategies": [
                "Test low-cost reactivation emails.",
                "Offer selective comeback discounts.",
                "Avoid expensive campaigns until engagement improves.",
                "Monitor responses before investing further in this segment."
            ]
        }
    }

    # Select a customer segment
    selected_cluster = st.selectbox(
        "Select a Customer Segment",
        options=sorted(cluster_summary.index),
        format_func=lambda cluster: (
            f"Cluster {cluster}: "
            f"{artifacts['cluster_names'][int(cluster)]}"
        )
    )

    selected_profile = cluster_summary.loc[selected_cluster]
    selected_recommendation = recommendations[int(selected_cluster)]

    segment_name = artifacts["cluster_names"][int(selected_cluster)]

    st.divider()

    st.subheader(f"Cluster {selected_cluster}: {segment_name}")

    # Key business metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        f"{int(selected_profile['Customer_Count']):,}"
    )

    col2.metric(
        "Customer Share",
        f"{selected_profile['Customer_Percentage']:.2f}%"
    )

    col3.metric(
        "Revenue Contribution",
        f"{selected_profile['Revenue_Percentage']:.2f}%"
    )

    # Customer behaviour profile
    st.subheader("Customer Behaviour Profile")

    profile_data = pd.DataFrame({
        "Metric": [
            "Average Recency (days)",
            "Average Purchase Frequency",
            "Average Monetary Value (£)",
            "Total Revenue (£)"
        ],
        "Value": [
            f"{selected_profile['Mean_Recency']:.2f}",
            f"{selected_profile['Mean_Frequency']:.2f}",
            f"£{selected_profile['Mean_Monetary']:,.2f}",
            f"£{selected_profile['Revenue']:,.2f}"
        ]
    })

    st.dataframe(profile_data, hide_index=True, use_container_width=True)

    # Business objective
    st.subheader("Business Objective")
    st.write(selected_recommendation["objective"])

    # Recommended strategies
    st.subheader("Recommended Marketing Strategies")

    for strategy in selected_recommendation["strategies"]:
        st.markdown(f"- {strategy}")

    # Business interpretation
    st.subheader("Business Interpretation")

    if selected_cluster == 0:
        st.info(
            "This segment generates the majority of business revenue. "
            "Customer retention should be the primary priority."
        )
    elif selected_cluster == 1:
        st.info(
            "Most customers in this segment have made only one purchase. "
            "Encouraging a second purchase is the key opportunity."
        )
    elif selected_cluster == 2:
        st.info(
            "This is a large group of relatively regular customers. "
            "Increasing their purchase frequency and average spending "
            "could improve revenue."
        )
    elif selected_cluster == 3:
        st.info(
            "These customers have purchased repeatedly in the past but "
            "have been inactive for a considerable period. "
            "Reactivation campaigns may recover some of their value."
        )
    else:
        st.info(
            "These customers are generally long-inactive and have low "
            "purchase frequency and spending. Use inexpensive campaigns "
            "to test whether they can be re-engaged."
        )

