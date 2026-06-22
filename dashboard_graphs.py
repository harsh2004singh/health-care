import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO
import requests


class MaternalHealthDashboard:
    def __init__(self, api_endpoint):
        self.api_endpoint = api_endpoint
        self.maternal_health_data = self.fetch_data()

    def fetch_data(self):
        try:
            response = requests.get(self.api_endpoint, timeout=100)

            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text))

            st.error(f"API Error: {response.status_code}")
            return None

        except Exception as e:
            st.error(f"Connection Error: {e}")
            return None

    def drop_all_india(self, df):
        if df is not None and "State/UT" in df.columns:
            return df[df["State/UT"] != "All India"]
        return df

    def create_bubble_chart(self):
        if self.maternal_health_data is None:
            st.warning("No data available to render the Bubble Chart.")
            return

        df = self.drop_all_india(self.maternal_health_data)
        st.subheader("Bubble Chart: Performance vs Assessed Needs")
        st.write(
            "This chart maps regional performance. Larger bubbles indicate a higher percentage of achievement relative to target expectations.")

        # Determine size column dynamically to prevent API mapping mismatches
        size_col = None
        for col in df.columns:
            if "% Achvt" in col or "(E=" in col:
                size_col = col
                break

        fig = px.scatter(
            df,
            x="Need Assessed (2019-20) - (A)",
            y="Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)",
            size=size_col if size_col else "Need Assessed (2019-20) - (A)",
            color="State/UT",
            hover_name="State/UT",
            labels={
                "Need Assessed (2019-20) - (A)": "Need Assessed",
                "Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)": "Achievement Count"
            },
        )
        fig.update_layout(legend_title_text='Regions')
        st.plotly_chart(fig, use_container_width=True)

    def create_pie_chart(self):
        if self.maternal_health_data is None:
            return

        st.subheader("Proportion of Total Institutional Deliveries (April - June 2019-20)")
        df = self.drop_all_india(self.maternal_health_data)

        fig = px.pie(
            df,
            names="State/UT",
            values="Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)",
            labels={"Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)": "Deliveries"}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    def create_stacked_bar_chart(self):
        """Added method to prevent the structural `AttributeError` from your main entry point"""
        if self.maternal_health_data is None:
            return

        st.subheader("Regional Delivery Target Comparison")
        df = self.drop_all_india(self.maternal_health_data)

        # Creating comparative distribution frames
        fig = px.bar(
            df,
            x="State/UT",
            y=["Need Assessed (2019-20) - (A)",
               "Achievement during April to June - Total Institutional Deliveries - (2019-20) - (B)"],
            barmode="group",
            labels={"value": "Count", "variable": "Metric Types"}
        )
        st.plotly_chart(fig, use_container_width=True)

    def get_bubble_chart_data(self):
        return """
        ### 📊 Breakdown of the Bubble Chart Metrics

        * **Horizontal Axis (X-axis): Need Assessed (2019-20) - (A)** Represents baseline targeted maternal medical infrastructure capacity requirements across tracked territories.

        * **Vertical Axis (Y-axis): Total Institutional Deliveries Achieved - (B)** Tracks real absolute occurrences of clinical institutional deliveries managed successfully through the targeted window.

        * **Bubble Size Dimension (% Achievement)** Calculated dynamically via $\\frac{B}{A} \\times 100$. Scales out visually so anomalies or high efficiency models pop out instantly.
        """

    def get_pie_graph_data(self):
        return """
        ### 🍕 Share Breakdown Matrix

        * **Volume Proportions:** Slices illustrate real percentage distribution. It quickly highlights which specific states or territories are handling the largest operational load nationwide.
        """