"""
SmartFuel AI — AI Manager Assistant
======================================
A controlled natural-language query layer that answers store-manager questions
using ONLY real data from the cleaned dataset, forecast outputs and inventory
calculations.

Design principle
----------------
This assistant does NOT use an external LLM API.  All numerical answers are
derived from the actual data.  This makes the system:
  (a) reliable offline
  (b) academically honest — no hallucinated numbers
  (c) deterministic and reproducible

The NLP layer uses keyword/intent matching rather than a neural language model,
which is appropriate for this academic prototype.  Future work could integrate
a local LLM (e.g., Ollama/LLaMA) using the same data-query layer as the
ground-truth source.
"""

import re
import numpy as np
import pandas as pd
from typing import Optional


# ── Intent definitions ────────────────────────────────────────────────────
INTENTS = {
    "reorder":       r"reorder|replenish|order|run out|run low|stock up|top.?up",
    "stockout":      r"stockout|stock.?out|out of stock|risk|danger|urgent",
    "top_sellers":   r"top sell|best sell|highest demand|most popular|best perform",
    "slow_movers":   r"slow.?mov|worst sell|low demand|least popular|poor perform",
    "forecast":      r"forecast|predict|expect|next week|next month|future demand|upcoming",
    "inventory":     r"inventory|stock level|on hand|current stock",
    "category":      r"category|categor|department|section|type",
    "revenue":       r"revenue|sales value|income|rand|price|earnings",
    "compare":       r"compare|comparison|versus|vs\.?|against",
    "trend":         r"trend|over time|growing|declining|increase|decrease|pattern",
    "weather":       r"weather|rain|sunny|cloudy",
    "promotion":     r"promot|discount|sale|special",
    "help":          r"\bhelp\b|what can you|what do you|commands|options|guide",
    "greeting":      r"^(hi|hello|hey|good morning|good afternoon|good evening|howzit)[\s!.,?]*$",
}


def _match_intent(query: str) -> str:
    """Return the first matching intent label, or 'unknown'."""
    ql = query.lower().strip()
    for intent, pattern in INTENTS.items():
        if re.search(pattern, ql):
            return intent
    return "unknown"


def _extract_product_mention(query: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Check whether the query mentions a product name or ID and return
    the filtered rows, or None if no product-specific match is found.
    """
    ql = query.lower()
    for _, row in df.drop_duplicates("Product Name").iterrows():
        name = str(row.get("Product Name", "")).lower()
        pid  = str(row.get("Product ID", "")).lower()
        if name in ql or pid in ql or name.split("(")[0].strip() in ql:
            return df[df["Product ID"] == row["Product ID"]].copy()
    return None


def _extract_category_mention(query: str, categories: list) -> Optional[str]:
    """Return the matching category string if mentioned in the query."""
    ql = query.lower()
    for cat in categories:
        if cat.lower() in ql:
            return cat
        # also try partial match on first word
        first_word = cat.lower().split()[0]
        if len(first_word) > 3 and first_word in ql:
            return cat
    return None


class ManagerAssistant:
    """
    AI Manager Assistant for SmartFuel AI.

    Usage
    -----
    assistant = ManagerAssistant(df_processed, forecast_df)
    response  = assistant.ask("Which products should I reorder?")
    # response is a dict: {'text': str, 'table': DataFrame|None, 'chart_hint': str|None}
    """

    def __init__(
        self,
        df_processed: pd.DataFrame,
        forecast_df: pd.DataFrame,
        metrics_df: Optional[pd.DataFrame] = None,
        test_preds_df: Optional[pd.DataFrame] = None,
    ):
        self.df  = df_processed.copy()
        self.fc  = forecast_df.copy()
        self.metrics = metrics_df
        self.preds   = test_preds_df

        # Ensure Inventory columns exist in forecast
        for col in ["Recommended Order", "Stockout Risk", "Safety Stock",
                    "Total Predicted Demand"]:
            if col not in self.fc.columns and "Predicted Demand" in self.fc.columns:
                if col == "Total Predicted Demand":
                    self.fc["Total Predicted Demand"] = self.fc.get("Predicted Demand", 0)

        self._categories = sorted(self.df["Category"].dropna().unique().tolist())
        self._products   = sorted(self.df["Product Name"].dropna().unique().tolist())
        self._stores     = sorted(self.df["Store ID"].dropna().unique().tolist())

        # Pre-compute KPIs
        self._total_demand    = int(self.df["Demand"].sum())
        self._total_units     = int(self.df["Units Sold"].sum()) if "Units Sold" in self.df.columns else self._total_demand
        self._date_min        = self.df["Date"].min().strftime("%d %b %Y")
        self._date_max        = self.df["Date"].max().strftime("%d %b %Y")
        self._horizon         = int(self.fc.get("Forecast Horizon Days", pd.Series([7])).iloc[0]) if "Forecast Horizon Days" in self.fc.columns else 7

    # ── Public API ─────────────────────────────────────────────────────────
    def ask(self, query: str) -> dict:
        """
        Process a natural-language question and return a structured response.

        Returns
        -------
        dict with keys:
            'text'       : str — main answer paragraph
            'table'      : pd.DataFrame or None
            'chart_hint' : str or None — hint for the dashboard to show a chart
        """
        if not query or not query.strip():
            return self._wrap("Please type a question and I'll do my best to help.")

        intent = _match_intent(query)

        handlers = {
            "greeting":    self._handle_greeting,
            "help":        self._handle_help,
            "reorder":     self._handle_reorder,
            "stockout":    self._handle_stockout,
            "top_sellers": self._handle_top_sellers,
            "slow_movers": self._handle_slow_movers,
            "forecast":    self._handle_forecast,
            "inventory":   self._handle_inventory,
            "category":    self._handle_category,
            "revenue":     self._handle_revenue,
            "compare":     self._handle_compare,
            "trend":       self._handle_trend,
            "weather":     self._handle_weather,
            "promotion":   self._handle_promotion,
            "unknown":     self._handle_unknown,
        }

        handler = handlers.get(intent, self._handle_unknown)
        return handler(query)

    # ── Intent handlers ────────────────────────────────────────────────────
    def _handle_greeting(self, query: str) -> dict:
        n_high = int((self.fc["Stockout Risk"] == "HIGH").sum()) if "Stockout Risk" in self.fc.columns else 0
        text = (
            f"Hello! I'm your SmartFuel AI Manager Assistant.\n\n"
            f"Here's a quick snapshot of your store:\n"
            f"• Data covers {self._date_min} to {self._date_max}\n"
            f"• {len(self._products)} products tracked across {len(self._stores)} branches\n"
            f"• {n_high} product/store combinations at HIGH stockout risk right now\n\n"
            f"Ask me anything — try *'Which products need reordering?'* or *'Show top sellers'*."
        )
        return self._wrap(text)

    def _handle_help(self, query: str) -> dict:
        text = (
            "**Here are some things you can ask me:**\n\n"
            "📦 **Inventory & Reordering**\n"
            "- *Which products should I reorder?*\n"
            "- *Which products are at high stockout risk?*\n"
            "- *What are the current stock levels?*\n\n"
            "📈 **Sales & Performance**\n"
            "- *What are the top selling products?*\n"
            "- *Which products are slow moving?*\n"
            "- *How is the Beverages category performing?*\n"
            "- *Compare Snacks and Beverages*\n\n"
            "🔮 **Demand Forecasting**\n"
            "- *What is the expected demand for Bottled Water?*\n"
            "- *Show the demand forecast*\n\n"
            "📊 **Trends & Analysis**\n"
            "- *What are my sales trends?*\n"
            "- *What is the impact of promotions?*\n"
            "- *How does weather affect demand?*"
        )
        return self._wrap(text)

    def _handle_reorder(self, query: str) -> dict:
        if self.fc.empty:
            return self._wrap("No forecast data available. Please run the training pipeline first.")

        fc = self.fc.copy()
        if "Recommended Order" not in fc.columns:
            return self._wrap("Reorder data is not available in the current forecast.")

        reorder = fc[fc["Recommended Order"] > 0].copy()
        from src.inventory import RISK_ORDER
        reorder["_r"] = reorder["Stockout Risk"].map(RISK_ORDER).fillna(99)
        reorder = reorder.sort_values(["_r", "Recommended Order"], ascending=[True, False])
        reorder = reorder.drop(columns=["_r"], errors="ignore")

        n      = len(reorder)
        n_high = int((reorder["Stockout Risk"] == "HIGH").sum())
        total  = int(reorder["Recommended Order"].sum())

        text = (
            f"**{n} product/branch combinations require replenishment** "
            f"over the next {self._horizon}-day forecast period.\n\n"
            f"• **{n_high}** items are at HIGH stockout risk — order immediately.\n"
            f"• Total units to order: **{total:,}**\n\n"
            f"The table below is sorted by urgency."
        )
        cols = [c for c in ["Store ID", "Product Name", "Category",
                             "Inventory Level", "Total Predicted Demand",
                             "Safety Stock", "Recommended Order", "Stockout Risk"]
                if c in reorder.columns]
        return self._wrap(text, table=reorder[cols].head(20), chart_hint="reorder_bar")

    def _handle_stockout(self, query: str) -> dict:
        if self.fc.empty or "Stockout Risk" not in self.fc.columns:
            return self._wrap("No stockout data available. Please run the training pipeline first.")

        high = self.fc[self.fc["Stockout Risk"] == "HIGH"].copy()
        if high.empty:
            return self._wrap("✅ Great news! No products are currently at HIGH stockout risk.")

        high = high.sort_values("Recommended Order", ascending=False)
        text = (
            f"⚠️ **{len(high)} product/branch combinations are at HIGH stockout risk.**\n\n"
            f"These products have current inventory BELOW forecast demand for the "
            f"next {self._horizon} days. Immediate restocking is recommended."
        )
        cols = [c for c in ["Store ID", "Product Name", "Category",
                             "Inventory Level", "Total Predicted Demand",
                             "Recommended Order", "Stockout Risk"]
                if c in high.columns]
        return self._wrap(text, table=high[cols].head(20), chart_hint="risk_chart")

    def _handle_top_sellers(self, query: str) -> dict:
        agg = (
            self.df.groupby(["Product ID", "Product Name", "Category"], as_index=False)
            .agg(Total_Demand=("Demand", "sum"), Avg_Daily=("Demand", "mean"))
            .sort_values("Total_Demand", ascending=False)
        )

        # Check if a category was mentioned
        cat = _extract_category_mention(query, self._categories)
        if cat:
            agg = agg[agg["Category"] == cat]
            prefix = f"in {cat}"
        else:
            prefix = "across all categories"

        top = agg.head(10)
        text = (
            f"**Top 10 best-selling products {prefix}** "
            f"(by total demand, {self._date_min} – {self._date_max}):\n\n"
            + "\n".join(
                f"{i+1}. **{row['Product Name']}** — {int(row['Total_Demand']):,} units"
                for i, row in top.iterrows()
            )
        )
        return self._wrap(text, table=top.rename(columns={
            "Total_Demand": "Total Demand", "Avg_Daily": "Avg Daily Demand"
        }), chart_hint="top_sellers_bar")

    def _handle_slow_movers(self, query: str) -> dict:
        agg = (
            self.df.groupby(["Product ID", "Product Name", "Category"], as_index=False)
            .agg(Total_Demand=("Demand", "sum"), Avg_Daily=("Demand", "mean"))
            .sort_values("Total_Demand", ascending=True)
        )
        slow = agg.head(10)
        text = (
            f"**Bottom 10 slowest-moving products** "
            f"(by total demand, {self._date_min} – {self._date_max}):\n\n"
            + "\n".join(
                f"{i+1}. **{row['Product Name']}** — {int(row['Total_Demand']):,} units total"
                for i, row in slow.iterrows()
            )
            + "\n\n_Consider reducing stock levels or running promotions on these items._"
        )
        return self._wrap(text, table=slow.rename(columns={
            "Total_Demand": "Total Demand", "Avg_Daily": "Avg Daily Demand"
        }), chart_hint="slow_movers_bar")

    def _handle_forecast(self, query: str) -> dict:
        if self.fc.empty:
            return self._wrap("No forecast available. Please run the training pipeline first.")

        # Check if a specific product is mentioned
        product_match = _extract_product_mention(query, self.fc)
        if product_match is not None and not product_match.empty:
            row = product_match.iloc[0]
            demand_col = "Total Predicted Demand" if "Total Predicted Demand" in row.index else "Predicted Demand"
            pred   = row.get(demand_col, 0)
            inv    = row.get("Inventory Level", 0)
            order  = row.get("Recommended Order", 0)
            risk   = row.get("Stockout Risk", "UNKNOWN")
            name   = row.get("Product Name", row.get("Product ID", "This product"))
            text = (
                f"**{name}** — {self._horizon}-day forecast:\n\n"
                f"• Predicted demand: **{pred:.0f} units**\n"
                f"• Current stock: **{inv:.0f} units**\n"
                f"• Recommended replenishment: **{order:.0f} units**\n"
                f"• Stockout risk: **{risk}**"
            )
            return self._wrap(text, table=product_match.head(10), chart_hint="forecast_line")

        # General forecast summary
        demand_col = "Total Predicted Demand" if "Total Predicted Demand" in self.fc.columns else "Predicted Demand"
        total_pred = self.fc[demand_col].sum()
        avg_pred   = self.fc[demand_col].mean()
        n_high     = int((self.fc["Stockout Risk"] == "HIGH").sum()) if "Stockout Risk" in self.fc.columns else 0

        top = self.fc.sort_values(demand_col, ascending=False).head(5)
        top_list = "\n".join(
            f"  {i+1}. {row.get('Product Name', row.get('Product ID', ''))}: "
            f"{row[demand_col]:.0f} units"
            for i, row in top.iterrows()
        )
        text = (
            f"**{self._horizon}-day demand forecast summary:**\n\n"
            f"• Total predicted demand: **{total_pred:,.0f} units**\n"
            f"• Average per product/branch: **{avg_pred:.0f} units**\n"
            f"• Products at HIGH stockout risk: **{n_high}**\n\n"
            f"**Highest demand products:**\n{top_list}"
        )
        return self._wrap(text, chart_hint="forecast_bar")

    def _handle_inventory(self, query: str) -> dict:
        if self.fc.empty:
            return self._wrap("No inventory data available. Please run the training pipeline first.")

        if "Inventory Level" not in self.fc.columns:
            return self._wrap("Inventory levels are not available in the current forecast data.")

        product_match = _extract_product_mention(query, self.fc)
        if product_match is not None and not product_match.empty:
            row  = product_match.iloc[0]
            name = row.get("Product Name", row.get("Product ID", "This product"))
            text = (
                f"**{name}** — current inventory: **{row['Inventory Level']:.0f} units**\n"
                f"• Stockout risk: {row.get('Stockout Risk', 'N/A')}\n"
                f"• Recommended order: {row.get('Recommended Order', 0):.0f} units"
            )
            return self._wrap(text, table=product_match.head(5))

        summary = self.fc.groupby("Category", as_index=False)["Inventory Level"].sum()
        text = (
            f"**Current inventory by category** ({self._horizon}-day planning horizon):\n\n"
            + "\n".join(
                f"• {row['Category']}: **{int(row['Inventory Level']):,} units** on hand"
                for _, row in summary.iterrows()
            )
        )
        return self._wrap(text, table=summary, chart_hint="inventory_bar")

    def _handle_category(self, query: str) -> dict:
        cat = _extract_category_mention(query, self._categories)

        agg = (
            self.df.groupby("Category", as_index=False)
            .agg(
                Total_Demand=("Demand", "sum"),
                Avg_Daily=("Demand", "mean"),
                Products=("Product ID", "nunique"),
            )
            .sort_values("Total_Demand", ascending=False)
        )

        if cat:
            row = agg[agg["Category"] == cat]
            if row.empty:
                return self._wrap(f"No data found for category: {cat}.")
            r = row.iloc[0]
            text = (
                f"**{cat}** category performance:\n\n"
                f"• Total demand: **{int(r['Total_Demand']):,} units**\n"
                f"• Average daily demand: **{r['Avg_Daily']:.1f} units**\n"
                f"• Number of products: **{int(r['Products'])}**"
            )
            cat_df = self.df[self.df["Category"] == cat].groupby(
                ["Product ID", "Product Name"], as_index=False
            ).agg(Total=("Demand", "sum")).sort_values("Total", ascending=False)
            return self._wrap(text, table=cat_df)

        text = (
            f"**Category performance** ({self._date_min} – {self._date_max}):\n\n"
            + "\n".join(
                f"• **{row['Category']}**: {int(row['Total_Demand']):,} units "
                f"({int(row['Products'])} products)"
                for _, row in agg.iterrows()
            )
        )
        return self._wrap(text, table=agg.rename(columns={
            "Total_Demand": "Total Demand", "Avg_Daily": "Avg Daily Demand",
            "Products": "# Products",
        }), chart_hint="category_bar")

    def _handle_revenue(self, query: str) -> dict:
        if "Price" not in self.df.columns:
            return self._wrap("Price data is not available in the dataset.")

        self.df["Revenue"] = self.df["Demand"] * self.df["Price"]
        agg = (
            self.df.groupby(["Product ID", "Product Name", "Category"], as_index=False)
            .agg(Total_Revenue=("Revenue", "sum"))
            .sort_values("Total_Revenue", ascending=False)
        )
        top = agg.head(10)
        total_rev = self.df["Revenue"].sum()
        text = (
            f"**Total estimated revenue: R{total_rev:,.2f}** "
            f"({self._date_min} – {self._date_max})\n\n"
            f"**Top 10 products by estimated revenue (R):**\n"
            + "\n".join(
                f"{i+1}. {row['Product Name']} — R{row['Total_Revenue']:,.2f}"
                for i, row in top.iterrows()
            )
            + "\n\n_Note: Revenue estimates use recorded Price × Demand. "
            "Actual sales may differ due to promotions and discounts._"
        )
        return self._wrap(text, table=top.rename(columns={"Total_Revenue": "Est. Revenue (R)"}),
                          chart_hint="revenue_bar")

    def _handle_compare(self, query: str) -> dict:
        cats_mentioned = [c for c in self._categories if c.lower() in query.lower()]
        if len(cats_mentioned) >= 2:
            subset = self.df[self.df["Category"].isin(cats_mentioned)]
            agg = subset.groupby("Category", as_index=False).agg(
                Total_Demand=("Demand", "sum"),
                Avg_Daily=("Demand", "mean"),
                Products=("Product ID", "nunique"),
            )
            text = "**Category comparison:**\n\n"
            for _, row in agg.iterrows():
                text += (
                    f"**{row['Category']}**: {int(row['Total_Demand']):,} total units | "
                    f"{row['Avg_Daily']:.1f} avg/day | {int(row['Products'])} products\n"
                )
            return self._wrap(text, table=agg, chart_hint="compare_bar")

        # Fallback to general category breakdown
        return self._handle_category(query)

    def _handle_trend(self, query: str) -> dict:
        monthly = (
            self.df.assign(Month=self.df["Date"].dt.to_period("M"))
            .groupby("Month", as_index=False)["Demand"]
            .sum()
        )
        monthly["Month"] = monthly["Month"].astype(str)
        first = int(monthly["Demand"].iloc[0])
        last  = int(monthly["Demand"].iloc[-1])
        direction = "📈 increasing" if last > first else "📉 decreasing"

        text = (
            f"**Sales trend analysis** ({self._date_min} – {self._date_max}):\n\n"
            f"Overall demand appears to be {direction} over the recorded period.\n"
            f"• First month: **{first:,} units**\n"
            f"• Latest month: **{last:,} units**\n\n"
            f"_See the Sales Analytics page for the full interactive trend chart._"
        )
        return self._wrap(text, table=monthly.rename(columns={"Demand": "Total Demand"}),
                          chart_hint="trend_line")

    def _handle_weather(self, query: str) -> dict:
        if "Weather Condition" not in self.df.columns:
            return self._wrap("Weather data is not available in the dataset.")

        weather_agg = (
            self.df.groupby("Weather Condition", as_index=False)["Demand"]
            .agg(["mean", "sum"])
            .rename(columns={"mean": "Avg Daily Demand", "sum": "Total Demand"})
            .sort_values("Avg Daily Demand", ascending=False)
        )
        best = weather_agg.iloc[0]
        text = (
            f"**Weather impact on demand:**\n\n"
            f"Demand is highest on **{best['Weather Condition']}** days "
            f"({best['Avg Daily Demand']:.1f} units/day on average).\n\n"
        )
        for _, row in weather_agg.iterrows():
            text += f"• {row['Weather Condition']}: {row['Avg Daily Demand']:.1f} avg units/day\n"
        return self._wrap(text, table=weather_agg, chart_hint="weather_bar")

    def _handle_promotion(self, query: str) -> dict:
        if "Promotion" not in self.df.columns:
            return self._wrap("Promotion data is not available in the dataset.")

        promo_agg = (
            self.df.groupby("Promotion", as_index=False)["Demand"]
            .mean()
            .rename(columns={"Demand": "Avg Demand"})
        )
        promo_agg["Promotion"] = promo_agg["Promotion"].map({0: "No Promotion", 1: "On Promotion"})

        no_promo = promo_agg[promo_agg["Promotion"] == "No Promotion"]["Avg Demand"].values
        on_promo = promo_agg[promo_agg["Promotion"] == "On Promotion"]["Avg Demand"].values
        if len(no_promo) > 0 and len(on_promo) > 0:
            lift = ((on_promo[0] - no_promo[0]) / no_promo[0] * 100)
            text = (
                f"**Promotion impact analysis:**\n\n"
                f"• Avg demand WITHOUT promotion: **{no_promo[0]:.1f} units/day**\n"
                f"• Avg demand WITH promotion: **{on_promo[0]:.1f} units/day**\n"
                f"• Demand lift from promotions: **{lift:+.1f}%**"
            )
        else:
            text = "Could not calculate promotion impact from the available data."
        return self._wrap(text, table=promo_agg, chart_hint="promo_bar")

    def _handle_unknown(self, query: str) -> dict:
        text = (
            f"I'm not sure how to answer *\"{query}\"* specifically.\n\n"
            "Here are some things I can help with:\n"
            "- *Which products should I reorder?*\n"
            "- *Show top selling products*\n"
            "- *Which items are at high stockout risk?*\n"
            "- *What is the demand forecast?*\n"
            "- *How is the Beverages category performing?*\n"
            "- *Compare Snacks and Beverages*\n\n"
            "Type **help** for a full list of questions."
        )
        return self._wrap(text)

    # ── Utility ────────────────────────────────────────────────────────────
    @staticmethod
    def _wrap(
        text: str,
        table: Optional[pd.DataFrame] = None,
        chart_hint: Optional[str] = None,
    ) -> dict:
        return {"text": text, "table": table, "chart_hint": chart_hint}


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_cleaning import clean
    from src.feature_engineering import build_features
    from src.forecasting import load_forecast

    df_clean, _ = clean(verbose=False)
    df_feat      = build_features(df_clean, verbose=False)
    fc           = load_forecast()

    assistant = ManagerAssistant(df_feat, fc)
    for q in [
        "Hello",
        "Which products should I reorder?",
        "What are the top selling products?",
        "How is the Beverages category performing?",
        "What is the expected demand for Bottled Water?",
        "Which products are slow moving?",
        "How does weather affect demand?",
    ]:
        print(f"\nQ: {q}")
        r = assistant.ask(q)
        print(f"A: {r['text'][:200]}")
