import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Buyer Performance (CP Listed, But Yield/Loss Global)")

    uploaded_file = st.file_uploader("Upload your Excel", type=["xlsx"])
    if uploaded_file is not None:
        # Read Excel; row 5 has column headers => header=4
        df = pd.read_excel(uploaded_file, header=4)

        # Rename columns to consistent names
        df.rename(columns={
            df.columns[0]: "Harvest_ID",       # A
            df.columns[1]: "Buyer",            # B
            df.columns[3]: "Collection_Point", # D
            df.columns[4]: "Fresh_Purchased",  # E
            df.columns[7]: "Juice_Loss_Kasese",# H
            df.columns[15]: "Dry_Output"       # P
        }, inplace=True)

        # Sort descending so head(3) in each group is the last 3 valid harvests
        df.sort_index(ascending=False, inplace=True)

        # Calculate global yield & global juice loss for each Buyer
        global_stats = []
        grouped = df.groupby("Buyer")

        for buyer, buyer_df in grouped:
            # 1) Filter valid rows for yield calculation: non-null numeric Fresh & Dry
            valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
            valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
            valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]

            # Take the last 3 valid (already sorted descending by index)
            last_3 = valid.head(3)
            total_fresh = last_3["Fresh_Purchased"].sum()
            total_dry = last_3["Dry_Output"].sum()

            # 2) Global yield calculation = (total_fresh / total_dry) * 100
            if total_dry != 0:
                yield_percentage = (total_fresh / total_dry) * 100
            else:
                yield_percentage = np.nan

            # 3) Most recent non-null juice loss for this Buyer
            latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
            if not latest_juice_loss_row.empty:
                juice_loss_val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
            else:
                juice_loss_val = np.nan

            global_stats.append({
                "Buyer": buyer,
                "Global_Yield": yield_percentage,
                "Global_Juice_Loss": juice_loss_val
            })

        # Convert list of dicts to DataFrame
        global_stats_df = pd.DataFrame(global_stats)

        # Merge these global stats back onto the original DataFrame, matching on Buyer
        merged_df = pd.merge(
            df, 
            global_stats_df, 
            on="Buyer", 
            how="left"
        )

        # Convert the numeric yield and juice loss to a percentage string for display
        merged_df["Global_Yield_Display"] = merged_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        merged_df["Global_Juice_Loss_Display"] = merged_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # -----------
        # 1) Display a table of CP, Buyer, Global Yield, Global Juice Loss
        # -----------
        display_df = merged_df[[
            "Collection_Point", 
            "Buyer", 
            "Global_Yield_Display", 
            "Global_Juice_Loss_Display"
        ]].drop_duplicates()

        st.subheader("Buyer Performance by CP (Global Yield & Juice Loss)")
        st.dataframe(display_df)

        # -----------
        # 2) Rank Buyers within each CP:
        #    Sort by Global_Yield (descending), then by Global_Juice_Loss (ascending)
        #    Then pick Best, Second, Third
        # -----------

        # We'll parse out numeric values from the string columns we created,
        # or we could just use the numeric columns "Global_Yield" and "Global_Juice_Loss"
        # from merged_df. Let's use the numeric columns for better sorting.

        # We only need each CP-Buyer combination once
        # plus the numeric yield/loss for sorting
        rank_df = merged_df[[
            "Collection_Point", 
            "Buyer", 
            "Global_Yield", 
            "Global_Juice_Loss"
        ]].drop_duplicates()

        # Group by CP and rank
        ranking_rows = []
        for cp, sub_df in rank_df.groupby("Collection_Point"):
            sub_df = sub_df.copy()

            # Fill missing yields with 0 for sorting
            sub_df["Global_Yield"] = sub_df["Global_Yield"].fillna(0)
            # Fill missing juice loss with a large number so they sort to the bottom
            sub_df["Global_Juice_Loss"] = sub_df["Global_Juice_Loss"].fillna(9999)

            # Sort: yield desc, juice loss asc
            sub_df.sort_values(
                by=["Global_Yield", "Global_Juice_Loss"], 
                ascending=[False, True],
                inplace=True
            )

            # Identify best, second, third
            best_buyer = sub_df.iloc[0]["Buyer"] if len(sub_df) > 0 else None
            second_buyer = sub_df.iloc[1]["Buyer"] if len(sub_df) > 1 else None
            third_buyer = sub_df.iloc[2]["Buyer"] if len(sub_df) > 2 else None

            ranking_rows.append({
                "Collection_Point": cp,
                "Best_Buyer": best_buyer,
                "Second_Best_Buyer": second_buyer,
                "Third_Best_Buyer": third_buyer
            })

        ranking_df = pd.DataFrame(ranking_rows)

        st.subheader("Best, Second Best, and Third Best Buyer per CP")
        st.dataframe(ranking_df)

if __name__ == "__main__":
    main()
