import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("Buyer Performance by CP with Rankings")

    uploaded_file = st.file_uploader("Upload your Excel", type=["xlsx"])
    if uploaded_file is not None:
        # Read Excel; row 5 has column headers (so header=4)
        df = pd.read_excel(uploaded_file, header=4)

        # Rename columns to easier names:
        df.rename(columns={
            df.columns[0]: "Harvest_ID",       # Column A
            df.columns[1]: "Buyer",            # Column B
            df.columns[3]: "Collection_Point", # Column D
            df.columns[4]: "Fresh_Purchased",  # Column E
            df.columns[7]: "Juice_Loss_Kasese",# Column H
            df.columns[15]: "Dry_Output"       # Column P
        }, inplace=True)

        # Sort descending so that head(3) gives us the most recent valid harvests
        df.sort_index(ascending=False, inplace=True)

        # Calculate global yield & global juice loss for each buyer across all CPs
        # Only use a harvest if both Fresh_Purchased and Dry_Output are valid
        global_stats = []
        grouped = df.groupby("Buyer")
        for buyer, buyer_df in grouped:
            # Filter valid rows: ignore rows where either value is missing
            valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
            valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
            valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
            
            # Take the 3 most recent valid harvests
            last_3 = valid.head(3)
            total_fresh = last_3["Fresh_Purchased"].sum()
            total_dry = last_3["Dry_Output"].sum()
            
            # Calculate yield as (total_dry / total_fresh) * 100, if total_fresh > 0
            if total_fresh > 0:
                yield_percentage = (total_dry / total_fresh) * 100
            else:
                yield_percentage = np.nan
            
            # Get the most recent non-null Juice Loss value for this buyer
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

        # Convert global stats to DataFrame
        global_stats_df = pd.DataFrame(global_stats)

        # Merge global stats back onto the main DataFrame (by Buyer)
        merged_df = pd.merge(df, global_stats_df, on="Buyer", how="left")

        # For display, create formatted strings for yield and juice loss
        merged_df["Global_Yield_Display"] = merged_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        merged_df["Global_Juice_Loss_Display"] = merged_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # Create a table showing unique CP and Buyer with their global stats.
        cp_buyer_df = merged_df[[
            "Collection_Point", "Buyer", "Global_Yield", "Global_Juice_Loss",
            "Global_Yield_Display", "Global_Juice_Loss_Display"
        ]].drop_duplicates()

        # --- Ranking within each CP ---
        # For each CP, rank the buyers based on:
        #   1) Global_Yield (descending; higher yield is better)
        #   2) Global_Juice_Loss (ascending; lower loss is better)
        ranking_rows = []
        for cp, sub_df in cp_buyer_df.groupby("Collection_Point"):
            sub_df = sub_df.copy()
            # Replace NaN yields with 0 for ranking purposes
            sub_df["Global_Yield_Rank"] = sub_df["Global_Yield"].fillna(0)
            # Replace NaN juice loss with a high number so they sort lower
            sub_df["Global_Juice_Loss_Rank"] = sub_df["Global_Juice_Loss"].fillna(9999)
            # Sort: yield descending, juice loss ascending
            sub_df.sort_values(
                by=["Global_Yield_Rank", "Global_Juice_Loss_Rank"],
                ascending=[False, True],
                inplace=True
            )

            # Extract best, second best, and third best if available
            best = sub_df.iloc[0] if len(sub_df) > 0 else None
            second = sub_df.iloc[1] if len(sub_df) > 1 else None
            third = sub_df.iloc[2] if len(sub_df) > 2 else None

            ranking_rows.append({
                "Collection_Point": cp,
                "Best_Buyer": best["Buyer"] if best is not None else "",
                "Best_Yield": best["Global_Yield_Display"] if best is not None else "",
                "Best_Juice_Loss": best["Global_Juice_Loss_Display"] if best is not None else "",
                "Second_Best_Buyer": second["Buyer"] if second is not None else "",
                "Second_Best_Yield": second["Global_Yield_Display"] if second is not None else "",
                "Second_Best_Juice_Loss": second["Global_Juice_Loss_Display"] if second is not None else "",
                "Third_Best_Buyer": third["Buyer"] if third is not None else "",
                "Third_Best_Yield": third["Global_Yield_Display"] if third is not None else "",
                "Third_Best_Juice_Loss": third["Global_Juice_Loss_Display"] if third is not None else "",
            })

        ranking_df = pd.DataFrame(ranking_rows)

        # Sort the ranking table by Collection_Point
        ranking_df.sort_values(by="Collection_Point", inplace=True)

        st.subheader("Buyer Performance by CP (with Global Yield & Juice Loss)")
        st.dataframe(cp_buyer_df.sort_values(by="Collection_Point"))

        st.subheader("Best, Second Best, and Third Best Buyer per CP")
        st.dataframe(ranking_df)

if __name__ == "__main__":
    main()
