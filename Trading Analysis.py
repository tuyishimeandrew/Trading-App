import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("LTC Buyer Performance")

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

        # Sort descending so that head(3) gives us the most recent valid harvests.
        df.sort_index(ascending=False, inplace=True)

        # -------------------------------------------
        # 1. Calculate Global Yield & Juice Loss per Buyer
        #    Only include a harvest if both Fresh_Purchased and Dry_Output are valid numbers.
        #    Global Yield = (sum of Dry from last 3 valid harvests) / (sum of Fresh from same harvests) * 100%
        # -------------------------------------------
        global_stats = []
        grouped = df.groupby("Buyer")
        for buyer, buyer_df in grouped:
            valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
            valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
            valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
            
            last_3 = valid.head(3)
            total_fresh = last_3["Fresh_Purchased"].sum()
            total_dry = last_3["Dry_Output"].sum()
            
            if total_fresh > 0:
                yield_percentage = (total_dry / total_fresh) * 100
            else:
                yield_percentage = np.nan
            
            # Most recent non-null Juice Loss for this buyer
            latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
            if not latest_juice_loss_row.empty:
                juice_loss_val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
                # If numeric, multiply by 100 and round to 2 decimals (to convert to percentage)
                if pd.notnull(juice_loss_val) and isinstance(juice_loss_val, (int, float)):
                    juice_loss_val = round(juice_loss_val * 100, 2)
            else:
                juice_loss_val = np.nan

            global_stats.append({
                "Buyer": buyer,
                "Global_Yield": yield_percentage,
                "Global_Juice_Loss": juice_loss_val
            })

        global_stats_df = pd.DataFrame(global_stats)

        # -------------------------------------------
        # 2. Merge the global stats back onto the DataFrame (by Buyer)
        # -------------------------------------------
        merged_df = pd.merge(df, global_stats_df, on="Buyer", how="left")

        # Create formatted strings for display.
        # For Global_Yield, display two decimals and a "%" sign.
        merged_df["Global_Yield_Display"] = merged_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        # For Global_Juice_Loss, display the rounded and multiplied value as a percentage.
        merged_df["Global_Juice_Loss_Display"] = merged_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # Create a CP-Buyer table with unique rows.
        cp_buyer_df = merged_df[[
            "Collection_Point", "Buyer", "Global_Yield", "Global_Juice_Loss",
            "Global_Yield_Display", "Global_Juice_Loss_Display"
        ]].drop_duplicates()

        # -------------------------------------------
        # 3. Compute the ranking per CP (best, second, third) applying conditions:
        #    - Buyer must have Global_Yield >= 36%
        #    - Buyer must have Global_Juice_Loss <= 18%
        # -------------------------------------------
        ranking_rows = []
        for cp, sub_df in cp_buyer_df.groupby("Collection_Point"):
            sub_df = sub_df.copy()
            # Apply conditions
            valid = sub_df[(sub_df["Global_Yield"] >= 36) & (sub_df["Global_Juice_Loss"] <= 18)]
            
            # If no buyer meets conditions, leave ranking blank for this CP.
            if valid.empty:
                best = second = third = ""
            else:
                # For ranking, fill missing yields with 0 and missing juice loss with a high number.
                valid["Yield_Rank"] = valid["Global_Yield"].fillna(0)
                valid["Juice_Rank"] = valid["Global_Juice_Loss"].fillna(9999)
                
                valid.sort_values(
                    by=["Yield_Rank", "Juice_Rank"],
                    ascending=[False, True],
                    inplace=True
                )
                best = valid.iloc[0]["Buyer"] if len(valid) > 0 else ""
                second = valid.iloc[1]["Buyer"] if len(valid) > 1 else ""
                third = valid.iloc[2]["Buyer"] if len(valid) > 2 else ""
            
            ranking_rows.append({
                "Collection_Point": cp,
                "Best_Buyer": best,
                "Second_Buyer": second,
                "Third_Buyer": third
            })

        ranking_df = pd.DataFrame(ranking_rows)

        # -------------------------------------------
        # 4. Merge the ranking information back into the CP-Buyer table.
        #    The ranking columns will only show a value on the row where the buyer name matches.
        # -------------------------------------------
        final_df = cp_buyer_df.merge(ranking_df, on="Collection_Point", how="left")

        final_df["Best Buyer for CP"] = final_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Best_Buyer"] else "", axis=1
        )
        final_df["Second Best Buyer for CP"] = final_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Second_Buyer"] else "", axis=1
        )
        final_df["Third Best Buyer for CP"] = final_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Third_Buyer"] else "", axis=1
        )

        # -------------------------------------------
        # 5. Rename and select final columns.
        #    - Rename Global_Yield_Display to "Yield three prior harvest(%)"
        #    - Rename Global_Juice_Loss_Display to "Juice loss at Kasese(%)"
        # -------------------------------------------
        final_df.rename(columns={
            "Global_Yield_Display": "Yield three prior harvest(%)",
            "Global_Juice_Loss_Display": "Juice loss at Kasese(%)"
        }, inplace=True)

        # Select only the required columns and sort by Collection_Point.
        final_display = final_df[[
            "Collection_Point", "Buyer", 
            "Yield three prior harvest(%)", "Juice loss at Kasese(%)", 
            "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
        ]].sort_values(by="Collection_Point")

        st.subheader("Buyer Performance by CP")
        st.dataframe(final_display)

if __name__ == "__main__":
    main()
