import pandas as pd

ORIGINAL  = "data/General/^VIX_2015_2025.csv"
TO_SAVE = "data/General/^VIX5_2011_2025.csv"
TO_FORMAT = "./$VIX.X_5Minute_2011_to_2025.csv"


def formatting(df, ticker="^VIX"):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["datetime"]).dt.date.astype(str)

    df = df[["Date", "close", "high", "low", "open", "volume"]]

    df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

    header1 = ["Ticker"] + [ticker] * 5

    header2 = ["Date", None, None, None, None, None]

    header_df = pd.DataFrame([header1, header2], columns=df.columns)

    df_final = pd.concat([header_df, df], ignore_index=True)

    return df_final

def main():
    df_original = pd.read_csv(ORIGINAL)
    df_format   = pd.read_csv(TO_FORMAT)

    print("\nOriginal  :\n", df_original.head())
    print("\nTo Format :\n", df_format.head())

    df_format = formatting(df=df_format)

    print("\nFormated  :\n", df_format.head())
    df_format.to_csv(TO_SAVE, index=False)

if __name__ == "__main__":
    main()