import pandas as pd


def load_forex_data(filepath):

    df = pd.read_csv(filepath, sep="\t")

    print("Columns:", df.columns.tolist())

    # create proper datetime column
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])

    df = df.sort_values("time")
    df = df.set_index("time")

    # rename columns to clean names
    df = df.rename(columns={
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close"
    })

    df = df[["open", "high", "low", "close"]]

    print("✅ Data Loaded")
    print(df.head())

    return df


if __name__ == "__main__":
    data = load_forex_data("../../data/raw/eurusd.csv")