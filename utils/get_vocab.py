import pandas as pd

def get_vocab_data():
    df = pd.read_csv("./custom_vocab/CoinedWord.csv", sep=",", encoding="utf-8", header=None, )
    print(df.head())
    print(df.head()[0])
    print(df.head()[0].to_list())
    
if __name__ == "__main__":
    get_vocab_data()
    