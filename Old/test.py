import pandas as pd

def main():
    # Added sep=";" to handle semicolon-separated files
    df = pd.read_csv("./dataset/Z-V4-JAPAN-AMP-VS400_M250.csv", sep=";")
    print(f"Nombre de lignes dans le fichier : {len(df)}")

if __name__ == "__main__":
    main()