import pandas as pd

if __name__ == "__main__":
    original_path = 'datasets/Agnews.txt'
    target_path = 'datasets/Agnews.csv'
    data = pd.read_table(original_path, sep='\t')
    data.columns = ['label', 'text']
    data.to_csv(target_path, sep=',', index=False)