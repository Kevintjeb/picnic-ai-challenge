import pandas as pd

frame = pd.read_csv("./result3.tsv", delimiter='\t')

print(frame.head())