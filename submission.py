import pandas as pd
from collections import Counter

df1 = pd.read_csv("submission1.csv")
df2 = pd.read_csv("submission2.csv")
df3 = pd.read_csv("submission3.csv")

def mode_of_list(lst):
    count = Counter(lst)
    mode = max(count, key=count.get)
    return mode


df = pd.merge(df1, df2, on='Index')
df = pd.merge(df, df3, on='Index')

df['target'] = df[['target_x', 'target_y', 'target']].values.tolist()
df['target'] = df['target'].apply(mode_of_list)

df = df[['Index', 'target']]
df.to_csv("submission.csv", index=False)