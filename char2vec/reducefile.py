import pandas as pd

df = pd.read_csv('../dataset/benigns.csv',index_col=0)
df = df[:10000]
df.to_csv('../dataset/benigns_10000.csv')
df = pd.read_csv('../dataset/phishings.csv',index_col=0)
df = df[:10000]
df.to_csv('../dataset/phishings_10000.csv')