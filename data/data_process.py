import pandas as pd

# Read the CSV file using ";" as the delimiter
df = pd.read_csv('/Users/wangyunjeff/PycharmProjects/llm_mulmodel_WD_prediction/data/la-haute-borne-data-2013-2016.csv', delimiter=';')

# Select only the 'P_avg', 'ws_avg', and 'Gost_avg' columns
df_filtered = df[['P_avg', 'Ws_avg', 'Gost_avg']]

# Write the filtered DataFrame to a new CSV file using "," as the delimiter
df_filtered.to_csv('la-haute-borne-data-2013-2016_new-3columns.csv', index=False)
