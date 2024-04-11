import pandas as pd

# Read the CSV file
df = pd.read_csv('H:/Code_F/llm_multimodel_WS_predictiondict/data/la-haute-borne-data-2013-2016_new.csv', delimiter=',')

# Assuming you know the name of the wind turbine you want to filter by (e.g., 'WTG01')
selected_turbine = 'R80711'

# Filter the DataFrame for the selected wind turbine and specific columns
df_filtered = df[df['Wind_turbine_name'] == selected_turbine][['Date_time', 'P_avg', 'Ws_avg', 'Gost_avg']]

# Convert 'Date_time' to datetime and set it as the index
df_filtered['Date_time'] = pd.to_datetime(df_filtered['Date_time'])
df_filtered.set_index('Date_time', inplace=True)

# Write the filtered DataFrame to a new CSV file
df_filtered.to_csv('la-haute-borne-data-2013-2016_new-3columns.csv')
