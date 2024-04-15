import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_and_save_csv_to_numpy(csv_file, output_file):
    # Load data from CSV
    data = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    # Select the relevant columns (modify this as per your requirement)
    # If you need specific columns you can specify them like data = data[['Column1', 'Column2']]

    # Interpolate missing values
    data = data.interpolate()

    # Normalize data
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_data = scaler.fit_transform(data)

    # Save processed data to a NumPy binary file
    np.save(output_file, data)


# Usage
import time
s = time.time()
csv_file_path = r'H:\Code_F\llm_multimodel_WS_predictiondict\data\la-haute-borne-data-2017-2020_new.csv'
data = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
e = time.time()
print(e-s)

s = time.time()
output_file_path = r'H:\Code_F\llm_multimodel_WS_predictiondict\data\la-haute-borne-data-2013-2016_new.npy'
data = np.load(output_file_path, allow_pickle=True)
e = time.time()
print(e-s)


# Call the function
preprocess_and_save_csv_to_numpy(csv_file_path, output_file_path)
