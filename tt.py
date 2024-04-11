import pandas as pd

# Your ISO 8601 timestamp
timestamp = '2017-02-10T10:00:00+01:00'

# Parsing the timestamp with pandas
datetime_obj = pd.to_datetime(timestamp)

print(datetime_obj)