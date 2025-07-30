import pandas as pd

def normalization(dataframe, min_values, max_values): # min-max scaling
    for col in ["max_speed", "weight", "acceleration", "num_seats"]:
        min_value = min_values[col]
        max_value = max_values[col]
        dataframe[col] = (dataframe[col] - min_value) / (max_value - min_value)

def save_min_max_values(dataframe, min_values, max_values):
    for col in ["max_speed", "weight", "acceleration", "num_seats"]:
        min_value = dataframe[col].min()
        max_value = dataframe[col].max()
        min_values[col] = min_value
        max_values[col] = max_value

def get_must_results(dataframe):
    m_r = []
    for i in range(len(dataframe)):
        m_r.append(int(dataframe.loc[i, "is_sport"]))
    return m_r

def load_data(file):
    return pd.read_csv(file)