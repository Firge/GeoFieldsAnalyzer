import pandas as pd
import os

data_dir = "data/landsat"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

data = None

for file in files:
    channel_name = file.split('-')[0]  # Название канала из имени файла (например, ATMOSPHERIC_TRANSMITTANCE)
    channel_df = pd.read_csv(os.path.join(data_dir, file), sep=';')

    channel_df = channel_df.rename(columns={col: f"{channel_name}_{col}" for col in channel_df.columns[2:]})

    if data is None:
        data = channel_df
    else:
        data = pd.merge(data, channel_df, on=["x", "y"], how="inner")

# Сохранение объединённых данных в CSV (опционально)
data.to_csv("merged_data.csv", index=False)
