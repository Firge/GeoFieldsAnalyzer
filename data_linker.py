import os
import zarr
import numpy as np
import pandas as pd

print("Data linking")
print(os.listdir("data/landsat/"))

# Путь к директориям с данными
data_dir = 'data/landsat'

# Создание Zarr-хранилища
zarr_store = zarr.open('sentinel_linked_data.zarr', mode='w')

# Список каналов
channels = os.listdir(data_dir)

# Индексация каналов
channel_index_map = {channel: idx for idx, channel in enumerate(channels)}

# Итерация по каналам
for channel in channels:
    channel_path = os.path.join(data_dir, channel)

    if os.path.isdir(channel_path):

        # Итерация по файлам в каждой директории
        for file in os.listdir(channel_path):
            if file.endswith('.csv'):
                field_name = file.split('.')[0]  # Имя участка поля
                file_path = os.path.join(channel_path, file)

                # Чтение CSV с координатами и значениями
                df = pd.read_csv(file_path, delimiter=';')
                x_coords = df['x'].values  # Координаты x
                y_coords = df['y'].values  # Координаты y
                time_stamps = df.columns[2:]  # Временные метки (столбцы после x и y)

                # Объединяем координаты x и y в один массив
                coordinates = np.column_stack((x_coords, y_coords))  # массив вида [[x1, y1], [x2, y2], ...]

                # Создаем группу в Zarr для текущего участка поля, если не существует
                if field_name not in zarr_store:
                    field_group = zarr_store.create_group(field_name)

                    # Сохранение объединённых координат
                    field_group.create_dataset('coordinates',
                                               data=coordinates,
                                               dtype='f4',
                                               chunks=(len(coordinates), 2))

                    # Сохранение временных меток
                    field_group.create_dataset('time', data=np.array(time_stamps, dtype='S10'))

                    # Создаём массив данных
                    data_shape = (len(coordinates), len(time_stamps), len(channels))  # новая структура
                    field_group.create_dataset(
                        'data',
                        shape=data_shape,
                        chunks=(100, len(time_stamps), 1),  # Чанки изменены
                        dtype='f4',
                        fill_value=np.nan
                    )

                # Ссылка на группу данных
                field_group = zarr_store[field_name]
                data_array = field_group['data']

                # Получаем индекс текущего канала
                channel_idx = channel_index_map[channel]

                # Заполнение данных для каждого временного штампа
                for time_idx, time_stamp in enumerate(time_stamps):
                    if time_stamp in df.columns:
                        values = df[time_stamp].values

                        # Убедимся, что длина значений совпадает с количеством точек координат
                        if len(values) != len(coordinates):
                            raise ValueError(
                                f"Количество значений ({len(values)}) не соответствует числу координат ({len(coordinates)}).")

                        # Записываем одномерный массив значений в соответствующий временной и канальный срез
                        data_array[:, time_idx, channel_idx] = values

print("Data linking completed!")
