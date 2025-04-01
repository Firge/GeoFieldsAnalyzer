try:
    import sys
    import os
    import numpy as np
    from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
                                 QCheckBox, QGridLayout, QTextEdit, QRadioButton, QButtonGroup,
                                 QMessageBox, QProgressBar, QDialog, QListWidget)
    from PyQt5.QtCore import Qt
    import openpyxl
    from collections import defaultdict
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import datetime
    from lmfit import Model
    from scipy.interpolate import interp1d, CubicSpline
    from tsfresh import extract_features, select_features

    class GeoDataPreparing(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()

        def initUI(self):
            self.setWindowTitle('GeoDataPreparing')
            self.setGeometry(100, 100, 1200, 700)

            self.folder_paths = {}
            self.fields_mapping_paths = {}
            self.culture_checkboxes = {}
            self.cultures_data = {}
            self.channel_radiobuttons = {}
            self.selected_channel = None
            self.channel_button_group = QButtonGroup(self)

            self.left_tabs = QTabWidget()
            self.tab_sentinel = self.create_data_source_tab('Sentinel')
            self.tab_landsat = self.create_data_source_tab('Landsat')
            self.tab_retinor = self.create_data_source_tab('Retinor')
            self.tab_drone = self.create_data_source_tab('Drone')
            self.tab_custom = self.create_data_source_tab('Custom')

            self.left_tabs.addTab(self.tab_sentinel, 'Sentinel')
            self.left_tabs.addTab(self.tab_landsat, 'Landsat')
            self.left_tabs.addTab(self.tab_retinor, 'Retinor')
            self.left_tabs.addTab(self.tab_drone, 'Drone')
            self.left_tabs.addTab(self.tab_custom, 'Custom')

            self.fields_tab = QTableWidget()
            self.fields_tab.setColumnCount(2)
            self.fields_tab.setHorizontalHeaderLabels(['Поле', 'Культура'])
            self.fields_tab.setEditTriggers(QTableWidget.NoEditTriggers)

            self.channels_tab = QTableWidget()
            self.channels_tab.setColumnCount(2)
            self.channels_tab.setHorizontalHeaderLabels(['Канал', 'Использование'])
            self.channels_tab.setEditTriggers(QTableWidget.NoEditTriggers)

            self.bottom_left_tabs = QTabWidget()
            self.bottom_left_tabs.addTab(self.fields_tab, 'Поля')
            self.bottom_left_tabs.addTab(self.channels_tab, 'Каналы')

            self.left_layout = QVBoxLayout()
            self.left_layout.addWidget(self.left_tabs)
            self.left_layout.addWidget(self.bottom_left_tabs)

            self.left_container = QWidget()
            self.left_container.setLayout(self.left_layout)
            self.left_container.setFixedWidth(400)

            self.right_layout = QVBoxLayout()

            self.stats_text = QTextEdit()
            self.stats_text.setReadOnly(True)
            self.stats_text.setMaximumHeight(200)

            self.figure = Figure(figsize=(5, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setMinimumHeight(300)

            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)

            self.right_layout.addWidget(self.stats_text)
            self.right_layout.addWidget(self.canvas)
            self.right_layout.addWidget(self.progress_bar)

            self.right_container = QWidget()
            self.right_container.setLayout(self.right_layout)

            self.approximation_settings = self.create_approximation_settings()

            self.main_layout = QHBoxLayout()
            self.main_layout.addWidget(self.left_container)
            self.main_layout.addWidget(self.right_container)

            self.main_vertical_layout = QVBoxLayout(self)
            self.main_vertical_layout.addLayout(self.main_layout)
            self.main_vertical_layout.addWidget(self.approximation_settings)

        def create_data_source_tab(self, source_name):
            tab = QWidget()
            layout = QVBoxLayout()

            layout.addWidget(QLabel(f'Путь к папке {source_name}:'))
            folder_path_layout = QHBoxLayout()
            folder_path_edit = QLineEdit()
            folder_path_edit.setStyleSheet("text-align: left;")
            folder_path_edit.setReadOnly(True)
            folder_path_edit.setObjectName(f"folder_path_{source_name}")
            self.folder_paths[source_name] = folder_path_edit

            folder_path_button = QPushButton('Выбрать')
            folder_path_button.clicked.connect(lambda: self.select_folder(source_name))
            folder_path_layout.addWidget(folder_path_edit)
            folder_path_layout.addWidget(folder_path_button)
            layout.addLayout(folder_path_layout)

            layout.addWidget(QLabel('Путь к файлу для сопоставления названий полей:'))
            fields_mapping_path_layout = QHBoxLayout()
            fields_mapping_path_edit = QLineEdit()
            fields_mapping_path_edit.setStyleSheet("text-align: left;")
            fields_mapping_path_edit.setReadOnly(True)
            fields_mapping_path_edit.setObjectName(f"fields_mapping_path_{source_name}")
            self.fields_mapping_paths[source_name] = fields_mapping_path_edit

            fields_mapping_path_button = QPushButton('Выбрать')
            fields_mapping_path_button.clicked.connect(lambda: self.select_fields_mapping_file(source_name))
            fields_mapping_path_layout.addWidget(fields_mapping_path_edit)
            fields_mapping_path_layout.addWidget(fields_mapping_path_button)
            layout.addLayout(fields_mapping_path_layout)

            cultures_checkboxes_grid = QGridLayout()
            self.culture_checkboxes[source_name] = {}
            layout.addLayout(cultures_checkboxes_grid)

            show_stats_button = QPushButton('Показать статистику')
            show_stats_button.clicked.connect(lambda: self.show_statistics(source_name))
            layout.addWidget(show_stats_button)

            tab.setLayout(layout)
            return tab

        def select_folder(self, source_name):
            folder_path = QFileDialog.getExistingDirectory(self, 'Выбрать папку')
            if folder_path:
                self.folder_paths[source_name].setText(folder_path)
                self.scan_folder_structure(folder_path)

        def scan_folder_structure(self, folder_path):
            self.channels_tab.setRowCount(0)
            self.channel_radiobuttons = {}

            for button in self.channel_button_group.buttons():
                self.channel_button_group.removeButton(button)

            try:
                subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
                for i, channel in enumerate(sorted(subdirs)):
                    self.channels_tab.insertRow(i)

                    channel_item = QTableWidgetItem(channel)
                    channel_item.setFlags(channel_item.flags() & ~Qt.ItemIsEditable)
                    self.channels_tab.setItem(i, 0, channel_item)

                    radio_widget = QWidget()
                    radio_layout = QHBoxLayout(radio_widget)
                    radio_layout.setAlignment(Qt.AlignCenter)
                    radio_layout.setContentsMargins(0, 0, 0, 0)

                    radio_button = QRadioButton()
                    radio_button.clicked.connect(lambda checked, ch=channel: self.select_channel(ch))
                    radio_layout.addWidget(radio_button)

                    self.channel_button_group.addButton(radio_button)
                    self.channel_radiobuttons[channel] = radio_button

                    self.channels_tab.setCellWidget(i, 1, radio_widget)

                if subdirs:
                    first_channel = sorted(subdirs)[0]
                    self.channel_radiobuttons[first_channel].setChecked(True)
                    self.selected_channel = first_channel

                self.stats_text.append(f"Найдено {len(subdirs)} каналов: {', '.join(sorted(subdirs))}")

            except Exception as e:
                self.stats_text.append(f"Ошибка при сканировании структуры папки: {e}")

        def select_channel(self, channel):
            self.selected_channel = channel
            self.stats_text.append(f"Выбран канал: {channel}")

        def select_fields_mapping_file(self, source_name):
            file_path, _ = QFileDialog.getOpenFileName(self, 'Выбрать файл сопоставления полей', '', 'Excel Files (*.xlsx)')
            if file_path:
                self.fields_mapping_paths[source_name].setText(file_path)
                self.load_cultures_from_excel(file_path, source_name)

        def load_cultures_from_excel(self, file_path, source_name):
            try:
                workbook = openpyxl.load_workbook(file_path)
                sheet = workbook.active
                cultures = defaultdict(list)
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    if row[1] and row[0]:
                        cultures[row[1]].append(row[0])
                self.create_culture_checkboxes(cultures, source_name)
                self.cultures_data[source_name] = cultures
            except Exception as e:
                self.stats_text.append(f"Ошибка при чтении файла Excel: {e}")

        def create_culture_checkboxes(self, cultures, source_name):
            current_tab = None
            if source_name == 'Sentinel':
                current_tab = self.tab_sentinel
            elif source_name == 'Landsat':
                current_tab = self.tab_landsat
            elif source_name == 'Retinor':
                current_tab = self.tab_retinor
            elif source_name == 'Drone':
                current_tab = self.tab_drone
            elif source_name == 'Custom':
                current_tab = self.tab_custom

            if not current_tab:
                return

            for layout_idx in range(current_tab.layout().count()):
                item = current_tab.layout().itemAt(layout_idx)
                if isinstance(item, QGridLayout):
                    for i in range(item.count()):
                        widget = item.itemAt(i).widget()
                        if widget:
                            widget.deleteLater()
                    current_tab.layout().removeItem(item)

            self.culture_checkboxes[source_name] = {}
            cultures_grid = QGridLayout()
            row = 0
            col = 0
            for culture in cultures:
                checkbox = QCheckBox(culture)
                cultures_grid.addWidget(checkbox, row, col)
                self.culture_checkboxes[source_name][culture] = checkbox
                col += 1
                if col > 1:
                    col = 0
                    row += 1

            for i in range(current_tab.layout().count()):
                item = current_tab.layout().itemAt(i)
                if item.widget() and isinstance(item.widget(), QPushButton):
                    current_tab.layout().insertLayout(i, cultures_grid)
                    break
            else:
                current_tab.layout().addLayout(cultures_grid)

        def show_statistics(self, source_name):
            try:
                if source_name in self.cultures_data and self.cultures_data[source_name]:
                    selected_cultures = [culture for culture, checkbox in self.culture_checkboxes[source_name].items()
                                         if checkbox.isChecked()]

                    total_fields = sum(len(self.cultures_data[source_name].get(culture, [])) for culture in selected_cultures)
                    unique_cultures = len(selected_cultures)

                    stats_text = f"Выбрано культур: {unique_cultures}\n"
                    stats_text += f"Общее количество полей: {total_fields}\n"
                    for culture in selected_cultures:
                        stats_text += f"- {culture}: {len(self.cultures_data[source_name].get(culture, []))} полей\n"

                    self.stats_text.setText(stats_text)
                    self.fill_fields_table(selected_cultures, source_name)
                else:
                    self.stats_text.setText("Нет данных о культурах для этой вкладки.")

                if self.selected_channel:
                    self.plot_channel_data(source_name)
            except Exception as ex:
                print(ex)

        def fill_fields_table(self, selected_cultures, source_name):
            self.fields_tab.setRowCount(0)
            for culture in selected_cultures:
                for field in self.cultures_data[source_name].get(culture, []):
                    row_position = self.fields_tab.rowCount()
                    self.fields_tab.insertRow(row_position)

                    field_item = QTableWidgetItem(field)
                    culture_item = QTableWidgetItem(culture)

                    field_item.setFlags(field_item.flags() & ~Qt.ItemIsEditable)
                    culture_item.setFlags(culture_item.flags() & ~Qt.ItemIsEditable)

                    self.fields_tab.setItem(row_position, 0, field_item)
                    self.fields_tab.setItem(row_position, 1, culture_item)

        def plot_channel_data(self, source_name):
            try:
                self.figure.clear()

                folder_path = self.folder_paths[source_name].text()
                channel_path = os.path.join(folder_path, self.selected_channel)
                scl_path = os.path.join(folder_path, "SCL")

                if not os.path.exists(channel_path) or not os.path.isdir(channel_path):
                    self.stats_text.append(f"Канал {self.selected_channel} не найден")
                    return
                if not os.path.exists(scl_path) or not os.path.isdir(scl_path):
                    self.stats_text.append(f"Папка SCL не найдена")
                    return

                selected_cultures = [culture for culture, checkbox in self.culture_checkboxes[source_name].items()
                                     if checkbox.isChecked()]
                if not selected_cultures:
                    self.stats_text.append("Не выбраны культуры для отображения")
                    return

                xx = ['121', '128', '135', '142', '149', '156', '163', '170', '177', '184', '191', '198',
                      '205', '212', '219', '226', '233', '240', '247', '254', '261', '268', '275', '282', '289', '296']

                combined_df = pd.DataFrame()
                total_fields = sum(len(self.cultures_data[source_name][culture]) for culture in selected_cultures)
                processed_fields = 0

                for culture in selected_cultures:
                    for field in self.cultures_data[source_name][culture]:
                        field_file = os.path.join(channel_path, f"{field}.csv")
                        scl_file = os.path.join(scl_path, f"{field}.csv")

                        if os.path.exists(field_file) and os.path.exists(scl_file):
                            try:
                                df = pd.read_csv(field_file, sep=";", encoding="windows-1251")
                                df_scl = pd.read_csv(scl_file, sep=";", encoding="windows-1251")
                            except UnicodeDecodeError:
                                df = pd.read_csv(field_file, sep=";", encoding="utf-8")
                                df_scl = pd.read_csv(scl_file, sep=";", encoding="utf-8")

                            df = df.drop(columns=['x', 'y'], errors='ignore')
                            df_scl = df_scl.drop(columns=['x', 'y'], errors='ignore')

                            df = df.dropna(axis=0, thresh=len(df.columns) * 0.5)
                            df_scl = df_scl.dropna(axis=0, thresh=len(df_scl.columns) * 0.5)

                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                if col in df_scl.columns:
                                    df_scl[col] = pd.to_numeric(df_scl[col], errors='coerce')

                            df = df.where((df_scl <= 5) & (df_scl >= 4))
                            df = df.drop(columns=[col for col in df.columns if df[col].isna().sum() > len(df[col]) * 0.9])

                            if df.empty or df.dropna(how='all').empty:
                                continue

                            df = df.fillna(df.mean())
                            df['culture'] = culture

                            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

                            processed_fields += 1
                            progress = int((processed_fields / total_fields) * 100)
                            self.progress_bar.setValue(progress)

                if combined_df.empty:
                    self.stats_text.append("Нет данных для выбранных культур после фильтрации")
                    self.progress_bar.setValue(0)
                    return

                combined_df = combined_df.copy()
                grouped_df = combined_df.groupby('culture').mean()

                grouped_df = grouped_df.dropna(how='all')
                if grouped_df.empty:
                    self.stats_text.append("Все данные после усреднения содержат только NaN")
                    self.progress_bar.setValue(0)
                    return

                dates = sorted(set(grouped_df.columns))
                days = [datetime.datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday for date in dates]

                selected_method = self.approximation_group.checkedButton()
                if selected_method:
                    approx_method = selected_method.text()
                else:
                    approx_method = "Аппроксимирующая сплайн функция"
                    self.stats_text.append("Метод аппроксимации не выбран, используется сплайн")

                approximated_data = {}
                for culture in grouped_df.index:
                    y = grouped_df.loc[culture].values
                    mask = np.isfinite(y)
                    if np.sum(mask) < 2:
                        self.stats_text.append(f"Пропущена культура {culture}: недостаточно данных для интерполяции")
                        continue
                    y_interp = np.interp(np.arange(len(y)), np.arange(len(y))[mask], y[mask])

                    if approx_method == "Аппроксимирующая сплайн функция":
                        interp_func = CubicSpline(days, y_interp)
                        approximated_data[culture] = interp_func([int(day) for day in xx])
                    elif approx_method == "Аппроксимирующая полиномиальной функцией 2 степени":
                        coeffs = np.polyfit(days, y_interp, 2)
                        poly_func = np.poly1d(coeffs)
                        approximated_data[culture] = poly_func([int(day) for day in xx])
                    elif approx_method == "Аппроксимирующая функция Гаусса - двойная":
                        gmodel = Model(self.gaussian_double)
                        result = gmodel.fit(y_interp, x=days, a1=0.7, b1=240, c1=55, a2=0.5, b2=150, c2=20)
                        params = result.params
                        approximated_data[culture] = [self.gaussian_double(int(day), params['a1'].value, params['b1'].value,
                                                                          params['c1'].value, params['a2'].value,
                                                                          params['b2'].value, params['c2'].value) for day in xx]
                    elif approx_method == "Аппроксимирующая функция Сумма синусов":
                        gmodel = Model(self.sinus_double)
                        result = gmodel.fit(y_interp, x=days, a1=0.5, b1=0.01, c1=-1, a2=0.15, b2=0.05, c2=-6)
                        params = result.params
                        approximated_data[culture] = [self.sinus_double(int(day), params['a1'].value, params['b1'].value,
                                                                       params['c1'].value, params['a2'].value,
                                                                       params['b2'].value, params['c2'].value) for day in xx]
                    elif approx_method == "Аппроксимирующая функция Фурье":
                        gmodel = Model(self.fourier_double)
                        result = gmodel.fit(y_interp, x=days, a0=0.5, a1=0.05, b1=0.2, a2=-0.1, b2=0.003, w=0.03)
                        params = result.params
                        approximated_data[culture] = [self.fourier_double(int(day), params['a0'].value, params['a1'].value,
                                                                         params['b1'].value, params['a2'].value,
                                                                         params['b2'].value, params['w'].value) for day in xx]
                    elif approx_method == "Аппроксимация двойной логистической функцией":
                        gmodel = Model(self.double_logistic)
                        result = gmodel.fit(y_interp, x=days, c1=0.2, c2=0.7, a1=140, a2=14, a3=294, a4=14)
                        params = result.params
                        approximated_data[culture] = [self.double_logistic(int(day), params['c1'].value, params['c2'].value,
                                                                          params['a1'].value, params['a2'].value,
                                                                          params['a3'].value, params['a4'].value) for day in xx]
                    elif approx_method == "Аппроксимирующая функция Гаусса":
                        gmodel = Model(self.gauss)
                        result = gmodel.fit(y_interp, x=days, amp=5, cen=days[np.argmax(y_interp)], wid=len(days))
                        params = result.params
                        approximated_data[culture] = [self.gauss(int(day), params['amp'].value, params['cen'].value,
                                                                params['wid'].value) for day in xx]

                if not approximated_data:
                    self.stats_text.append("Нет данных для аппроксимации после интерполяции")
                    self.progress_bar.setValue(0)
                    return

                dpi = 100
                canvas_width = self.canvas.width() / dpi
                canvas_height = self.canvas.height() / dpi
                fig = Figure(figsize=(canvas_width, canvas_height), dpi=dpi)
                ax = fig.add_subplot(111)

                ax.set_xlabel('День от начала года', fontsize=10, fontweight="bold")
                ax.set_ylabel(self.selected_channel, fontsize=16, fontweight="bold")
                ax.patch.set_edgecolor('black')
                ax.patch.set_linewidth(1)
                if self.selected_channel == 'NDVI':
                    ax.set_ylim([0.1, 0.9])

                colors = plt.cm.coolwarm(np.linspace(0, 1, len(approximated_data)))
                for i, culture in enumerate(approximated_data.keys()):
                    ax.plot(xx, approximated_data[culture], label=culture, linestyle='-', marker='s',
                            markersize=6, linewidth=2, color=colors[i])

                plt.yticks(fontsize=10, fontweight="bold")
                ax.legend(fontsize=10, loc='upper left')
                fig.tight_layout()

                self.canvas.figure = fig
                self.canvas.draw()
                self.stats_text.append(f"График {self.selected_channel} построен с аппроксимацией: {approx_method}")
                self.progress_bar.setValue(0)

            except Exception as e:
                self.stats_text.append(f"Ошибка при построении графика: {str(e)}")
                import traceback
                self.stats_text.append(f"Трассировка: {traceback.format_exc()}")
                self.progress_bar.setValue(0)

        def create_approximation_settings(self):
            group_box = QWidget()
            layout = QVBoxLayout()

            self.approximation_group = QButtonGroup(self)
            self.approximation_radiobuttons = {
                "Гаусс - двойная": QRadioButton("Аппроксимирующая функция Гаусса - двойная"),
                "Сумма синусов": QRadioButton("Аппроксимирующая функция Сумма синусов"),
                "Фурье": QRadioButton("Аппроксимирующая функция Фурье"),
                "Логистическая": QRadioButton("Аппроксимация двойной логистической функцией"),
                "Гаусс": QRadioButton("Аппроксимирующая функция Гаусса"),
                "Полином 2 степени": QRadioButton("Аппроксимирующая полиномиальной функцией 2 степени"),
                "Сплайн": QRadioButton("Аппроксимирующая сплайн функция")
            }

            for i, (method, radiobutton) in enumerate(self.approximation_radiobuttons.items()):
                self.approximation_group.addButton(radiobutton, i)
                layout.addWidget(radiobutton)

            self.approximation_radiobuttons["Сплайн"].setChecked(True)

            self.process_button = QPushButton("Начать обработку")
            self.process_button.clicked.connect(self.start_processing)
            layout.addWidget(self.process_button)

            group_box.setLayout(layout)
            return group_box

        def start_processing(self):
            try:
                selected_source = self.left_tabs.tabText(self.left_tabs.currentIndex())
                folder_path = self.folder_paths[selected_source].text()
                save_folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")

                # Проверка прав на запись
                if not os.access(save_folder, os.W_OK):
                    raise ValueError(f"Нет прав на запись в папку {save_folder}")

                scl_path = os.path.join(folder_path, "SCL")

                # Проверка входных данных
                if not folder_path or not save_folder:
                    raise ValueError("Необходимо выбрать все папки")
                if not os.path.exists(scl_path) or not os.path.isdir(scl_path):
                    raise ValueError(f"Папка SCL не найдена по пути: {scl_path}")

                selected_cultures = [culture for culture, checkbox in self.culture_checkboxes[selected_source].items()
                                     if checkbox.isChecked()]
                if not selected_cultures:
                    raise ValueError("Не выбраны культуры для обработки")

                channel_path = os.path.join(folder_path, self.selected_channel)
                if not os.path.exists(channel_path) or not os.path.isdir(channel_path):
                    raise ValueError(f"Канал {self.selected_channel} не найден")

                # Инициализация DataFrame для объединённых данных
                combined_df = pd.DataFrame()
                total_fields = sum(len(self.cultures_data[selected_source][culture]) for culture in selected_cultures)
                processed_fields = 0

                # Обработка данных по культурам и полям
                for culture in selected_cultures:
                    for field in self.cultures_data[selected_source][culture]:
                        field_file = os.path.join(channel_path, f"{field}.csv")
                        scl_file = os.path.join(scl_path, f"{field}.csv")

                        if os.path.exists(field_file) and os.path.exists(scl_file):
                            # Чтение файлов
                            try:
                                df = pd.read_csv(field_file, sep=";", encoding="windows-1251")
                                df_scl = pd.read_csv(scl_file, sep=";", encoding="windows-1251")
                            except UnicodeDecodeError:
                                df = pd.read_csv(field_file, sep=";", encoding="utf-8")
                                df_scl = pd.read_csv(scl_file, sep=";", encoding="utf-8")

                            # Удаление ненужных столбцов
                            df = df.drop(columns=['x', 'y'], errors='ignore')
                            df_scl = df_scl.drop(columns=['x', 'y'], errors='ignore')

                            # Удаление строк с большим количеством пропусков
                            df = df.dropna(axis=0, thresh=len(df.columns) * 0.5)
                            df_scl = df_scl.dropna(axis=0, thresh=len(df_scl.columns) * 0.5)

                            # Приведение данных к числовому формату
                            for col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                if col in df_scl.columns:
                                    df_scl[col] = pd.to_numeric(df_scl[col], errors='coerce')

                            # Фильтрация по условиям SCL
                            df = df.where((df_scl <= 5) & (df_scl >= 4))
                            df = df.drop(
                                columns=[col for col in df.columns if df[col].isna().sum() > len(df[col]) * 0.9])

                            # Пропуск пустых данных
                            if df.empty or df.dropna(how='all').empty:
                                continue

                            # Обработка выбросов
                            df = df.clip(lower=-1e6, upper=1e6)

                            # Важное изменение: сначала добавляем колонку с культурой,
                            # потом заполняем пропуски средними
                            df['culture'] = culture

                            # Заполнение пропущенных значений для числовых колонок
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            for col in numeric_cols:
                                if col != 'culture':
                                    df[col] = df[col].fillna(df[col].mean())

                            # Добавление данных в общий DataFrame
                            combined_df = pd.concat([combined_df, df], axis=0, ignore_index=True)

                            processed_fields += 1
                            progress = int((processed_fields / total_fields) * 100)
                            self.progress_bar.setValue(progress)

                # Проверка на пустоту
                if combined_df.empty:
                    raise ValueError("Нет данных для выбранных культур после фильтрации")

                # Финальная очистка данных - заменяем inf на NaN, но не удаляем строки
                combined_df = combined_df.replace([np.inf, -np.inf], np.nan)

                # Ограничение экстремальных значений
                combined_df = combined_df.clip(lower=-1e6, upper=1e6)

                # Заполнение оставшихся NaN значений для всего датафрейма
                for col in combined_df.columns:
                    if col != 'culture' and combined_df[col].dtype.kind in 'bifc':
                        combined_df[col] = combined_df[col].fillna(combined_df[col].mean())

                # Создаем уникальное имя файла с временной меткой
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                processed_data_path = os.path.join(save_folder, f'processed_data_{timestamp}.csv')

                # Сохранение данных до усреднения с подробной обработкой ошибок
                try:
                    combined_df.to_csv(processed_data_path, index=False, sep=';', encoding='utf-8')

                    # Проверка успешности сохранения
                    if os.path.exists(processed_data_path):
                        file_size = os.path.getsize(processed_data_path)
                        self.stats_text.append(
                            f"Обработанные данные сохранены в {processed_data_path} (размер: {file_size} байт)")
                    else:
                        self.stats_text.append(f"Предупреждение: файл {processed_data_path} не был создан")
                except Exception as save_error:
                    self.stats_text.append(f"Ошибка при сохранении данных: {str(save_error)}")
                    # Пробуем альтернативный вариант сохранения
                    try:
                        alternative_path = os.path.join(save_folder, f'processed_data_alt_{timestamp}.csv')
                        combined_df.to_csv(alternative_path, index=False, sep=',', encoding='utf-8')
                        self.stats_text.append(f"Данные сохранены альтернативным способом в {alternative_path}")
                    except Exception as alt_save_error:
                        self.stats_text.append(
                            f"Не удалось сохранить данные альтернативным способом: {str(alt_save_error)}")

                # Усреднение для дальнейшего использования
                grouped_df = combined_df.groupby('culture').mean()

                # Сохранение атрибутов для доступа из других методов
                self.processed_data = grouped_df
                self.combined_df = combined_df
                self.save_folder = save_folder
                self.selected_source = selected_source

                # Переход к следующему окну
                self.hide()
                self.processing_window = ProcessingWindow(self)
                self.processing_window.show()
                self.progress_bar.setValue(0)

            except ValueError as e:
                QMessageBox.critical(self, "Ошибка", str(e))
                self.progress_bar.setValue(0)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
                import traceback
                self.stats_text.append(f"Подробная ошибка: {traceback.format_exc()}")
                self.progress_bar.setValue(0)

        def gaussian_double(self, x, a1, b1, c1, a2, b2, c2):
            # Ограничиваем аргумент экспоненты
            arg1 = -((x - b1) / c1) ** 2
            arg2 = -((x - b2) / c2) ** 2
            exp1 = np.exp(np.clip(arg1, -500, 0))  # Ограничение до -500, чтобы избежать переполнения
            exp2 = np.exp(np.clip(arg2, -500, 0))
            return a1 * exp1 + a2 * exp2

        def sinus_double(self, x, a1, b1, c1, a2, b2, c2):
            return a1 * np.sin(b1 * x + c1) + a2 * np.sin(b2 * x + c2)

        def fourier_double(self, x, a0, a1, b1, a2, b2, w):
            return a0 + a1 * np.cos(x * w) + b1 * np.sin(x * w) + a2 * np.cos(2 * x * w) + b2 * np.sin(2 * x * w)

        def double_logistic(self, x, c1, c2, a1, a2, a3, a4):
            # Ограничиваем аргументы экспоненты
            arg1 = (a1 - x) / a2
            arg2 = (a3 - x) / a4
            exp1 = np.exp(np.clip(arg1, -500, 500))  # Ограничение в обе стороны
            exp2 = np.exp(np.clip(arg2, -500, 500))
            return c1 + c2 * (1 / (1 + exp1) - 1 / (1 + exp2))

        def gauss(self, x, amp, cen, wid):
            # Ограничиваем аргумент экспоненты
            arg = -(x - cen) ** 2 / (2 * wid ** 2)
            exp_val = np.exp(np.clip(arg, -500, 0))
            return (amp / (np.sqrt(2 * np.pi) * wid)) * exp_val

    class ProcessingWindow(QDialog):
        def __init__(self, parent):
            super().__init__(parent)
            self.parent = parent
            self.setWindowTitle("Обработка и анализ данных")
            self.setGeometry(150, 150, 800, 600)

            layout = QVBoxLayout()

            # Статистика
            self.stats_text = QTextEdit()
            self.stats_text.setReadOnly(True)
            self.stats_text.setText(self.parent.stats_text.toPlainText())
            layout.addWidget(QLabel("Статистика:"))
            layout.addWidget(self.stats_text)

            # Выбранные культуры
            self.cultures_list = QListWidget()
            selected_cultures = [culture for culture, checkbox in self.parent.culture_checkboxes[self.parent.selected_source].items()
                                 if checkbox.isChecked()]
            for culture in selected_cultures:
                self.cultures_list.addItem(culture)
            self.cultures_list.setEnabled(False)
            self.cultures_list.setFixedHeight(100)  # Увеличиваем высоту списка
            self.cultures_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # Добавляем скролл
            layout.addWidget(QLabel("Выбранные культуры:"))
            layout.addWidget(self.cultures_list)

            # Таблица полей
            self.fields_table = QTableWidget()
            self.fields_table.setColumnCount(2)
            self.fields_table.setHorizontalHeaderLabels(['Поле', 'Культура'])
            self.fill_fields_table()
            layout.addWidget(QLabel("Поля:"))
            layout.addWidget(self.fields_table)

            # Выбранный канал
            self.channel_label = QLabel(f"Выбранный канал: {self.parent.selected_channel}")
            layout.addWidget(self.channel_label)

            # Кнопки
            self.generate_features_button = QPushButton("Генерировать фичи")
            self.generate_features_button.clicked.connect(self.generate_features)
            layout.addWidget(self.generate_features_button)

            self.select_features_button = QPushButton("Отбор параметров")
            self.select_features_button.clicked.connect(self.select_features)
            layout.addWidget(self.select_features_button)

            # Прогресс-бар
            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            layout.addWidget(self.progress_bar)

            self.setLayout(layout)

        def fill_fields_table(self):
            self.fields_table.setRowCount(0)
            selected_cultures = [culture for culture, checkbox in self.parent.culture_checkboxes[self.parent.selected_source].items()
                                 if checkbox.isChecked()]
            for culture in selected_cultures:
                for field in self.parent.cultures_data[self.parent.selected_source].get(culture, []):
                    row_position = self.fields_table.rowCount()
                    self.fields_table.insertRow(row_position)
                    self.fields_table.setItem(row_position, 0, QTableWidgetItem(field))
                    self.fields_table.setItem(row_position, 1, QTableWidgetItem(culture))

        def generate_features(self):
            self.stats_text.append("Генерация фич началась...")
            self.progress_bar.setValue(10)

            df = self.parent.combined_df
            df_melted = df.melt(id_vars=['culture'], var_name='date', value_name='value')
            df_melted['time'] = df_melted['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)
            df_melted['id'] = df_melted.index

            features = extract_features(df_melted, column_id='id', column_sort='time', column_value='value')
            features_path = os.path.join(self.parent.save_folder, 'features.csv')
            features.to_csv(features_path)

            self.progress_bar.setValue(100)
            self.stats_text.append(f"Фичи сгенерированы и сохранены в {features_path}")
            self.features = features

        def select_features(self):
            self.stats_text.append("Отбор параметров начался...")
            self.progress_bar.setValue(10)

            if not hasattr(self, 'features'):
                self.stats_text.append("Ошибка: сначала сгенерируйте фичи")
                self.progress_bar.setValue(0)
                return

            y = self.parent.combined_df['culture']
            selected_features = select_features(self.features, y)

            selected_params = list(selected_features.columns)
            self.stats_text.append("Отобранные параметры:\n" + "\n".join(selected_params))
            self.progress_bar.setValue(100)

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        window = GeoDataPreparing()
        window.show()
        sys.exit(app.exec_())
except Exception as ex:
    print(ex)