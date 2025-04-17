import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
                             QCheckBox, QGridLayout, QTextEdit, QRadioButton, QButtonGroup,
                             QMessageBox, QProgressBar, QDialog, QListWidget, QDoubleSpinBox)
from PyQt5.QtCore import Qt
import openpyxl
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rc
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
        self.tab_Meteor = self.create_data_source_tab('Meteor')
        self.tab_drone = self.create_data_source_tab('Drone')
        self.tab_custom = self.create_data_source_tab('Custom')

        self.left_tabs.addTab(self.tab_sentinel, 'Sentinel')
        self.left_tabs.addTab(self.tab_landsat, 'Landsat')
        self.left_tabs.addTab(self.tab_Meteor, 'Meteor')
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
        elif source_name == 'Meteor':
            current_tab = self.tab_Meteor
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
            rc('xtick', labelsize=8)
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
            # Получаем пути и текущую вкладку
            selected_source = self.left_tabs.tabText(self.left_tabs.currentIndex())
            folder_path = self.folder_paths[selected_source].text()
            save_folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")

            if not os.access(save_folder, os.W_OK):
                raise ValueError(f"Нет прав на запись в папку {save_folder}")

            scl_path = os.path.join(folder_path, "SCL")
            channel_path = os.path.join(folder_path, self.selected_channel)

            # Проверка входных данных
            if not folder_path or not save_folder:
                raise ValueError("Необходимо выбрать все папки")
            if not os.path.exists(scl_path) or not os.path.isdir(scl_path):
                raise ValueError(f"Папка SCL не найдена: {scl_path}")
            if not os.path.exists(channel_path) or not os.path.isdir(channel_path):
                raise ValueError(f"Канал {self.selected_channel} не найден")

            selected_cultures = [culture for culture, checkbox in self.culture_checkboxes[selected_source].items()
                                 if checkbox.isChecked()]
            if not selected_cultures:
                raise ValueError("Не выбраны культуры для обработки")

            selected_method = self.approximation_group.checkedButton()
            if selected_method:
                approx_method = selected_method.text()
            else:
                approx_method = "Аппроксимирующая сплайн функция"
                self.stats_text.append("Метод аппроксимации не выбран, используется сплайн")

            # Список для хранения данных по пикселям
            pixel_data_list = []
            total_fields = sum(len(self.cultures_data[selected_source][culture]) for culture in selected_cultures)
            processed_fields = 0

            # Обработка данных по культурам и полям
            for culture in selected_cultures:
                for field in self.cultures_data[selected_source][culture]:
                    field_file = os.path.join(channel_path, f"{field}.csv")
                    scl_file = os.path.join(scl_path, f"{field}.csv")

                    if os.path.exists(field_file) and os.path.exists(scl_file):
                        try:
                            df = pd.read_csv(field_file, sep=";", encoding="windows-1251")
                            df_scl = pd.read_csv(scl_file, sep=";", encoding="windows-1251")
                        except UnicodeDecodeError:
                            df = pd.read_csv(field_file, sep=";", encoding="utf-8")
                            df_scl = pd.read_csv(scl_file, sep=";", encoding="utf-8")

                        # Сохраняем координаты и исходные столбцы данных
                        pixel_coords = df[['x', 'y']].copy()
                        original_columns = [col for col in df.columns if col not in ['x', 'y']]
                        df = df.drop(columns=['x', 'y'], errors='ignore')
                        df_scl = df_scl.drop(columns=['x', 'y'], errors='ignore')

                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            if col in df_scl.columns:
                                df_scl[col] = pd.to_numeric(df_scl[col], errors='coerce')

                        # Фильтрация по SCL (облачность), заменяем неподходящие значения на NaN
                        df = df.where((df_scl <= 5) & (df_scl >= 4))

                        dates = df.columns
                        days = [datetime.datetime.strptime(date, '%Y-%m-%d').timetuple().tm_yday for date in dates]

                        # Горизонтальная интерполяция для каждого пикселя
                        for idx, row in df.iterrows():
                            pixel_values = row.values
                            # Заменяем нули на NaN для интерполяции
                            pixel_values = np.where(pixel_values == 0, np.nan, pixel_values)
                            mask = np.isfinite(pixel_values)

                            if np.sum(mask) < 2:
                                continue  # Пропускаем пиксель, если меньше 2 валидных точек

                            x_data = np.array(days)[mask]
                            y_data = pixel_values[mask]

                            # Применение выбранного метода аппроксимации
                            try:
                                if approx_method == "Аппроксимирующая сплайн функция":
                                    interp_func = CubicSpline(x_data, y_data, extrapolate=True)
                                elif approx_method == "Аппроксимирующая полиномиальной функцией 2 степени":
                                    if len(x_data) < 3:
                                        raise ValueError("Для полинома 2-й степени нужно минимум 3 точки")
                                    coeffs = np.polyfit(x_data, y_data, 2)
                                    interp_func = np.poly1d(coeffs)
                                elif approx_method == "Аппроксимирующая функция Гаусса":
                                    gmodel = Model(self.gauss)
                                    result = gmodel.fit(y_data, x=x_data, amp=5, cen=x_data[np.argmax(y_data)],
                                                        wid=len(x_data))
                                    interp_func = lambda x: self.gauss(x, *result.params.values())
                                elif approx_method == "Аппроксимирующая функция Гаусса - двойная":
                                    gmodel = Model(self.gaussian_double)
                                    result = gmodel.fit(y_data, x=x_data, a1=0.7, b1=240, c1=55, a2=0.5, b2=150, c2=20)
                                    params = result.params
                                    interp_func = lambda x: self.gaussian_double(x, params['a1'].value,
                                                                                 params['b1'].value,
                                                                                 params['c1'].value, params['a2'].value,
                                                                                 params['b2'].value, params['c2'].value)
                                elif approx_method == "Аппроксимирующая функция Сумма синусов":
                                    gmodel = Model(self.sinus_double)
                                    result = gmodel.fit(y_data, x=x_data, a1=0.5, b1=0.01, c1=-1, a2=0.15, b2=0.05,
                                                        c2=-6)
                                    params = result.params
                                    interp_func = lambda x: self.sinus_double(x, params['a1'].value, params['b1'].value,
                                                                              params['c1'].value, params['a2'].value,
                                                                              params['b2'].value, params['c2'].value)
                                elif approx_method == "Аппроксимирующая функция Фурье":
                                    gmodel = Model(self.fourier_double)
                                    result = gmodel.fit(y_data, x=x_data, a0=0.5, a1=0.05, b1=0.2, a2=-0.1, b2=0.003,
                                                        w=0.03)
                                    params = result.params
                                    interp_func = lambda x: self.fourier_double(x, params['a0'].value,
                                                                                params['a1'].value,
                                                                                params['b1'].value, params['a2'].value,
                                                                                params['b2'].value, params['w'].value)
                                elif approx_method == "Аппроксимация двойной логистической функцией":
                                    gmodel = Model(self.double_logistic)
                                    result = gmodel.fit(y_data, x=x_data, c1=0.2, c2=0.7, a1=140, a2=14, a3=294, a4=14)
                                    params = result.params
                                    interp_func = lambda x: self.double_logistic(x, params['c1'].value,
                                                                                 params['c2'].value,
                                                                                 params['a1'].value, params['a2'].value,
                                                                                 params['a3'].value, params['a4'].value)
                                else:
                                    raise ValueError(f"Неизвестный метод аппроксимации: {approx_method}")

                                # Вычисление аппроксимированных значений для всех дат
                                interpolated_values = interp_func(np.array(days))

                                # Обработка NaN и нулей в начале и конце
                                for i in range(len(interpolated_values)):
                                    if not np.isfinite(interpolated_values[i]) or interpolated_values[i] == 0:
                                        if i == 0:
                                            for j in range(i + 1, len(interpolated_values)):
                                                if np.isfinite(interpolated_values[j]) and interpolated_values[j] != 0:
                                                    interpolated_values[i] = interpolated_values[j]
                                                    break
                                        elif i == len(interpolated_values) - 1:
                                            for j in range(i - 1, -1, -1):
                                                if np.isfinite(interpolated_values[j]) and interpolated_values[j] != 0:
                                                    interpolated_values[i] = interpolated_values[j]
                                                    break

                            except Exception as e:
                                self.stats_text.append(f"Ошибка аппроксимации для пикселя {idx} поля {field}: {str(e)}")
                                continue

                            # Формируем данные для пикселя
                            pixel_data = {
                                'field': field,
                                'x': pixel_coords.iloc[idx]['x'],
                                'y': pixel_coords.iloc[idx]['y'],
                            }
                            for date_idx, date in enumerate(original_columns):
                                pixel_data[date] = interpolated_values[date_idx] if date_idx < len(
                                    interpolated_values) else np.nan
                            pixel_data['culture'] = culture
                            pixel_data_list.append(pixel_data)

                        processed_fields += 1
                        self.progress_bar.setValue(int((processed_fields / total_fields) * 100))

            if not pixel_data_list:
                raise ValueError("Нет данных для выбранных культур после фильтрации")

            # Создаем DataFrame с правильным порядком столбцов
            pixel_df = pd.DataFrame(pixel_data_list)
            column_order = ['field', 'x', 'y'] + original_columns + ['culture']
            pixel_df = pixel_df[column_order]

            # Сохранение с заданным именем файла
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_data_path = os.path.join(save_folder, f'processed_data_{timestamp}.csv')
            pixel_df.to_csv(processed_data_path, index=False, sep=';', encoding='utf-8')

            if os.path.exists(processed_data_path):
                file_size = os.path.getsize(processed_data_path)
                self.stats_text.append(f"Данные сохранены в {processed_data_path} (размер: {file_size} байт)")
            else:
                self.stats_text.append(f"Ошибка: файл {processed_data_path} не был создан")

            # Передача данных в следующее окно
            self.combined_df = pixel_df
            self.save_folder = save_folder
            self.selected_source = selected_source

            self.hide()
            self.processing_window = ProcessingWindow(self)
            self.processing_window.show()
            self.progress_bar.setValue(0)

        except ValueError as e:
            QMessageBox.critical(self, "Ошибка", str(e))
            self.progress_bar.setValue(0)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {str(e)}")
            self.stats_text.append(f"Подробности: {str(e)}")
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

        # Список выбранных культур
        self.cultures_list = QListWidget()
        selected_cultures = [culture for culture, checkbox in
                             self.parent.culture_checkboxes[self.parent.selected_source].items()
                             if checkbox.isChecked()]
        for culture in selected_cultures:
            self.cultures_list.addItem(culture)
        self.cultures_list.setEnabled(False)
        self.cultures_list.setFixedHeight(100)
        self.cultures_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
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

        # Настройки отбора признаков
        settings_group = QWidget()
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(20)

        # Чекбокс для удаления коррелирующих признаков
        self.remove_correlated_checkbox = QCheckBox("Удалять коррелирующие признаки")
        self.remove_correlated_checkbox.setToolTip("Если включено, будут удаляться признаки, которые сильно коррелируют друг с другом.")
        settings_layout.addWidget(self.remove_correlated_checkbox)

        # Поле для регулирования порога корреляции
        corr_layout = QHBoxLayout()
        corr_label = QLabel("Порог корреляции:")
        corr_label.setToolTip("Порог корреляции для удаления признаков (от 0 до 1).")
        self.corr_spinbox = QDoubleSpinBox()
        self.corr_spinbox.setRange(0.0, 1.0)
        self.corr_spinbox.setSingleStep(0.05)
        self.corr_spinbox.setValue(0.9)  # Дефолтное значение
        self.corr_spinbox.setDecimals(2)
        self.corr_spinbox.setFixedWidth(80)
        corr_layout.addWidget(corr_label)
        corr_layout.addWidget(self.corr_spinbox)
        settings_layout.addLayout(corr_layout)

        # Поле для регулирования порога p-value
        pvalue_layout = QHBoxLayout()
        pvalue_label = QLabel("Порог p-value:")
        pvalue_label.setToolTip("Порог значимости для отбора признаков (меньше значение — строже отбор).")
        self.pvalue_spinbox = QDoubleSpinBox()
        self.pvalue_spinbox.setRange(0.01, 1.0)
        self.pvalue_spinbox.setSingleStep(0.01)
        self.pvalue_spinbox.setValue(0.05)  # Дефолтное значение
        self.pvalue_spinbox.setDecimals(2)
        self.pvalue_spinbox.setFixedWidth(80)
        pvalue_layout.addWidget(pvalue_label)
        pvalue_layout.addWidget(self.pvalue_spinbox)
        pvalue_layout.addStretch()
        settings_layout.addLayout(pvalue_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

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

        # Стилизация для user-friendly вида
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
            }
            QCheckBox {
                font-size: 14px;
            }
            QDoubleSpinBox {
                font-size: 14px;
                padding: 2px;
            }
            QPushButton {
                font-size: 14px;
                padding: 5px;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit, QListWidget, QTableWidget {
                font-size: 12px;
            }
        """)

    def fill_fields_table(self):
        self.fields_table.setRowCount(0)
        selected_cultures = [culture for culture, checkbox in
                             self.parent.culture_checkboxes[self.parent.selected_source].items()
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

        # Используем данные из этапа обработки
        df = self.parent.combined_df.copy()
        if df.empty:
            self.stats_text.append("Ошибка: данные из обработки отсутствуют")
            self.progress_bar.setValue(0)
            return

        # Удаляем ненужные столбцы
        columns_to_drop = ['x', 'y', 'field']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        if df.empty or len(df.columns) <= 1:  # Только culture остаётся
            self.stats_text.append("Ошибка: после удаления x, y, field данных не осталось")
            self.progress_bar.setValue(0)
            return

        # Проверяем данные на NaN и бесконечности
        data_columns = [col for col in df.columns if col != 'culture']
        df[data_columns] = df[data_columns].replace([np.inf, -np.inf], np.nan)
        nan_count = df[data_columns].isna().sum().sum()
        self.stats_text.append(f"Количество NaN в данных перед генерацией: {nan_count}")

        # Удаляем строки, где все значения NaN (кроме culture)
        df = df.dropna(subset=data_columns, how='all')
        if df.empty:
            self.stats_text.append("Ошибка: после удаления строк с NaN данные пусты")
            self.progress_bar.setValue(0)
            return

        # Сохраняем исходные индексы для синхронизации
        df['original_id'] = df.index

        y = df['culture']

        # Преобразуем в длинный формат для tsfresh
        df_melted = df.melt(id_vars=['culture', 'original_id'], var_name='date', value_name='value')
        df_melted['time'] = df_melted['date'].apply(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)

        self.stats_text.append(f"Длинный формат: {len(df_melted)} строк")

        # Проверяем df_melted на NaN
        nan_count_melted = df_melted['value'].isna().sum()
        self.stats_text.append(f"Количество NaN в df_melted['value']: {nan_count_melted}")
        df_melted = df_melted.dropna(subset=['value'])
        self.stats_text.append(f"После удаления NaN в df_melted: {len(df_melted)} строк")

        if df_melted.empty:
            self.stats_text.append("Ошибка: после удаления NaN в df_melted данные пусты")
            self.progress_bar.setValue(0)
            return

        # Генерация признаков
        try:
            from tsfresh.feature_extraction import EfficientFCParameters
            features = extract_features(df_melted,
                                        column_id='original_id',
                                        column_sort='time',
                                        column_value='value',
                                        default_fc_parameters=EfficientFCParameters())
            self.progress_bar.setValue(50)

            # Проверяем сгенерированные признаки
            nan_features_count = features.isna().sum().sum()
            self.stats_text.append(f"Количество NaN в сгенерированных признаках: {nan_features_count}")

            # Синхронизируем y с features
            common_indices = features.index.intersection(y.index)
            if len(common_indices) == 0:
                self.stats_text.append("Ошибка: нет общих индексов между признаками и y")
                self.progress_bar.setValue(0)
                return
            features = features.loc[common_indices]
            y = y.loc[common_indices]
            self.stats_text.append(f"После синхронизации с y: размер features: {features.shape[0]}")

            # Добавляем колонку cultures
            features['cultures'] = y
            self.stats_text.append("Добавлена колонка 'cultures' с информацией о культурах")

            # Сохранение признаков
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            features_path = os.path.join(self.parent.save_folder, f'features_{timestamp}.csv')
            features.to_csv(features_path, index=True, sep=';', encoding='utf-8')

            self.features = features
            self.stats_text.append(f"Фичи сгенерированы и сохранены в {features_path}")
            self.stats_text.append(f"Количество сгенерированных признаков (без учёта колонки cultures): {len(features.columns) - 1}")
            self.progress_bar.setValue(100)

        except Exception as e:
            self.stats_text.append(f"Ошибка при генерации фич: {str(e)}")
            import traceback
            self.stats_text.append(f"Трассировка: {traceback.format_exc()}")
            self.progress_bar.setValue(0)
            return

    def select_features(self):
        self.stats_text.append("Отбор параметров начался...")
        self.progress_bar.setValue(10)

        if not hasattr(self, 'features'):
            self.stats_text.append("Ошибка: сначала сгенерируйте фичи")
            self.progress_bar.setValue(0)
            return

        # Используем culture как целевую переменную
        y = self.parent.combined_df['culture']
        features = self.features

        self.stats_text.append(f"Размер признаков: {features.shape[0]}, размер y: {len(y)}")
        if len(y) != features.shape[0]:
            self.stats_text.append("Несоответствие размеров, синхронизация...")
            # Синхронизация по индексам
            common_indices = features.index.intersection(y.index)
            if len(common_indices) == 0:
                self.stats_text.append("Ошибка: нет общих индексов между признаками и y")
                self.progress_bar.setValue(0)
                return
            features = features.loc[common_indices]
            y = y.loc[common_indices]
            self.stats_text.append(f"После синхронизации: размер y: {len(y)}, размер features: {features.shape[0]}")

        features = features.replace([np.inf, -np.inf], np.nan)
        nan_percentage = features.isna().mean()
        threshold = 0.5  # Порог: удаляем столбцы, где более 50% значений NaN
        columns_to_keep = nan_percentage[nan_percentage <= threshold].index
        self.stats_text.append(f"Удалено столбцов с NaN > {threshold*100}%: {len(features.columns) - len(columns_to_keep)} из {len(features.columns)}")
        features = features[columns_to_keep]

        # Удаляем оставшиеся столбцы с любыми NaN
        features = features.dropna(axis=1)
        self.stats_text.append(f"После полной очистки: {len(features.columns)} признаков осталось")

        if features.empty:
            self.stats_text.append("Ошибка: после очистки признаков данные пусты")
            self.progress_bar.setValue(0)
            return

        # Синхронизация после очистки
        y = y.loc[features.index]
        self.stats_text.append(f"После очистки: размер y: {len(y)}, размер features: {features.shape[0]}")

        # Проверяем y на пропуски
        if y.isna().any():
            self.stats_text.append("Предупреждение: в целевой переменной y есть пропуски, они будут удалены")
            y = y.dropna()
            features = features.loc[y.index]
            self.stats_text.append(f"После удаления пропусков в y: размер y: {len(y)}, размер features: {features.shape[0]}")

        if 'cultures' in features.columns:
            cultures_column = features['cultures']
            features = features.drop(columns=['cultures'])
            self.stats_text.append("Колонка 'cultures' временно удалена из features для обработки в select_features")

        # Отбор значимых признаков с учётом настроек
        try:
            fdr_level = self.pvalue_spinbox.value()
            remove_correlated = self.remove_correlated_checkbox.isChecked()
            corr_threshold = self.corr_spinbox.value()

            self.stats_text.append(f"Настройки отбора: p-value = {fdr_level:.2f}, удаление коррелирующих признаков = {remove_correlated}, порог корреляции = {corr_threshold:.2f}")

            # Отбор значимых признаков по p-value
            selected_features = select_features(features, y, fdr_level=fdr_level)
            self.stats_text.append(f"После отбора по p-value: {len(selected_features.columns)} признаков")

            # Удаление коррелирующих признаков
            if remove_correlated:
                corr_matrix = selected_features.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
                initial_count = len(selected_features.columns)
                selected_features = selected_features.drop(columns=to_drop)
                self.stats_text.append(f"Удалено коррелирующих признаков: {initial_count - len(selected_features.columns)}")

            selected_features['cultures'] = y
            self.stats_text.append("Добавлена колонка 'cultures' с информацией о культурах")

            self.progress_bar.setValue(50)

            # Сохранение отобранных признаков
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            selected_features_path = os.path.join(self.parent.save_folder, f'selected_features_{timestamp}.csv')
            selected_features.to_csv(selected_features_path, index=True, sep=';', encoding='utf-8')

            selected_params = list(selected_features.columns)
            self.stats_text.append("Отобранные параметры (без колонки cultures):\n" + "\n".join([param for param in selected_params if param != 'cultures']))
            self.stats_text.append(f"Отобрано признаков (без учёта колонки cultures): {len(selected_params) - 1}")
            self.stats_text.append(f"Сохранено в {selected_features_path}")
            self.progress_bar.setValue(100)

        except Exception as e:
            self.stats_text.append(f"Ошибка при отборе параметров: {str(e)}")
            import traceback
            self.stats_text.append(f"Трассировка: {traceback.format_exc()}")
            self.progress_bar.setValue(0)
            return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GeoDataPreparing()
    window.show()
    sys.exit(app.exec_())
