import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QTableWidget, QTableWidgetItem, QCheckBox,
                             QScrollArea, QGridLayout, QDialog, QTextEdit)
from PyQt5.QtCore import Qt
import openpyxl
from collections import defaultdict

class GeoDataPreparing(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GeoDataPreparing')
        self.setGeometry(100, 100, 1200, 700)

        # Левая часть
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

        self.coefficients_tab = QTableWidget()
        self.coefficients_tab.setColumnCount(2)
        self.coefficients_tab.setHorizontalHeaderLabels(['Коэффициент', 'Значение'])

        self.bottom_left_tabs = QTabWidget()
        self.bottom_left_tabs.addTab(self.fields_tab, 'Поля')
        self.bottom_left_tabs.addTab(self.coefficients_tab, 'Коэффициенты')

        self.left_layout = QVBoxLayout()
        self.left_layout.addWidget(self.left_tabs)
        self.left_layout.addWidget(self.bottom_left_tabs)

        self.left_container = QWidget()
        self.left_container.setLayout(self.left_layout)
        self.left_container.setFixedWidth(400)

        # Правая часть
        self.right_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.right_layout.addWidget(self.stats_text)

        self.right_container = QWidget()
        self.right_container.setLayout(self.right_layout)

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.left_container)
        self.main_layout.addWidget(self.right_container)

        self.setLayout(self.main_layout)

    def create_data_source_tab(self, source_name):
        tab = QWidget()
        layout = QVBoxLayout()

        # Путь к файлу с данными
        layout.addWidget(QLabel(f'Путь к файлу {source_name}:'))
        file_path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setStyleSheet("text-align: left;")
        self.file_path_button = QPushButton('Выбрать')
        self.file_path_button.clicked.connect(lambda: self.select_file(self.file_path_edit))
        file_path_layout.addWidget(self.file_path_edit)
        file_path_layout.addWidget(self.file_path_button)
        layout.addLayout(file_path_layout)

        # Путь к файлу для сопоставления названий полей
        layout.addWidget(QLabel('Путь к файлу для сопоставления названий полей:'))
        fields_mapping_path_layout = QHBoxLayout()
        self.fields_mapping_path_edit = QLineEdit()
        self.fields_mapping_path_edit.setStyleSheet("text-align: left;")
        self.fields_mapping_path_button = QPushButton('Выбрать')
        self.fields_mapping_path_button.clicked.connect(self.select_fields_mapping_file)
        fields_mapping_path_layout.addWidget(self.fields_mapping_path_edit)
        fields_mapping_path_layout.addWidget(self.fields_mapping_path_button)
        layout.addLayout(fields_mapping_path_layout)

        # Контейнер для чекбоксов культур
        self.cultures_checkboxes_grid = QGridLayout()
        layout.addLayout(self.cultures_checkboxes_grid)

        self.show_stats_button = QPushButton('Показать статистику')
        self.show_stats_button.clicked.connect(self.show_statistics)
        layout.addWidget(self.show_stats_button)

        tab.setLayout(layout)
        return tab

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Выбрать файл')
        if file_path:
            line_edit.setText(file_path)

    def select_fields_mapping_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Выбрать файл сопоставления полей', '', 'Excel Files (*.xlsx)')
        if file_path:
            # Получаем активную вкладку
            current_tab_index = self.left_tabs.currentIndex()
            current_tab = self.left_tabs.widget(current_tab_index)
            # Находим поле пути в активной вкладке
            for i in range(current_tab.layout().count()):
                item = current_tab.layout().itemAt(i)
                if item.widget() and isinstance(item.widget(), QLineEdit):
                    self.fields_mapping_path_edit = item.widget()
                    break
            self.fields_mapping_path_edit.setText(file_path)
            self.load_cultures_from_excel(file_path, current_tab)

    def load_cultures_from_excel(self, file_path, tab):
        try:
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            cultures = defaultdict(list)
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if row[1] and row[0]:
                    cultures[row[1]].append(row[0])
            self.create_culture_checkboxes(cultures, tab)
            self.cultures_data = cultures
        except Exception as e:
            print(f"Ошибка при чтении файла Excel: {e}")

    def create_culture_checkboxes(self, cultures, tab):
        for i in reversed(range(tab.layout().count())):
            item = tab.layout().itemAt(i)
            if item.widget() and isinstance(item.widget(), QCheckBox):
                item.widget().setParent(None)

        for i in reversed(range(tab.layout().count())):
            item = tab.layout().itemAt(i)
            if item.layout() and isinstance(item.layout(), QGridLayout):
                for j in reversed(range(item.layout().count())):
                    item.layout().itemAt(j).widget().setParent(None)
                item.layout().setParent(None)

        self.culture_checkboxes = {}
        self.cultures_checkboxes_grid = QGridLayout()
        row = 0
        col = 0
        for culture in cultures:
            checkbox = QCheckBox(culture)
            self.cultures_checkboxes_grid.addWidget(checkbox, row, col)
            self.culture_checkboxes[culture] = checkbox
            col += 1
            if col > 1:
                col = 0
                row += 1
        tab.layout().insertLayout(tab.layout().count() - 1, self.cultures_checkboxes_grid)  # Добавляем QGridLayout перед кнопкой "Показать статистику"

    def show_statistics(self):
        selected_cultures = [culture for culture, checkbox in self.culture_checkboxes.items() if checkbox.isChecked()]
        total_fields = sum(len(self.cultures_data.get(culture, [])) for culture in selected_cultures)
        unique_cultures = len(selected_cultures)

        stats_text = f"Выбрано культур: {unique_cultures}\n"
        stats_text += f"Общее количество полей: {total_fields}\n"
        for culture in selected_cultures:
            stats_text += f"- {culture}: {len(self.cultures_data.get(culture, []))} полей\n"

        # Отображаем статистику в правой части интерфейса
        self.stats_text.setText(stats_text)

        # Заполняем таблицу "Поля"
        self.fill_fields_table(selected_cultures)

    def fill_fields_table(self, selected_cultures):
        self.fields_tab.setRowCount(0)  # Очищаем таблицу
        for culture in selected_cultures:
            for field in self.cultures_data.get(culture, []):
                row_position = self.fields_tab.rowCount()
                self.fields_tab.insertRow(row_position)
                self.fields_tab.setItem(row_position, 0, QTableWidgetItem(field))
                self.fields_tab.setItem(row_position, 1, QTableWidgetItem(culture))

    def start_processing(self):
        # Здесь будет код для запуска обработки данных
        print('Начинаем обработку данных...')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GeoDataPreparing()
    window.show()
    sys.exit(app.exec_())