import openpyxl


# путь к классификатору полей
file_path = "data/Номер-культура 2023 Хабаровск (шейп_Вега_2023_с атрибутами).xlsx"
sheet_name = "Лист1"  # имя листа


def parse_field_data(file_path, sheet_name=None):
    wb = openpyxl.load_workbook(file_path)
    sheet = wb[sheet_name] if sheet_name else wb.active

    field_data = {}

    for row in sheet.iter_rows(min_row=2, values_only=True):
        field_number = row[0]  # Номер поля в первом столбце
        crop_culture = row[1]  # Культура — во втором столбце

        if field_number is not None and crop_culture is not None:
            field_data[field_number] = crop_culture

    return field_data


def fields2culture_convert_culture2fields(fields_to_culture):
    culture_data = []
    return dict(zip(fields_to_culture.values(), fields_to_culture.keys()))

# словарь поле-культура
fields_to_culture = parse_field_data(file_path, sheet_name)
culture_to_fields = fields2culture_convert_culture2fields(fields_to_culture)
