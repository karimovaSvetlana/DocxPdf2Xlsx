import re
import json
import json_repair


def fix_quotes(text):
    pattern = r'(?<="наименование": ")(.*?)(?=", "рз")'

    def replace_quotes(match):
        return match.group(1).replace('"', "'")
    text_fixed = re.sub(pattern, replace_quotes, text, flags=re.DOTALL)

    if text_fixed:
        return text_fixed
    return text

def fix_json_numbers(text):
    pattern = r'("\d{4}_год":\s*")([^"]+)"'
    
    def replacer(match):
        key_part = match.group(1) # Это '"2024_год": "'
        value_part = match.group(2) # Это само число '1 132 667,5'
        
        # Удаляем пробелы (обычные и неразрывные \xa0)
        # И заменяем запятую на точку
        clean_value = value_part.replace(" ", "").replace("\xa0", "").replace(",", ".")
        
        return f'{key_part}{clean_value}"'

    text_fixed = re.sub(pattern, replacer, text)
    if text_fixed:
        return text_fixed
    return text

def fix_trailing_commas(text):
    text_fixed = re.sub(r',\s*([\]}])', r'\1', text)

    if text_fixed:
        return text_fixed
    return text
    
def fix_json_repair(text):
    decoded_object = json_repair.loads(text)
    if isinstance(decoded_object, dict) and "rows" in decoded_object:
        return decoded_object
    return text

def extract_json_from_text(text):
    """
    Извлекает JSON из текста, исправляя типичные ошибки LLM.
    """
    if not text:
        return None

    # 1. Пытаемся найти границы самого глубокого/большого объекта { ... }
    # Находим первую { и последнюю }
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx == -1 or end_idx == -1:
        print("Ошибка: Символы JSON {} не найдены в тексте.")
        return None

    json_str = text[start_idx:end_idx + 1]

    # 2. Чистка текста
    # json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    # 2.1 Меняем двойные кавычки на одинарные между `"наименование": "` и `", "рз"`
    json_str = fix_quotes(json_str)

    # 2.2 Меняем в чистах 202X_год запятые на точки и удаляем пробелы
    json_str = fix_json_numbers(json_str)

    # 2.3 Удаляем запятую перед закрытием json (trailing comma)
    json_str = fix_trailing_commas(json_str)

    # 2.4 Восстанавливаем все что можно
    repaired = fix_json_repair(json_str)

    if isinstance(repaired, str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return None
    else:
        return repaired


def clean_extracted_data(rows):
    if not rows: 
        return []
    cleaned = []
    current_row = None
    for row in rows:
        has_data = any(str(row.get(k, "")).strip() for k in ["2024_год", "2025_год", "2026_год"])
        if not has_data and current_row is not None:
            current_row["наименование"] = f"{current_row.get('наименование', '')} {row.get('наименование', '')}".strip()
        else:
            if current_row:
                cleaned.append(current_row)
            current_row = row
    if current_row:
        cleaned.append(current_row)
    return cleaned
