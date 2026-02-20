import torch
import json
import re
import pandas as pd
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "zai-org/GLM-OCR"

# Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Загрузка модели (может занять время)...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)
model.eval()


def clean_extracted_data(rows):
    if not rows:
        return []
    
    cleaned = []
    current_row = None

    for row in rows:
        # Проверяем, есть ли в строке финансовые данные (2024_год)
        # Мы ищем любое число, чтобы понять, завершена ли строка
        has_data = any(str(row.get(k, "")).strip() for k in ["2024_год", "2025_год", "2026_год"])

        if not has_data and current_row is not None:
            # Если чисел нет — приклеиваем текст к предыдущему наименованию
            current_row["наименование"] += " " + row.get("наименование", "")
        else:
            # Если числа есть или это первая строка — создаем новую запись
            if current_row:
                cleaned.append(current_row)
            current_row = row
            
    if current_row:
        cleaned.append(current_row)
    return cleaned


def process_pdf_to_excel(pdf_path, output_xlsx):
    pages = convert_from_path(pdf_path)
    all_extracted_data = []

    json_schema = {
        "rows": [{
            "наименование": "", "рз": "", "пр": "", "цср": "", "вр": "",
            "2024_год": "", "2025_год": "", "2026_год": ""
        }]
    }
    
    prompt_text = f"""请按下列JSON格式输出图中信息
Rules:
1. "наименование" (Description) often spans multiple lines. You MUST merge these lines into a single string.
2. If a line has text but NO numbers, it is a continuation of the previous "наименование".
3. The numbers are usually at the bottom of the text block - use this to identify the end of a record.
5. Strict JSON output only.

Schema:
{json.dumps(json_schema, ensure_ascii=False)}
"""

    for i, page_image in enumerate(pages):
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Парсинг страницы {i+1}...")
        
        # Формируем контент строго по доке
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image.convert("RGB")},
                    {"type": "text", "text": prompt_text}
                ],
            }
        ]

        # Применяем шаблон чата
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # vLLM/Transformers баг: удаляем лишние ключи, если они есть
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=4096,
                do_sample=False # Для OCR лучше выключить семплирование
            )

        # Декодируем только ответ модели (отрезаем входные токены)
        input_len = inputs["input_ids"].shape[1]
        output_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        
        end_time = datetime.now()
        duration = str(end_time - start_time).split(".")[0]
        print(f"[{end_time.strftime('%H:%M:%S')}] Ответ страницы {i+1}:\n{output_text[:100]}...\n== Duration: {duration}")

        with open("output_texts.jsonl", "a") as f:
            f.write(json.dumps({"n": i, "output_text": output_text}, ensure_ascii=False) + '\n')

        try:
            json_str = re.search(r'\{.*\}', output_text, re.DOTALL).group()
            page_data = json.loads(json_str)
            if "rows" in page_data:
                all_extracted_data.extend(page_data["rows"])
            else:
                print(f"Не смог чет спарсить хз: {page_data}")
        except Exception:
            print(f"Ошибка парсинга JSON на стр. {i+1}")

    if all_extracted_data:
        pd.DataFrame(all_extracted_data).to_excel(output_xlsx, index=False)
        print(f"Сохранено в {output_xlsx}")

if __name__ == "__main__":
    process_pdf_to_excel("2024_Прил15_проект_test.pdf", "budget_ai_output.xlsx")