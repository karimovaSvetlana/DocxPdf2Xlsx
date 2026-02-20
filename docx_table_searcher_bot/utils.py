import os
import pandas as pd
from docx import Document
from aiogram import types
from aiogram.types import FSInputFile


def extract_tables_from_docx(docx_path):
    document = Document(docx_path)
    all_data = []
    for table in document.tables:
        table_data = []
        for row in table.rows:
            row_content = [cell.text.strip() for cell in row.cells]
            table_data.append(row_content)
        if not table_data:
            continue
        df = pd.DataFrame(table_data)
        df = df.drop_duplicates().reset_index(drop=True)
        all_data.append(df)
    return all_data


async def split_and_send_excel(message: types.Message, df: pd.DataFrame, base_name: str):
    """Разбивает Excel на части, если строк слишком много или файл потенциально велик."""
    max_rows = 500000  # Max rows per file
    chunks = [df[i:i + max_rows] for i in range(0, df.shape[0], max_rows)]
    
    for i, chunk in enumerate(chunks):
        part_name = f"{base_name}_part_{i+1}.xlsx" if len(chunks) > 1 else f"{base_name}.xlsx"
        chunk.to_excel(part_name, index=False, header=False)
        
        await message.answer_document(
            FSInputFile(part_name), 
            caption=f"Лови таблицу! Часть {i+1}" if len(chunks) > 1 else "Все таблицы из файла:"
        )
        os.remove(part_name)
