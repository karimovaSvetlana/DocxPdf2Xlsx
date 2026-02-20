"""
How to start:
1. Fill telegram-bot token in `TOKEN = "..."`
2. Go to bot folder and start:
cd docx_table_searcher_bot
python bot.py
"""

import os
import logging
import asyncio
import pandas as pd
from pathlib import Path
from aiogram import Bot, Dispatcher, types, F

from utils import extract_tables_from_docx, split_and_send_excel


TOKEN = "..."
bot = Bot(token=TOKEN)
dp = Dispatcher()

tmp_dir = Path("tmp_processing")
tmp_dir.mkdir(exist_ok=True)

@dp.message(F.document)
async def handle_docs(message: types.Message):
    if not message.document.file_name.lower().endswith(".docx"):
        await message.reply("❌ Не могу принять этот формат. Пожалуйста, пришли файл .docx")
        return

    msg = await message.answer("⏳ Качаю и обрабатываю файл...")
    
    file_id = message.document.file_id
    file_name = message.document.file_name
    local_docx = tmp_dir / f"{message.chat.id}_{file_id}.docx"
    
    try:
        await bot.download(file_id, destination=local_docx)
        
        tables = extract_tables_from_docx(local_docx)
        
        if not tables:
            await msg.edit_text(f"ℹ️ В файле `{file_name}` не найдено ни одной таблицы.")
            return

        final_df = pd.concat(tables, ignore_index=True)
        
        await split_and_send_excel(message, final_df, Path(file_name).stem)
        await msg.delete()

    except Exception as e:
        logging.error(f"Error processing docx: {e}")
        await msg.edit_text("💥 Произошла ошибка при обработке файла.")
    
    finally:
        if local_docx.exists():
            os.remove(local_docx)

@dp.message()
async def other_messages(message: types.Message):
    await message.answer("Пришли мне файл `.docx`, и я вытащу из него все таблицы в Excel!")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())