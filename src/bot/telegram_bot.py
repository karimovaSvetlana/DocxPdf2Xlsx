"""
Telegram bot for DOCX table extraction.

Usage:
1. Set your bot token in TOKEN variable
2. Run: python telegram_bot.py

The bot will receive DOCX files, extract all tables, and send back Excel file(s).
"""

import os
import logging
import asyncio
import pandas as pd
from pathlib import Path
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import FSInputFile

import sys
sys.path.append(str(Path(__file__).parent.parent))
from parsers.docx_parser import extract_tables_from_docx


TOKEN = "..."  # Set your Telegram bot token here
bot = Bot(token=TOKEN)
dp = Dispatcher()

tmp_dir = Path("tmp_processing")
tmp_dir.mkdir(exist_ok=True)


async def split_and_send_excel(message: types.Message, df: pd.DataFrame, base_name: str):
    """
    Split Excel into chunks if too large and send to user.

    Args:
        message: Telegram message to reply to
        df: DataFrame to send
        base_name: Base name for output file
    """
    max_rows = 500000  # Max rows per file
    chunks = [df[i:i + max_rows] for i in range(0, df.shape[0], max_rows)]

    for i, chunk in enumerate(chunks):
        part_name = f"{base_name}_part_{i+1}.xlsx" if len(chunks) > 1 else f"{base_name}.xlsx"
        chunk.to_excel(part_name, index=False, header=False)

        caption = f"Here's your table! Part {i+1}" if len(chunks) > 1 else "All tables from the file:"
        await message.answer_document(FSInputFile(part_name), caption=caption)
        os.remove(part_name)


@dp.message(F.document)
async def handle_docs(message: types.Message):
    """Handle incoming document messages."""
    if not message.document.file_name.lower().endswith(".docx"):
        await message.reply("Cannot accept this format. Please send a .docx file")
        return

    msg = await message.answer("Downloading and processing file...")

    file_id = message.document.file_id
    file_name = message.document.file_name
    local_docx = tmp_dir / f"{message.chat.id}_{file_id}.docx"

    try:
        await bot.download(file_id, destination=local_docx)

        tables = extract_tables_from_docx(local_docx)

        if not tables:
            await msg.edit_text(f"No tables found in `{file_name}`.")
            return

        final_df = pd.concat(tables, ignore_index=True)

        await split_and_send_excel(message, final_df, Path(file_name).stem)
        await msg.delete()

    except Exception as e:
        logging.error(f"Error processing docx: {e}")
        await msg.edit_text("An error occurred while processing the file.")

    finally:
        if local_docx.exists():
            os.remove(local_docx)


@dp.message()
async def other_messages(message: types.Message):
    """Handle all other messages."""
    await message.answer("Send me a .docx file and I'll extract all tables to Excel!")


async def main():
    """Start the bot."""
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
