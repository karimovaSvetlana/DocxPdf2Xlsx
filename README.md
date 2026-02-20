# DocxPdf2Xlsx

A toolkit for extracting tables from DOCX and PDF documents and converting them to Excel format using vision-language models.

## Overview

DocxPdf2Xlsx provides two main capabilities:

1. **DOCX Table Extraction**: Extract all tables from DOCX files and convert them to Excel format
2. **PDF Table Extraction**: Use vision-language models (GLM-OCR or Qwen3-VL) to extract structured table data from PDF documents through OCR

The DOCX parser can be used both as a standalone tool and integrated with a Telegram bot for easy user access.

## Features

- Standalone DOCX to Excel conversion
- Telegram bot for DOCX table extraction
- PDF to Excel conversion using state-of-the-art vision-language models
- Two PDF processing options:
  - GLM-OCR: Lightweight model for basic extraction
  - Qwen3-VL: Advanced model with parallel processing and resume support
- Automatic handling of multi-line table entries
- JSON repair and cleaning utilities for robust LLM output processing

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended for PDF processing)
- For PDF conversion: poppler-utils

```bash
# Install poppler on Ubuntu/Debian
sudo apt-get install poppler-utils

# Install poppler on macOS
brew install poppler
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### DOCX to Excel (Standalone)

Extract all tables from a DOCX file:

```bash
python src/parsers/docx_parser.py input.docx output.xlsx
```

Or use the default output filename (input.xlsx):

```bash
python src/parsers/docx_parser.py input.docx
```

### DOCX to Excel (Telegram Bot)

1. Get a bot token from [@BotFather](https://t.me/botfather)
2. Edit `src/bot/telegram_bot.py` and set your token:
   ```python
   TOKEN = "your_bot_token_here"
   ```
3. Run the bot:
   ```bash
   python src/bot/telegram_bot.py
   ```
4. Send a DOCX file to the bot and receive Excel file(s) in return

### PDF to Excel (GLM-OCR)

Simple PDF table extraction using GLM-OCR model:

```bash
python src/parsers/pdf_glm_ocr.py input.pdf output.xlsx
```

This will:
- Load the GLM-OCR vision-language model
- Convert each PDF page to an image
- Extract structured table data using the model
- Save results to Excel

### PDF to Excel (Qwen3-VL)

Advanced PDF processing with parallel execution and resume support:

```bash
python src/parsers/pdf_qwen.py input.pdf output_directory prompt.txt
```

Features:
- Parallel page processing for faster execution
- Resumable processing (can continue after interruption)
- Progress logging in JSONL format
- Better accuracy with Qwen3-VL model

The output will be saved to `output_directory/<pdf_name>/<pdf_name>.xlsx`

## Project Structure

```
DocxPdf2Xlsx/
├── src/
│   ├── parsers/           # Document parsers
│   │   ├── docx_parser.py # DOCX table extraction
│   │   ├── pdf_glm_ocr.py # PDF extraction with GLM-OCR
│   │   └── pdf_qwen.py    # PDF extraction with Qwen3-VL
│   ├── utils/             # Utility functions
│   │   └── json_utils.py  # JSON extraction and cleaning
│   └── bot/               # Telegram bot
│       └── telegram_bot.py
├── data/
│   ├── input/             # Input files
│   ├── output/            # Generated outputs
│   └── prompts/           # Prompt templates for PDF extraction
├── requirements.txt       # Python dependencies
└── README.md
```

## How It Works

### DOCX Extraction

The DOCX parser uses the `python-docx` library to:
1. Read all tables from the document
2. Extract cell contents preserving structure
3. Remove duplicate rows
4. Combine all tables into a single Excel file

### PDF Extraction

PDF extraction uses vision-language models:

1. **Image Conversion**: PDF pages are converted to images using `pdf2image`
2. **Model Processing**: Each page image is processed by a vision-language model with a structured prompt
3. **JSON Extraction**: The model outputs JSON-formatted table data
4. **Cleaning**: Multiple cleaning steps fix common LLM formatting errors:
   - Quote escaping in text fields
   - Number formatting (remove spaces, convert commas to dots)
   - Trailing comma removal
   - Automatic JSON repair
5. **Merging**: Multi-line entries are merged based on presence of numerical data
6. **Export**: Final data is saved to Excel

## Requirements

### Core Dependencies

- pandas: Data manipulation and Excel export
- openpyxl: Excel file writing
- python-docx: DOCX file parsing

### PDF Processing

- torch: Deep learning framework
- transformers: Hugging Face transformers library
- pdf2image: PDF to image conversion
- pillow: Image processing
- json-repair: Automatic JSON repair

### Telegram Bot

- aiogram: Telegram bot framework

### Models

- GLM-OCR: `zai-org/GLM-OCR`
- Qwen3-VL: `Qwen/Qwen3-VL-8B-Instruct`

Models are automatically downloaded from Hugging Face on first use.

## Tips and Best Practices

### For DOCX Files

- Works best with well-formatted tables
- Automatically removes duplicate rows
- All tables are combined into a single output file

### For PDF Files

- GLM-OCR is faster but less accurate
- Qwen3-VL provides better accuracy at the cost of speed and memory
- Use Qwen3-VL for large documents with the resume feature
- Customize prompts in `data/prompts/` for your specific table structure
- GPU is highly recommended (CUDA-compatible)
- Processing time depends on document length and model used

### General

- Ensure sufficient disk space for temporary files
- For large PDFs, monitor GPU memory usage
- Check `raw_outputs.jsonl` for debugging extraction issues

## Limitations

- PDF extraction requires GPU for acceptable performance
- Complex table layouts may require prompt tuning
- Russian field names in table schema (can be customized in prompts)
- PDF extraction assumes specific table structure (budget tables)

## Author

Created by @svet_ds

## License

This project is provided as-is for educational and research purposes.
