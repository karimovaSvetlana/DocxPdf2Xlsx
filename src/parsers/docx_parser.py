"""
DOCX table extraction parser.

Extracts all tables from DOCX files and converts them to pandas DataFrames.
Can be used standalone or integrated with the Telegram bot.
"""

import pandas as pd
from pathlib import Path
from typing import List
from docx import Document


def extract_tables_from_docx(docx_path: str | Path) -> List[pd.DataFrame]:
    """
    Extract all tables from a DOCX file.

    Args:
        docx_path: Path to the DOCX file

    Returns:
        List of DataFrames, one per table found in the document
    """
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


def docx_to_excel(docx_path: str | Path, output_path: str | Path) -> None:
    """
    Convert all tables from DOCX to a single Excel file.

    Args:
        docx_path: Path to input DOCX file
        output_path: Path to output Excel file
    """
    tables = extract_tables_from_docx(docx_path)

    if not tables:
        print(f"No tables found in {docx_path}")
        return

    # Combine all tables into one DataFrame
    final_df = pd.concat(tables, ignore_index=True)
    final_df.to_excel(output_path, index=False, header=False)
    print(f"Saved {len(tables)} table(s) to {output_path}")


def main():
    """CLI entry point for standalone usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docx_parser.py <input.docx> [output.xlsx]")
        print("If output path is not provided, will use input filename with .xlsx extension")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: File {input_path} not found")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_suffix('.xlsx')

    docx_to_excel(input_path, output_path)


if __name__ == "__main__":
    main()
