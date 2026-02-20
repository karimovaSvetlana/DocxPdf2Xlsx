"""
Utilities for JSON extraction and cleaning from LLM outputs.
"""

import re
import json
import json_repair


def fix_quotes(text):
    """Fix double quotes inside 'наименование' field by replacing with single quotes."""
    pattern = r'(?<="наименование": ")(.*?)(?=", "рз")'

    def replace_quotes(match):
        return match.group(1).replace('"', "'")

    text_fixed = re.sub(pattern, replace_quotes, text, flags=re.DOTALL)
    return text_fixed if text_fixed else text


def fix_json_numbers(text):
    """Clean number formatting in year fields (remove spaces, replace comma with dot)."""
    pattern = r'("\d{4}_год":\s*")([^"]+)"'

    def replacer(match):
        key_part = match.group(1)  # '"2024_год": "'
        value_part = match.group(2)  # '1 132 667,5'

        # Remove spaces (regular and non-breaking \xa0) and replace comma with dot
        clean_value = value_part.replace(" ", "").replace("\xa0", "").replace(",", ".")

        return f'{key_part}{clean_value}"'

    text_fixed = re.sub(pattern, replacer, text)
    return text_fixed if text_fixed else text


def fix_trailing_commas(text):
    """Remove trailing commas before closing brackets."""
    text_fixed = re.sub(r',\s*([\]}])', r'\1', text)
    return text_fixed if text_fixed else text


def fix_json_repair(text):
    """Attempt to repair malformed JSON using json_repair library."""
    decoded_object = json_repair.loads(text)
    if isinstance(decoded_object, dict) and "rows" in decoded_object:
        return decoded_object
    return text


def extract_json_from_text(text):
    """
    Extract and repair JSON from text output, fixing common LLM errors.

    Args:
        text: Raw text output containing JSON

    Returns:
        Parsed JSON dict or None if extraction failed
    """
    if not text:
        return None

    # Find boundaries of the JSON object { ... }
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx == -1 or end_idx == -1:
        print("Error: JSON braces {} not found in text.")
        return None

    json_str = text[start_idx:end_idx + 1]

    # Apply fixes:
    # 1. Replace double quotes with single quotes inside 'наименование' field
    json_str = fix_quotes(json_str)

    # 2. Clean number formatting in year fields
    json_str = fix_json_numbers(json_str)

    # 3. Remove trailing commas
    json_str = fix_trailing_commas(json_str)

    # 4. Attempt repair with json_repair
    repaired = fix_json_repair(json_str)

    if isinstance(repaired, str):
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    else:
        return repaired


def clean_extracted_data(rows):
    """
    Merge multi-line entries in extracted table data.

    Rows without financial data (year fields) are merged with the previous row's
    'наименование' field, as they represent continuation of the description.

    Args:
        rows: List of row dictionaries

    Returns:
        List of cleaned row dictionaries
    """
    if not rows:
        return []

    cleaned = []
    current_row = None

    for row in rows:
        # Check if row has financial data in any year field
        has_data = any(str(row.get(k, "")).strip() for k in ["2024_год", "2025_год", "2026_год"])

        if not has_data and current_row is not None:
            # No data - append text to previous 'наименование'
            current_row["наименование"] = f"{current_row.get('наименование', '')} {row.get('наименование', '')}".strip()
        else:
            # Has data or first row - create new entry
            if current_row:
                cleaned.append(current_row)
            current_row = row

    if current_row:
        cleaned.append(current_row)

    return cleaned
