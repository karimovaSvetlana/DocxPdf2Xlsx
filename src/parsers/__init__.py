"""Parsers for DOCX and PDF table extraction."""

from .docx_parser import extract_tables_from_docx, docx_to_excel
from .pdf_glm_ocr import process_pdf_to_excel as process_pdf_glm
from .pdf_qwen import process_pdf_parallel as process_pdf_qwen

__all__ = [
    'extract_tables_from_docx',
    'docx_to_excel',
    'process_pdf_glm',
    'process_pdf_qwen',
]
