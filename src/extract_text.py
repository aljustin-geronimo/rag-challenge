from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import pdfplumber
from pathlib import Path

@dataclass
class PageText:
    page_num: int
    text: str

def extract_text(pdf_path: str) -> List[PageText]:
    pages: List[PageText] = []
    pdf_path = str(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append(PageText(page_num=i, text=txt))
    return pages
