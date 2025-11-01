from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytesseract
import pdfplumber
import os

from .utils import ensure_dir
import os

import pytesseract
from PIL import Image

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def extract_page_images(pdf_path: str, out_dir: str) -> List[str]:
    """Rasterize and export page images (PNG) using pdfplumber.
    Returns list of saved image paths."""
    ensure_dir(out_dir)
    saved = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Render at 150 DPI
            im = page.to_image(resolution=150).original  # PIL Image
            fname = f"page_{i:03d}.png"
            fpath = str(Path(out_dir) / fname)
            im.save(fpath)
            saved.append(fpath)
    return saved

def _try_bar_chart_digitize(img: np.ndarray) -> pd.DataFrame | None:
    """Best-effort bar chart digitization.
    - Detects vertical bars via contour analysis.
    - Estimates bar heights relative to chart area.
    - OCRs x-axis labels to name bars when possible.
    Returns DataFrame with columns [label, value_scaled] or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Find contours
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # Heuristic: tall-ish rectangles
        if w > 10 and h > 20 and h > w * 0.8:
            rects.append((x,y,w,h))
    if not rects:
        return None

    # Sort by x for left->right order
    rects.sort(key=lambda r: r[0])

    # Estimate baseline (bottom) and top among detected bars
    bottoms = [y+h for (x,y,w,h) in rects]
    tops = [y for (x,y,w,h) in rects]
    baseline = np.median(bottoms)
    top_ref = np.median(tops)
    height_span = max(1.0, baseline - top_ref)

    values = []
    labels = []

    # Try to OCR x-axis region for labels
    H, W = gray.shape
    axis_region = gray[int(min(H-1, baseline)):H, :]
    try:
        xtext = pytesseract.image_to_string(axis_region)
    except Exception:
        xtext = ""
    # crude tokenization
    ocr_tokens = [t for t in xtext.split() if any(ch.isalnum() for ch in t)]

    for i,(x,y,w,h) in enumerate(rects):
        value_scaled = max(0.0, (baseline - (y+h*0.5)) / height_span)  # center height proxy
        values.append(value_scaled)
        lbl = ocr_tokens[i] if i < len(ocr_tokens) else f"bar_{i+1}"
        labels.append(lbl)

    df = pd.DataFrame({"label": labels, "value_scaled": values})
    return df

def extract_charts_as_data(pdf_path: str, images_dir: str) -> Tuple[List[pd.DataFrame], List[str]]:
    """Exports page images and tries to create DataFrames for simple bar charts.
    Returns (list_of_dataframes, list_of_chart_image_paths).
    """
    imgs = extract_page_images(pdf_path, images_dir)
    dfs = []
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        df = _try_bar_chart_digitize(img)
        if df is not None and not df.empty:
            dfs.append(df)
    return dfs, imgs

def ocr_extract_text_from_images(image_paths):
    """Extracts text from chart images using OCR (pytesseract)."""
    ocr_results = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            ocr_results.append({"image": img_path, "text": text})
        except Exception as e:
            print(f"⚠️ OCR failed for {img_path}: {e}")
    return ocr_results