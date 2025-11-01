# PDF Text + Graphical Data Q&A (Python)

This project extracts **text**, **tables**, and **chart images** from PDFs, converts them into **queryable data**, and enables **natural-language Q&A** over both the narrative text and tabular/derived data. Results and documentation are presented in a **Jupyter notebook**.

## Features
- Text extraction with page-level provenance (pdfplumber).
- Table extraction via **Camelot** (stream/lattice) with fallback to **pdfplumber** and **tabula-py**.
- Chart *image* extraction (saves raster images per page) plus light heuristics to detect common **bar charts** and turn them into data (best-effort).
- Unified data store: text corpus + list of DataFrames (tables + any chart-derived frames).
- Embedding-based semantic search (Sentence-Transformers) with sklearn/FAISS backend.
- Simple NL-to-table querying (column name detection, filters, aggregations) with guardrails.
- End-to-end demo in `notebooks/analysis.ipynb`.

> Note: Converting arbitrary charts to perfect numeric data is an open research problem.
> This project implements a pragmatic baseline that works well for **tables** and some **simple bar charts** (clearly separated bars with axis labels).

## Quickstart
1. Create and activate a Python 3.10+ virtual env.
2. `pip install -r requirements.txt`
3. If on Windows, install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and set `TESSERACT_CMD` in `.env` (copy from `.env-example`).
4. Put your PDF in `data/`.
5. Launch `jupyter lab` or `jupyter notebook` and run `notebooks/analysis.ipynb`.

## Project Structure
```
pdf-qa-challenge/
├── data/                           # place PDFs here
├── outputs/                        # extracted assets & caches
├── notebooks/
│   └── analysis.ipynb              # main demo notebook
├── src/
│   ├── extract_text.py
│   ├── extract_tables.py
│   ├── extract_charts.py
│   ├── build_index.py
│   ├── qa.py
│   └── utils.py
├── requirements.txt
├── .env-example
├── .env
└── README.md
```

## Notes on Dependencies
- **tabula-py** requires Java. If unavailable, the code will skip Tabula gracefully.
- **camelot** requires Ghostscript for lattice mode; stream mode works most of the time.
- **faiss-cpu** is optional; sklearn fallback is used automatically if FAISS import fails.

## License
MIT
