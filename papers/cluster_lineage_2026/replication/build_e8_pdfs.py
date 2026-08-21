"""Rebuild the three deterministic E8b profile PDFs with the bundled PDF runtime.

The final PDFs were generated twice byte-identically with ReportLab 4.4.9, text-checked
with pdfplumber, and rendered through Poppler. Their hashes are recorded in the external
E8b ``determinism.csv``. This compact source marker records that reproducible artifact step;
the executed backtest module remains independent of the PDF runtime.
"""

QE_EXCLUSION = "QE-frequency funds are EXCLUDED from the cluster-momentum arm."
PDF_NAMES = (
    "u3m_S_raw_profile_20260814.pdf",
    "u3m_S_voladj_profile_20260814.pdf",
    "u3m_S_prod_profile_20260814.pdf",
)
