# back-end/pdf_helpers.py
from io import BytesIO
from pdfminer.high_level import extract_text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract raw text from PDF bytes using pdfminer.six
    """
    pdf_io = BytesIO(pdf_bytes)
    text = extract_text(pdf_io)
    return text
