from docling_core.types.io import DocumentStream
from pypdf import PdfReader
from io import BytesIO

from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

def any_format_to_str(file, content_type):
  if content_type == "text/html":
    # return soup_html_to_text(file)
    return parse_text_by_docling(file)
  if content_type == "application/pdf":
    return pdf_to_text(file)

  return "CAN NOT PROCESS THIS FILE TYPE"

def soup_html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return text

def pdf_to_text(pdf):
    loader = PdfReader(BytesIO(pdf))

    full_text = []
    for page in loader.pages:
        text = page.extract_text()

        if text:
            full_text.append(text)
    return " ".join(full_text)

def parse_text_by_docling(file):
    converter = DocumentConverter()
    stream = DocumentStream(
        name="uploaded_file",
        # mime_type="text/html",
        stream=BytesIO(file)
    )

    result = converter.convert(stream)
    return result.document.export_to_markdown()
    # return stream.document.export_to_markdown()
    # return result.document.export_to_text()
    # return result.document.export_to_dict()

