from pypdf import PdfReader
from io import BytesIO
import re

from bs4 import BeautifulSoup

#todo resolve format/check errors
def any_format_to_str(file, content_type):
  if content_type == "text/html":
    return soup_html_to_text(file)

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
