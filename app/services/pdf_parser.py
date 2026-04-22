from pypdf import PdfReader


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return "\n".join(pages)
