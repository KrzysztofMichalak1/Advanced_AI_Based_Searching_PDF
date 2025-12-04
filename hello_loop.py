import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: The path to the PDF file.
    :return: The extracted text as a string.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

pdf_file = "Oto opowiadanie.pdf"
extracted_content = extract_text_from_pdf(pdf_file)
print(extracted_content)
