import fitz  # PyMuPDF
import re

def chunk_text(text):
    """
    Splits text into a list of words (chunks) for embedding.
    Converts to lowercase and removes punctuation.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Split into words
    words = text.split()
    return words

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
print("--- Extracted Text ---")
print(extracted_content)

text_chunks = chunk_text(extracted_content)
print("\n--- Text Chunks for Embedding ---")
print(text_chunks)
