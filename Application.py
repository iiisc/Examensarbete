from PyPDF2 import PdfReader

pdf_path = 'CV.pdf'
reader = PdfReader(pdf_path)
page = reader.pages[0]
print(page.extract_text())

