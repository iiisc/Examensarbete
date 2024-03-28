import os
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

UPLOAD_FOLDER = 'Uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def readPdf(file):
    reader = PdfReader(file)
    page = reader.pages[0]
    return(page.extract_text())

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
        if 'file' not in request.files:
            print('No file part')    
            return redirect(request.url)
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        filepath = os.path.join("Uploads", file.filename)
        extension = os.path.splitext(filepath)[-1]
        
        if extension == '.pdf':
            return(readPdf(filepath))
        
        return("File uploaded")

if __name__ == '__main__':
    app.run()