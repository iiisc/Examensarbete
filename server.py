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

@app.route('/test', methods=['GET'])
def test():
    return "Hello"

@app.route('/upload', methods=['POST'])
def upload_file():
        if 'file' not in request.files:
            print('No file part')    
            return redirect(request.url)
        file = request.files['file']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        path = os.path.join("Uploads", file.filename)
        return(readPdf(path))

if __name__ == '__main__':
    app.run()