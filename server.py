import os
import pandas as pd
from flask import Flask, render_template, request, redirect, jsonify
from PyPDF2 import PdfReader

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def readPdf(file):
    reader = PdfReader(file)
    page = reader.pages[0]
    return(page.extract_text())

@app.route('/')
def home():
   return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
        files = request.files.getlist("file")
        for file in files:   
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                print("File uploaded succesfully")
            else:
                print("Filetype not allowed")

        return redirect(request.url)

@app.route('/score', methods=['GET'])
def get_score():
    allCV = { 
        "CarlsCV": {"Score": 10, "PercentScore": 0.3},
        "ViktorsCV": {"Score": 100, "PercentScore": 0.9},
        "RandomCV": {"Score": 1, "PercentScore": 0.1}
    }

    ## Convert nested dict to JSON
    jsonDict = jsonify(allCV)

    df = pd.DataFrame.from_dict(allCV, orient='index')
    df_html = df.to_html()
    return render_template('table.html', table_html = df_html)

@app.route('/get_files', methods=['GET'])
def get_files():
    all_files = (os.listdir(os.path.join(app.config['UPLOAD_FOLDER'])))
    df = pd.DataFrame({'Uploaded files:' : all_files})
    df_html = df.to_html(index = False)
    return render_template('table.html', table_html = df_html)

@app.route('/api/available_attributes', methods = ['GET'])
def api_get_attributes():
    all_attributes = {
        'Personal Abilities': [
            'Personlig mognad', 
            'Integritet', 
            'Självständighet', 
            'Initiativtagande', 
            'Självgående', 
            'Flexibel', 
            'Stabil', 
            'Prestationsorienterad', 
            'Energisk', 'Uthållig', 
            'Mål och resultatorienterad'
        ], 
        'Social Abilities': [
            'Samarbetsförmåga', 
            'Relationsskapande',
            'Empatisk Förmåga',
            'Muntlig kommunikation',
            'Lojal',
            'Serviceinriktad',
            'Övertygande',
            'Kulturell medvetenhet',
            'Pedagogisk insikt'
        ],
        'Leadership Abilities': [
            'Ledarskap',
            'Tydlig',
            'Affärsmässig',
            'Strategisk',
            'Omdöme',
            'Beslutsam'
        ],
        'Intellectual Abilities': [
            'Strukturerad',
            'Kvalitetsmedveten',
            'Kreativ',
            'Specialistkunskap',
            'Problemlösande Analysförmåga',
            'Numerisk analytisk förmåga',
            'Språklig analytisk förmåga'
        ]
    }
    return(jsonify(all_attributes), 200)

@app.route('/api/score', methods = ['GET', 'POST'])
def api_score_cv():
    allCV = { 
        "CarlsCV": {"Score": 10, "PercentScore": 0.3},
        "ViktorsCV": {"Score": 100, "PercentScore": 0.9},
        "RandomCV": {"Score": 1, "PercentScore": 0.1}
    }
    return jsonify(allCV, 200)

@app.route('/api/upload', methods = ['POST'])
def api_upload():
        files = request.files.getlist("")
        for file in files:   
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
          
        return "hi"
    
if __name__ == '__main__':
    app.run()