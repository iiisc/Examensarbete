import os
import pandas as pd
import scoringmodel
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
   return render_template('index.html')

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
    all_files = (os.listdir(os.path.join(app.config['UPLOAD_FOLDER'])))
    model = scoringmodel.Model()
    model.train_model()
    result = model.run_model(all_files)

    df = pd.DataFrame.from_dict(result, orient='index')
    df.style.set_properties(**{'text-align': 'left'})
    df_html = df.to_html(classes=["table table-bordered table-striped table-hover"])

    return render_template('table.html', table_html = df_html)

@app.route('/get_files', methods=['GET'])
def get_files():
    all_files = (os.listdir(os.path.join(app.config['UPLOAD_FOLDER'])))
    df = pd.DataFrame({'Uploaded files:' : all_files})
    df_html = df.to_html(index = False, classes=["table table-bordered table-striped table-hover"])
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

@app.route('/api/score_cv', methods = ['POST'])
def api_upload_and_score():
    files = request.files.getlist("file")
    file_names = []
    if request.files['file'].filename != '':
        for file in files:
            if file and allowed_file(file.filename):
                file_names.append(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))                

        if file_names:
            model = scoringmodel.Model()
            model.train_model()
            result = model.run_model(file_names)
            return jsonify(result), 200
    return "No file with .pdf extension uploaded"

if __name__ == '__main__':
    app.run()