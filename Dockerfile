# Dockerfile, Image, Container

FROM python:3.10

ADD server.py .
ADD finalScore.py .

COPY training_data.xlsx .
COPY training.xlsx .
COPY finalScore.py .
COPY Social.pkl .
COPY Leadership.pkl .
COPY Intellectual.pkl .
COPY Personal.pkl .
COPY metaData.json .

RUN pip install pandas numpy scikit-learn pypdf2 Flask openpyxl
RUN mkdir -p /Uploads
RUN mkdir -p /templates

COPY templates/ /templates/
RUN ls --recursive /templates/

CMD ["python", "./server.py"]