FROM python:3.11


# Install necessary packages for OCR
RUN apt-get update -y && apt-get install -y tesseract-ocr && apt-get install ffmpeg libsm6 libxext6 -y && apt-get install libgl1 -y && apt-get install pandoc -y

#if you are using a specific version, specify here

RUN pip install nltk numpy

RUN python3 -c "import nltk ; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"


ENV PYTHONPATH=/api
EXPOSE $PORT

# WORKDIR /code
WORKDIR /api

# COPY ./requirements.txt /code/requirements.txt
COPY requirements.txt /requirements.txt

# RUN pip install --upgrade -r /code/requirements.txt
RUN pip install --upgrade -r /requirements.txt

# COPY ./app /code/app
COPY ./api /api/api

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]