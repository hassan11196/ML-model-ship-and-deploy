import os

from fastapi import FastAPI


app = FastAPI(port=os.environ.get('PORT'))


@app.get('/')
def index():
    
    return {'status':'trained'}
