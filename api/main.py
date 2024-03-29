import os

from fastapi import FastAPI


app = FastAPI(port=os.environ.get('PORT'))


@app.get('/')
def index():
    print("hello world")
    return {'status':'trained', 
            "answer": "good",
            "question": "how are you?",
            "id": 1,
            "title": "title",
            "context": "context",
            "answers": "answers",}
