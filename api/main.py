import numpy as np
import pandas as pd
import os
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# from .ml.model import Model, get_model, n_features



app = FastAPI(port=os.environ.get('PORT'))


@app.get('/')
def index():
    
    return {'status':'trained'}
