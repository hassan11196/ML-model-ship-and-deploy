import joblib
import numpy as np

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
import random
class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()
        self.boston = load_boston()
        
        self.bos = pd.DataFrame(self.boston.data, columns = self.boston.feature_names)
        self.bos['PRICE'] = self.boston.target

    def train(self, X: np.ndarray, y: np.ndarray):
                
        self._model = RandomForestRegressor()
        self._model.fit(X, y)
        return self

    def auto_train(self):
        X_rooms = self.bos.drop('PRICE', axis = 1)
        y_price = self.bos.PRICE

        # X_rooms = np.array(X_rooms).reshape(-1,1)
        #y_price = np.array(y_price).reshape(-1,1)
        X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size = 0.2, random_state=5)        
        self._model = RandomForestRegressor()
        self._model.fit(X_train_1, Y_train_1)

        y_pred_1 = self._model.predict(X_test_1)
        rmse = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
        r2 = round(self._model.score(X_test_1, Y_test_1),2)
        random_test_row = random.randint(0, X_test_1.shape[0]) % X_test_1.shape[0]
        return {'Root Mean Squared Error : ': rmse, 'R^2':r2, 'test':X_test_1.iloc[random_test_row].values.tolist(), 'ans_test':Y_test_1.iloc[random_test_row]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            self._model = None
        return self


model_path = Path(__file__).parent / "model.joblib"
n_features = load_boston(return_X_y=True)[0].shape[1]
model = Model(model_path)


def get_model():
    return model


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    model.train(X, y)
    model.save()