import os, joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class GBMClassifier:
    def __init__(self, ticker: str, modelo_path=None):
        self.ticker = ticker
        self.modelo_path = modelo_path or f"modelos/gbm_{ticker}.joblib"
        self.model = None

    def treinar(self, X, y):
        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.modelo_path), exist_ok=True)
        joblib.dump(self.model, self.modelo_path)
        return self

    def carregar(self):
        if os.path.exists(self.modelo_path):
            self.model = joblib.load(self.modelo_path)
            return True
        return False

    def prever_prob_up(self, X_ult):
        if self.model is None:
            raise RuntimeError("Modelo GBM não carregado/treinado")
        p_up = float(self.model.predict_proba(X_ult)[0,1])
        return p_up