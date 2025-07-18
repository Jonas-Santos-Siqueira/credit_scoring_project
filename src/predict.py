from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import joblib
import sys

class HighCorrFilter(BaseEstimator, TransformerMixin):
    """
    Transformer que remove qualquer feature cuja correlação absoluta com o alvo exceda o limiar (padrão 0.9).
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.drop_cols_ = []

    def fit(self, X, y=None):
        # 1. X: DataFrame com features; y: Series do target 'class':
        df = pd.concat([X, y.rename('class')], axis=1)
        corr = df.corr()['class'].drop('class')
        
        # 2. Seleciona colunas para remoção:
        self.drop_cols_ = corr[abs(corr) > self.threshold].index.tolist()
        # print("Features excluídas:", self.drop_cols_, "\n")
        return self

    def transform(self, X):
        # 3. Remove colunas de alta correlação com o alvo:
        return X.drop(columns=self.drop_cols_, errors='ignore')

def load_model(model_path: str = 'knn_pipeline.joblib'):
    """
    Carrega e retorna o pipeline treinado salvo em disco.
    """
    return joblib.load(model_path)

def predict(df: pd.DataFrame, model_path: str = 'knn_pipeline.joblib') -> pd.DataFrame:
    """
    Recebe um DataFrame com as mesmas colunas originais (todas feats) e retorna a base completa,
    acrescida das colunas:
        - class_pred: previsão da classe (0/1)
        - prob: probabilidade da classe positiva
    """

    # Função para classificar o score:
    def classificar_score(score):
        if score >= 901:
            return "Risco muito baixo"
        elif score >= 701:
            return "Risco baixo"
        elif score >= 501:
            return "Risco médio"
        elif score >= 301:
            return "Risco alto"
        else:
            return "Risco muito alto"

    # Carrega modelo já treinado:
    model = load_model(model_path)

    # Gera predições:
    proba = model.predict_proba(df)[:, 1]
    pred  = model.predict(df)

    # Cria uma cópia para não modificar o DataFrame original e adiciona as colunas de predição:
    df = df.copy()
    df['class_pred'] = pred  
    df['prob'] = np.round(proba.astype(float)*100, 2) # Probabilidade de default.
    
    # Calcula o score de crédito: mapeia a probabilidade (0 a 1) para um intervalo de score (0 a 1000):
    df['score'] = ((1 - proba.astype(float)) * 1000).astype(int)
    df['class_score'] = df['score'].apply(classificar_score)
    return df

if __name__ == '__main__':
    # Espera: python predict.py <data/test_data_for_inference.csv> <output/result_output.csv>
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    data = pd.read_csv(input_path)
    results = predict(data)
    results.to_csv(output_path, index=False)
    print(f"Predições salvas em {output_path}")