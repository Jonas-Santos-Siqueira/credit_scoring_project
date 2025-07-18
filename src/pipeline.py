import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold,
                                    cross_validate)
from sklearn.metrics import (accuracy_score, precision_score,
                                balanced_accuracy_score, recall_score,
                                f1_score, roc_auc_score)

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


def build_pipeline(corr_threshold=0.9):
    """
    Cria um pipeline com seleção de features e classificação.
    
    Parâmetros:
        • corr_threshold: Limiar de correlação para remover features altamente correlacionadas (padrão 0.9).

    Retorna:
        • pipeline: Pipeline com os passos de filtragem, seleção de features e classificação.
    """
    
    corr_filter = HighCorrFilter(threshold=corr_threshold)
    
    pipeline = Pipeline([
    ('corr_filter', corr_filter),
    ('feature_selection', SelectFromModel(
        estimator=RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        threshold="mean")),
    ('classifier', CalibratedClassifierCV(
            estimator=KNeighborsClassifier(
            leaf_size=20, metric='minkowski', n_neighbors=10, p=1, weights='distance'
        ),
        method='isotonic',
        cv=10
    ))
])
    return pipeline

def train_and_export(df_path: str, model_path: str):
    """
    Treina o modelo e exporta o pipeline ajustado.
    
    Parâmetros:
        • df_path: Caminho para o arquivo CSV com os dados para o treinamento;
        • model_path: Caminho para salvar o pipeline ajustado.
    """
    # 1. Carrega dados:
    df = pd.read_csv(df_path).drop(columns=['Unnamed: 0'])
    X = df.drop(columns=['class'])
    y = df['class']

    # 2. Split: 80% treino | 10% validação | 10% teste:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, stratify = y, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size = 0.5, stratify = y_test, random_state = 42)

    # 3. Constrói pipeline:
    pipe = build_pipeline()

    # 4. Treina com X_train:
    pipe.fit(X_train, y_train)

    # 5. Avaliação final via cross_validate:
    cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    results = cross_validate(
                            pipe,
                            X_train, y_train,
                            cv=cv_outer,
                            scoring=['precision', 'accuracy', 'balanced_accuracy','recall', 'f1', 'roc_auc'],
                            n_jobs=-1
                        )

    y_pred_train = pipe.predict(X_train)
    y_proba_train = pipe.predict_proba(X_train)[:, 1]

    y_pred_val = pipe.predict(X_val)
    y_proba_val = pipe.predict_proba(X_val)[:, 1]

    y_pred_test = pipe.predict(X_test)
    y_proba_test = pipe.predict_proba(X_test)[:, 1]

    mask = pipe.named_steps['feature_selection'].get_support()
    selected_features = X_train.drop(columns=pipe.named_steps['corr_filter'].drop_cols_, errors='ignore').columns[mask]

    # 6. Resultados:
    print("Features selecionadas:", "\033[93m" + str(list(selected_features)) + "\033[0m \n")
    
    print("\nResultados da validação cruzada para o pipeline:")
    for metric in results:
        if metric.startswith('test_'):
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            print(f"  • {metric}: valor médio = \033[92m{mean_val*100:.2f}%\033[0m, desvio padrão = \033[92m{std_val*100:.4f}\033[0m")

    print("\nMétricas do Pipeline para o conjunto de treinamento e validação:")
    print("Treinamento - Acurácia: \033[93m{:.2f}%\033[0m".format(accuracy_score(y_train, y_pred_train) * 100))
    print("Treinamento - Precisão: \033[93m{:.2f}%\033[0m".format(precision_score(y_train, y_pred_train, average='weighted') * 100))
    print("Treinamento - Acurácia Balanceada: \033[93m{:.2f}%\033[0m".format(balanced_accuracy_score(y_train, y_pred_train) * 100))
    print("Treinamento - Recall: \033[93m{:.2f}%\033[0m".format(recall_score(y_train, y_pred_train, average='weighted') * 100))
    print("Treinamento - F1 Score: \033[93m{:.2f}%\033[0m".format(f1_score(y_train, y_pred_train, average='weighted') * 100))
    print("Treinamento - ROC-AUC: \033[92m{:.2f}%\033[0m".format(roc_auc_score(y_train, y_proba_train) * 100), "\n")

    print("Validação - Acurácia: \033[92m{:.2f}%\033[0m".format(accuracy_score(y_val, y_pred_val) * 100))
    print("Validação - Precisão: \033[92m{:.2f}%\033[0m".format(precision_score(y_val, y_pred_val, average='weighted') * 100))
    print("Validação - Acurácia Balanceada: \033[92m{:.2f}%\033[0m".format(balanced_accuracy_score(y_val, y_pred_val) * 100))
    print("Validação - Recall: \033[92m{:.2f}%\033[0m".format(recall_score(y_val, y_pred_val, average='weighted') * 100))
    print("Validação - F1 Score: \033[92m{:.2f}%\033[0m".format(f1_score(y_val, y_pred_val, average='weighted') * 100))
    print("Validação - ROC-AUC: \033[92m{:.2f}%\033[0m".format(roc_auc_score(y_val, y_proba_val) * 100), "\n")

    print("Métricas do Pipeline para o conjunto de teste:")
    print("Teste - Acurácia: \033[92m{:.2f}%\033[0m".format(accuracy_score(y_test, y_pred_test) * 100))
    print("Teste - Precisão: \033[92m{:.2f}%\033[0m".format(precision_score(y_test, y_pred_test, average='weighted') * 100))
    print("Teste - Acurácia Balanceada: \033[92m{:.2f}%\033[0m".format(balanced_accuracy_score(y_test, y_pred_test) * 100))
    print("Teste - Recall: \033[92m{:.2f}%\033[0m".format(recall_score(y_test, y_pred_test, average='weighted') * 100))
    print("Teste - F1 Score: \033[92m{:.2f}%\033[0m".format(f1_score(y_test, y_pred_test, average='weighted') * 100))
    print("Teste - ROC-AUC: \033[92m{:.2f}%\033[0m".format(roc_auc_score(y_test, y_proba_test) * 100))

    # 7. Exporta o pipeline já ajustado:
    joblib.dump(pipe, model_path)
    print(f"\nPipeline salvo em {model_path}")