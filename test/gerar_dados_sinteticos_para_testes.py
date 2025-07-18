import numpy as np
import pandas as pd

# Função para gerar dados sintéticos com mesma estrutura e distribuição do dataset original
def generate_synthetic_data(df: pd.DataFrame, n_samples: int, random_state: int = None) -> pd.DataFrame:
    """
    Gera um DataFrame sintético baseado nas médias, desvios e proporções de classe do DataFrame original.

    Parâmetros:
    df (pd.DataFrame): DataFrame original contendo colunas feat_1...feat_51 e 'class'.
    n_samples (int): Número de amostras sintéticas a gerar.
    random_state (int, opcional): Semente para reprodutibilidade.
    
    Retorna:
    pd.DataFrame: DataFrame com n_samples e mesmas colunas numéricas e de classe.
    """
    rng = np.random.default_rng(random_state)
    
    # Seleciona apenas as colunas de features numéricas
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    # Estima parâmetros de distribuição
    means = df[feature_cols].mean()
    stds = df[feature_cols].std()
    # class_probs = df['class'].value_counts(normalize=True).to_dict()

    # Gera amostras independentes para cada feature
    synthetic_data = {
        col: rng.normal(loc=means[col], scale=stds[col], size=n_samples)
        for col in feature_cols
    }
    
    # Gera rótulos de classe com as mesmas proporções
    # synthetic_data['class'] = rng.choice(
    #     list(class_probs.keys()), size=n_samples, p=list(class_probs.values())
    # )
    return pd.DataFrame(synthetic_data).to_csv('test_data_for_inference.csv', index=False)


if __name__ == "__main__":
    orig_df = pd.read_csv('data/dados.csv').drop(columns=['Unnamed: 0'])
    syn_df = generate_synthetic_data(orig_df, n_samples=100, random_state=42)
    print("Dados sintéticos gerados e salvos em 'test_data_for_inference.csv'.")