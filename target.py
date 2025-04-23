import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# Este script demonstra como criar e salvar target encoders
# Você precisaria executá-lo com seus dados reais de treinamento

# Função para criar e salvar target encoders
def criar_target_encoders(df, categorias, target='price'):
    """
    Cria target encoders para variáveis categóricas.
    
    Args:
        df: DataFrame com os dados
        categorias: Lista de colunas categóricas para codificar
        target: Nome da coluna alvo
    
    Returns:
        Dicionário com os target encoders
    """
    # Dividir os dados em treino e teste
    # Usamos apenas os dados de treino para calcular os encoders
    # para evitar vazamento de dados
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=[target]),
        df[target],
        test_size=0.2,
        random_state=42
    )
    
    # Dicionário para armazenar os encoders
    encoders = {}
    
    # Dicionário para armazenar os mapeamentos
    mapeamentos = {}
    
    # Para cada categoria, criar um target encoder
    for categoria in categorias:
        # Criar o encoder
        encoder = TargetEncoder(cols=[categoria])
        
        # Ajustar o encoder aos dados de treino
        encoder.fit(X_train[categoria], y_train)
        
        # Armazenar o encoder
        encoders[categoria] = encoder
        
        # Criar um dicionário com os mapeamentos
        valores_unicos = df[categoria].unique()
        mapeamento = {}
        
        for valor in valores_unicos:
            # Transformar o valor usando o encoder
            encoded_val = encoder.transform(pd.DataFrame({categoria: [valor]}))[categoria].iloc[0]
            mapeamento[valor] = encoded_val
        
        # Armazenar o mapeamento
        mapeamentos[categoria] = mapeamento
    
    # Salvar os mapeamentos em um arquivo pickle
    with open('target_encoders.pkl', 'wb') as f:
        pickle.dump(mapeamentos, f)
    
    return mapeamentos

caminho = r"C:\Users\Gabriel Lopes\Documents\PROJETOS_PROGRAMAÇÃO\TABELA DE PRECO DINAMICO\preco-dinamico\ipynb\csv\rideshare_kaggle.csv"

# Exemplo de uso (você precisaria adaptar para seus dados reais)
if __name__ == "__main__":
    # Carregar os dados (exemplo simulado)
    # No seu caso, carregue seus dados reais aqui
    df = pd.read_csv(caminho)

    df['price'] = df['price'].fillna(df['price'].median())
    
    # Lista de colunas categóricas a serem codificadas
    categorias = ['cab_type', 'source', 'destination', 'name', 'short_summary', 'long_summary']
    
    # Criar e salvar os target encoders
    mapeamentos = criar_target_encoders(df, categorias, target='price')
    
    print("Target encoders criados e salvos com sucesso!")
    
    # Exibir exemplos de mapeamentos
    for categoria, mapeamento in mapeamentos.items():
        print(f"\nMapeamento para {categoria}:")
        for valor, encoded in list(mapeamento.items())[:3]:  # Mostrar apenas os 3 primeiros
            print(f"  {valor} -> {encoded:.2f}")