import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Carregar o modelo e o scaler salvos
@st.cache_resource
def carregar_modelo():
    modelo = joblib.load('modelo_preco_uber.joblib')
    scaler = joblib.load('scaler_preco_uber.joblib')
    return modelo, scaler

modelo, scaler = carregar_modelo()

# Título da aplicação
st.title("🚗 Previsão de Preços de Uber")
st.write("Este aplicativo estima o preço de uma corrida de Uber com base em diferentes fatores.")

# Interface para entrada de dados
st.header("Dados da Corrida")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distância (em milhas)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
    surge_multiplier = st.number_input("Multiplicador de preço dinâmico", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    latitude = st.number_input("Latitude", min_value=30.0, max_value=50.0, value=40.0, step=0.01)
    
with col2:
    temperature = st.number_input("Temperatura (°F)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    pressure = st.number_input("Pressão atmosférica", min_value=900.0, max_value=1100.0, value=1000.0, step=1.0)
    
# Para as variáveis categóricas, criamos selectboxes com valores pré-definidos
st.header("Informações do Serviço")

col3, col4 = st.columns(2)

with col3:
    cab_type = st.selectbox("Tipo de serviço", options=["UberX", "UberXL", "UberBLACK", "Lyft", "Lyft XL", "Lyft Lux"])
    source = st.selectbox("Origem", options=["Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District", "Northeastern University"])

with col4:
    destination = st.selectbox("Destino", options=["Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District", "Northeastern University"])
    name = st.selectbox("Nome do serviço", options=["UberPool", "UberX", "UberXL", "Lyft", "Lyft XL", "Lux"])

# Simplificação para variáveis que usariam target encoding
# No app real, precisaríamos importar os dicionários de mapeamento salvos durante o treinamento
# Aqui estamos usando valores médios simulados
cab_type_map = {"UberX": 15.0, "UberXL": 22.0, "UberBLACK": 35.0, "Lyft": 14.0, "Lyft XL": 21.0, "Lyft Lux": 33.0}
source_map = {"Back Bay": 18.0, "Beacon Hill": 20.0, "Boston University": 15.0, "Fenway": 17.0, "Financial District": 25.0, "Northeastern University": 16.0}
destination_map = {"Back Bay": 18.0, "Beacon Hill": 20.0, "Boston University": 15.0, "Fenway": 17.0, "Financial District": 25.0, "Northeastern University": 16.0}
name_map = {"UberPool": 12.0, "UberX": 15.0, "UberXL": 22.0, "Lyft": 14.0, "Lyft XL": 21.0, "Lux": 33.0}

# Botão para realizar a previsão
if st.button("Calcular Preço Estimado"):
    # Criando o DataFrame com os dados de entrada
    dados_entrada = pd.DataFrame({
        'distance': [distance],
        'surge_multiplier': [surge_multiplier],
        'latitude': [latitude],
        'apparentTemperatureLow': [temperature - 10],  # Simplificação
        'pressure': [pressure],
        'temperatureHigh': [temperature + 5],  # Simplificação
        'source': [source_map[source]],
        'destination': [destination_map[destination]],
        'cab_type': [cab_type_map[cab_type]],
        'name': [name_map[name]],
        'long_summary': [20.0],  # Valores médios simulados
        'short_summary': [20.0]   # Valores médios simulados
    })
    
    # Normalizar os dados de entrada
    dados_entrada_scaled = scaler.transform(dados_entrada)
    
    # Realizar previsão
    preco_previsto = modelo.predict(dados_entrada_scaled)[0]
    
    # Exibir resultado
    st.success(f"💰 Preço estimado da corrida: ${preco_previsto:.2f}")
    
    # Adicionar insights sobre os fatores que mais impactaram o preço
    st.subheader("Análise da previsão")
    st.write(f"A distância de {distance} milhas é o principal fator nesta estimativa.")
    if surge_multiplier > 1.0:
        st.write(f"O multiplicador de preço dinâmico ({surge_multiplier}x) aumentou significativamente o preço estimado.")
    
    # Histograma de preços similares (simulado)
    st.subheader("Distribuição de preços para corridas similares")
    precos_similares = np.random.normal(preco_previsto, 3, 100)
    precos_similares = precos_similares[precos_similares > 0]
    st.bar_chart(pd.DataFrame(precos_similares).rename(columns={0: "Preço USD"}))

# Informações adicionais
st.sidebar.header("Sobre o modelo")
st.sidebar.write("""
Este modelo foi treinado com dados do Kaggle sobre corridas de Uber e Lyft.
O modelo tem precisão de 96% (R²) e erro médio de $1.82 por corrida.

Principais fatores que influenciam o preço:
- Distância da corrida
- Multiplicador de preço dinâmico
- Tipo de serviço
- Origem e destino
""")

st.sidebar.info("💡 Esta aplicação é apenas para fins demonstrativos.")