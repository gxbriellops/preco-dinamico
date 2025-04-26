# Previsão de Preços do Uber e Lyft
![ml app preview](https://github.com/gxbriellops/preco-dinamico/blob/main/Grava%C3%A7%C3%A3o-de-Tela-2025-04-26-115224.gif)
## Visão Geral
Este projeto analisa dados de transporte compartilhado do Uber e Lyft em Boston, MA para construir um modelo de aprendizado de máquina que prevê preços de corridas com base em vários fatores como distância, multiplicador de preço dinâmico, condições climáticas e localização. O modelo alcança 96% de precisão na explicação das variações de preço com um erro médio de apenas $1,82.

## Conjunto de Dados
O conjunto de dados vem do [Kaggle: Uber and Lyft Dataset (Boston, MA)](https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma) e contém informações sobre preços de corridas, detalhes das viagens, condições climáticas e outras variáveis que afetam os preços na área de Boston.

## Características
A análise explora diversos fatores que influenciam os preços das corridas:
- Distância da viagem
- Multiplicador de preço dinâmico
- Tipo de corrida e nível de serviço (UberX, Lyft, etc.)
- Condições climáticas (temperatura, precipitação, etc.)
- Locais de embarque e desembarque
- Hora do dia e outros fatores temporais

## Metodologia
O projeto segue estas etapas principais:
1. **Exploração de Dados**: Análise abrangente da distribuição de preços e correlações de fatores
2. **Engenharia de Características**: Codificação alvo de variáveis categóricas e padronização de dados
3. **Construção do Modelo**: Implementação de um regressor Random Forest
4. **Avaliação do Modelo**: Avaliação de desempenho usando métricas como R², MAE e MAPE
5. **Análise de Importância de Características**: Identificação dos determinantes de preço mais influentes
6. **Análise de Erro**: Exame detalhado da precisão de previsão em diferentes faixas de preço

## Principais Descobertas
- **A distância é o fator dominante** na determinação dos preços das corridas (correlação de 33%)
- **O multiplicador de preço dinâmico** tem impacto significativo nos preços finais (correlação de 16%)
- **O tipo de corrida e nível de serviço** afetam substancialmente os preços
- **As condições climáticas** têm efeitos modestos, mas mensuráveis, nos preços
- O modelo tem desempenho excepcionalmente bom para preços baixos a médios, com margens de erro ligeiramente maiores para corridas premium
- O modelo tende a subestimar ligeiramente corridas com preços mais caros (acima de $60), mas mantém excelente precisão em corridas mais baratas e de valor médio

## Aplicação Web
O projeto inclui uma aplicação web Streamlit que permite aos usuários:
- Inserir detalhes da viagem (origem, destino, tipo de corrida)
- Ver o preço previsto com explicações visuais
- Explorar variações de preço com base em diferentes fatores
- Visualizar a rota da viagem em um mapa interativo

## Arquivos neste Repositório
- `uber_analise.ipynb`: Notebook Jupyter com análise completa de dados e desenvolvimento do modelo
- `app.py`: Aplicação web Streamlit para previsão de preços
- `target.py`: Script utilitário para codificação alvo de variáveis categóricas
- Arquivos do modelo:
  - `modelo_preco_uber.joblib`: Modelo Random Forest salvo
  - `scaler_preco_uber.joblib`: StandardScaler para normalização de características

## Requisitos
- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- folium
- streamlit-folium
- plotly
- joblib
- requests
- dotenv

## Executando a Aplicação
1. Clone este repositório
2. Instale os requisitos: `pip install -r requirements.txt`
3. Execute a aplicação Streamlit: `streamlit run app.py`

## Resultados e Conclusões
O modelo explica 96% da variação nos preços das corridas com um erro médio de apenas $1,82. A análise fornece insights valiosos tanto para passageiros quanto para empresas de transporte compartilhado:
- Os passageiros podem entender melhor os fatores de preço e planejar viagens mais econômicas
- As empresas podem otimizar estratégias de preços dinâmicos com base nos fatores identificados
- O modelo pode ajudar a melhorar a transparência e previsibilidade dos preços

## Trabalho Futuro
- Incorporar dados de tráfego em tempo real para melhorar a estimativa de distância
- Adicionar características baseadas em tempo para capturar melhor os preços de pico/fora de pico
- Estender o modelo para outras cidades para testar a generalização geográfica
- Implementar testes A/B para otimização de estratégia de preços

## Licença
Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para obter detalhes.
