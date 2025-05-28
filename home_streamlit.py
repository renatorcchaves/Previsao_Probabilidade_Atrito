import pandas as pd
import streamlit as st

from joblib import load   

from notebooks.src.config import DADOS_TRATADOS, MODELO_FINAL_2

# 1ª ETAPA: Carregar dados e modelo ---------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

@st.cache_data      
def carregar_dados():
    return pd.read_parquet(DADOS_TRATADOS)

def carregar_modelo():
    return load(MODELO_FINAL_2)

df = carregar_dados()
modelo = carregar_modelo()

# 2ª ETAPA: Criando variáveis necessárias conforme as features do dataframe------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------

# Pegando os valores únicos de cada coluna do dataframe relacionadas à texto (mesmo que seja classificações como 1, 2, 3, 4 ou 5)
generos = sorted(df['Gender'].unique())
niveis_educacionais = sorted(df['Education'].unique())
area_formacao = sorted(df["EducationField"].unique())
departamentos = sorted(df["Department"].unique())
viagem_negocios = sorted(df["BusinessTravel"].unique())
hora_extra = sorted(df["OverTime"].unique())
satisfacao_trabalho = sorted(df["JobSatisfaction"].unique())
satisfacao_colegas = sorted(df["RelationshipSatisfaction"].unique())
satisfacao_ambiente = sorted(df["EnvironmentSatisfaction"].unique())
vida_trabalho = sorted(df["WorkLifeBalance"].unique())
opcao_acoes = sorted(df["StockOptionLevel"].unique())
envolvimento_trabalho = sorted(df["JobInvolvement"].unique())

niveis_educacionais_texto = {
    1: "Bellow College",
    2: "College",
    3: "Bachaler",
    4: "Master",
    5: "PhD"
}
niveis_satisfacao_texto = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High", 
}
niveis_vida_trabalho_texto = {
    1: "Bad",
    2: "Good",
    3: "Better",
    4: "Best", 
}

# Colunas numéricas do dataframe 
colunas_slider = [
    "DistanceFromHome",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

colunas_slider_min_max = {                      
    coluna: {"min_value": df[coluna].min(), "max_value": df[coluna].max()}
    for coluna in colunas_slider
}

colunas_ignoradas = (
    "Age",
    "DailyRate",
    "JobLevel",
    "HourlyRate",
    "MonthlyRate",
    "PerformanceRating",
)

mediana_colunas_ignoradas = {                      
    coluna: df[coluna].median() for coluna in colunas_ignoradas
}

# 3ª ETAPA: Criar componentes do site ----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

st.title('Previsão de Attrition')

with st.container(border=True):
    st.write("### Informações Pessoais")
    widget_genero = st.radio("Genero", generos)
    widget_distancia_casa = st.slider("Distância de casa", **colunas_slider_min_max["DistanceFromHome"])
    widget_nivel_educacional = st.selectbox("Nível Educacional", niveis_educacionais, format_func = lambda numero: niveis_educacionais_texto[numero])
    widget_area_formacao = st.selectbox( "Área de Formação", area_formacao)


with st.container(border=True):
    st.write("### Rotina na Empresa")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        widget_departamento = st.selectbox( "Departamento", departamentos)
        widget_viagem_negocios = st.selectbox( "Viagem de Negócios", viagem_negocios)

    with coluna_direita:
        widget_cargo = st.selectbox(
            "Cargo", 
            sorted(df[df['Department'] == widget_departamento]['JobRole'].unique()))
        widget_horas_extras = st.radio("Horas Extras", hora_extra)

    widget_salario_mensal = st.slider("Salário Mensal", **colunas_slider_min_max['MonthlyIncome']) 

with st.container(border=True):
    st.write("### Experiência Profissional")

    coluna_esquerda, coluna_meio, coluna_direita = st.columns(3)

    with coluna_esquerda:
        widget_empresas_trabalhadas = st.slider("Empresas Trabalhadas", **colunas_slider_min_max["NumCompaniesWorked"])
        widget_anos_trabalhados = st.slider("Anos Trabalhados", **colunas_slider_min_max["TotalWorkingYears"])

    with coluna_meio:
        widget_anos_empresa = st.slider("Anos na Empresa", **colunas_slider_min_max["YearsAtCompany"])
        widget_anos_cargo_atual = st.slider("Anos no Cargo Atual", **colunas_slider_min_max["YearsInCurrentRole"]) 

    with coluna_direita:
        widget_anos_mesmo_gerente = st.slider("Anos com Mesmo Gerente", **colunas_slider_min_max["YearsWithCurrManager"])
        widget_anos_ultima_promocao = st.slider("Anos Desde a Última Promoção", **colunas_slider_min_max["YearsSinceLastPromotion"])

with st.container(border=True):
    st.write("### Incentivos e Métricas")

    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        widget_satisfacao_trabalho = st.selectbox("Satisfação no Trabalho", satisfacao_trabalho, format_func= lambda numero: niveis_satisfacao_texto[numero])
        widget_satisfacao_colegas = st.selectbox("Satisfação com Colegas", satisfacao_colegas, format_func= lambda numero: niveis_satisfacao_texto[numero])
        widget_envolvimento_trabalho = st.selectbox("Envolvimento no Trabalho", envolvimento_trabalho, format_func= lambda numero: niveis_satisfacao_texto[numero])

    with coluna_direita:
        widget_satisfacao_ambiente = st.selectbox("Satisfação com Ambiente", satisfacao_ambiente, format_func= lambda numero: niveis_satisfacao_texto[numero])
        widget_satisfacao_vida_trabalho = st.selectbox("Balanço Vida-Trabalho", vida_trabalho, format_func= lambda numero: niveis_vida_trabalho_texto[numero])
        widget_opcao_acoes = st.radio("Opção de Ações", opcao_acoes)
    
    widget_aumento_salarial = st.slider("Aumento Salarial (%)", **colunas_slider_min_max['PercentSalaryHike'])
    widget_treinamentos_ultimo_ano = st.slider("Treinamentos no Último Ano", **colunas_slider_min_max['TrainingTimesLastYear'])


# 4ª ETAPA: Reunindo dados inputados no Streamlit para fazer Previsão do Modeo -----------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------

entrada_modelo = {
    'Gender': widget_genero, 
    'Education': widget_nivel_educacional, 
    'EducationField': widget_area_formacao, 
    'Department': widget_departamento, 
    'BusinessTravel': widget_viagem_negocios, 
    'OverTime': widget_horas_extras, 
    'JobSatisfaction': widget_satisfacao_trabalho, 
    'RelationshipSatisfaction': widget_satisfacao_colegas, 
    'EnvironmentSatisfaction': widget_satisfacao_ambiente, 
    'WorkLifeBalance': widget_satisfacao_vida_trabalho, 
    'StockOptionLevel': widget_opcao_acoes, 
    'JobInvolvement': widget_envolvimento_trabalho, 
    'JobRole': widget_cargo,
    'DistanceFromHome': widget_distancia_casa, 
    'MonthlyIncome': widget_salario_mensal, 
    'NumCompaniesWorked': widget_empresas_trabalhadas, 
    'PercentSalaryHike': widget_aumento_salarial, 
    'TotalWorkingYears': widget_anos_trabalhados, 
    'TrainingTimesLastYear': widget_treinamentos_ultimo_ano, 
    'YearsAtCompany': widget_anos_empresa, 
    'YearsInCurrentRole': widget_anos_cargo_atual, 
    'YearsSinceLastPromotion': widget_anos_ultima_promocao, 
    'YearsWithCurrManager': widget_anos_mesmo_gerente, 
    'Age': mediana_colunas_ignoradas['Age'], 
    'DailyRate': mediana_colunas_ignoradas['DailyRate'], 
    'JobLevel': mediana_colunas_ignoradas['JobLevel'], 
    'HourlyRate': mediana_colunas_ignoradas['HourlyRate'],
    'MonthlyRate': mediana_colunas_ignoradas['MonthlyRate'], 
    'PerformanceRating': mediana_colunas_ignoradas['PerformanceRating'], 
    'MaritalStatus': 'Single'    # é um valor que não influencia no modelo, e não é numérico para calcularmos a mediana
}

df_entrada_modelo = pd.DataFrame([entrada_modelo])

botao_previsao = st.button("Prever probabilidade de Attrition")

if botao_previsao:      
    modelo = load(MODELO_FINAL_2)
    previsao = modelo.predict(df_entrada_modelo)[0]
    probabilidade_attrition = modelo.predict_proba(df_entrada_modelo)[0][1]

    if previsao == 1:
        cor = ':red' 
    else:
        cor=':green'

    texto_probabilidade = (f"#### Probabilidade de Attrition: {cor}[{probabilidade_attrition:.0%}]")
    texto_attrition = (f"#### Attrition: {cor}[{'Sim' if previsao ==1 else 'Não'}]")

    st.markdown(texto_attrition)
    st.markdown(texto_probabilidade)
