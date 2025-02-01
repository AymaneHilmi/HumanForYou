# Import des bibliothèques principales
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# 📌 CONFIGURATION DE L'INTERFACE
st.set_page_config(page_title="Analyse RH", layout="wide")

# 📌 CHARGEMENT DES DONNÉES
@st.cache_data
def load_data():
    # Chargement des données RH
    hr_data = pd.read_csv('./data/general_data.csv')
    survey_data = pd.read_csv('./data/employee_survey_data.csv')
    manager_data = pd.read_csv('./data/manager_survey_data.csv')
    
    # Fusion des datasets
    hr_data = hr_data.merge(survey_data, on='EmployeeID')
    hr_data = hr_data.merge(manager_data, on='EmployeeID')
    
    # Nettoyage et conversions
    hr_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)
    hr_data.dropna(subset=['NumCompaniesWorked', 'TotalWorkingYears'], inplace=True)

    # Remplacement des valeurs manquantes
    for col in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
        hr_data[col].fillna(hr_data[col].median(), inplace=True)

    # Transformation des variables catégoriques
    hr_data['Age'] = hr_data['Age'].astype(int)
    hr_data['Attrition'] = hr_data['Attrition'].map({'Yes': 1, 'No': 0})
    hr_data['BusinessTravel'] = hr_data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    hr_data['DistanceFromHome'] = hr_data['DistanceFromHome'].astype(int)
    hr_data['Education'] = hr_data['Education'].astype('category')
    hr_data['EducationField'] = hr_data['EducationField'].astype('category')
    hr_data['EmployeeID'] = hr_data['EmployeeID'].astype(int)
    hr_data['Gender'] = hr_data['Gender'].map({'Male': 1, 'Female': 0})
    hr_data['JobLevel'] = hr_data['JobLevel'].astype(int)
    hr_data['JobRole'] = hr_data['JobRole'].astype('category')
    hr_data['MaritalStatus'] = hr_data['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    hr_data['MonthlyIncome'] = hr_data['MonthlyIncome'].astype(float)
    hr_data['NumCompaniesWorked'] = hr_data['NumCompaniesWorked'].astype(int)
    hr_data['PercentSalaryHike'] = hr_data['PercentSalaryHike'].astype(float) / 100
    hr_data['StockOptionLevel'] = hr_data['StockOptionLevel'].astype(int)
    hr_data['TotalWorkingYears'] = hr_data['TotalWorkingYears'].astype(int)
    hr_data['TrainingTimesLastYear'] = hr_data['TrainingTimesLastYear'].astype(int)
    hr_data['YearsAtCompany'] = hr_data['YearsAtCompany'].astype(int)
    hr_data['YearsSinceLastPromotion'] = hr_data['YearsSinceLastPromotion'].astype(int)
    hr_data['YearsWithCurrManager'] = hr_data['YearsWithCurrManager'].astype(int)
    hr_data['JobInvolvement'] = hr_data['JobInvolvement'].astype('category')
    hr_data['PerformanceRating'] = hr_data['PerformanceRating'].astype('category')
    hr_data['EnvironmentSatisfaction'] = hr_data['EnvironmentSatisfaction'].astype('category')
    hr_data['WorkLifeBalance'] = hr_data['WorkLifeBalance'].astype('category')

    # Chargement des données d'absentéisme
    in_time_data = pd.read_csv('./data/in_time.csv')
    out_time_data = pd.read_csv('./data/out_time.csv')
    in_time_data.rename(columns={"Unnamed: 0": 'EmployeeID'}, inplace=True)
    out_time_data.rename(columns={"Unnamed: 0": 'EmployeeID'}, inplace=True)

    # Calcul des jours d'absence
    absence_status = (in_time_data.iloc[:, 1:].isna() | out_time_data.iloc[:, 1:].isna())
    absence_status = absence_status.dropna(axis=0, how='all')
    absence_status = absence_status.replace({True: 'Absent', False: 'Present'})
    absence_status.insert(0, 'EmployeeID', in_time_data['EmployeeID'])
    
    # Comptage des jours d'absence
    absence_days = absence_status.iloc[:, 1:].apply(lambda x: (x == 'Absent').sum(), axis=1)
    absence_days = pd.DataFrame({'EmployeeID': absence_status['EmployeeID'], 'AbsenceDays': absence_days})
    
    hr_data = hr_data.merge(absence_days, on='EmployeeID', how='left')  # Ajouter le nombre de jours d'absence

    return hr_data, absence_status, absence_days

# Charger les données
df, absence_status, absence_days = load_data()

# 📌 SIDEBAR INTERACTIVE
st.sidebar.header("🔎 Options d'analyse")
selected_features = st.sidebar.multiselect("Sélectionnez les variables à afficher dans la matrice de corrélation :", 
                                           df.select_dtypes(include=['int64', 'float64']).columns.tolist(), 
                                           default=['Age', 'Attrition', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction'])

st.sidebar.write("💡 Astuce : Sélectionnez des variables pertinentes pour une meilleure lecture.")

# 📌 TITRE PRINCIPAL
st.title("📊 Analyse des Données")

# 📌 STATISTIQUES GÉNÉRALES
st.subheader("📌 Statistiques Clés")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🌍 Nombre total d'employés", df.shape[0])
    st.metric("🚀 Taux d'attrition", f"{df['Attrition'].mean() * 100:.2f} %")
    
with col2:
    st.metric("📈 Salaire moyen", f"${df['MonthlyIncome'].mean():,.2f}")
    st.metric("📅 Ancienneté moyenne", f"{df['YearsAtCompany'].mean():.1f} ans")
    
with col3:
    st.metric("👨‍💼 % Hommes", f"{df[df['Gender'] == 1].shape[0] / df.shape[0] * 100:.1f} %")
    st.metric("👩 % Femmes", f"{df[df['Gender'] == 0].shape[0] / df.shape[0] * 100:.1f} %")

# 📌 STATISTIQUES D'ABSENTÉISME
st.subheader("📌 Statistiques d'Absentéisme")
col1, col2 = st.columns(2)

with col1:
    st.metric("📊 Absence moyenne par employé", f"{absence_days['AbsenceDays'].mean():.1f} jours")

with col2:
    max_absences_employee = absence_days.loc[absence_days['AbsenceDays'].idxmax()]
    st.metric("👥 Employé avec le plus d'absences", f"ID :{max_absences_employee['EmployeeID']} avec {max_absences_employee['AbsenceDays']} jours")

# 📌 ONGLETS INTERACTIFS 
tab1, tab2, tab3 = st.tabs(["📈 Statistiques détaillées", "📊 Graphiques", "📁 Données brutes"])

with tab1:
    st.subheader("📌 Détails des statistiques par variable")
    st.dataframe(df.describe())

    st.subheader("📊 Répartition des valeurs catégoriques")
    for col in df.select_dtypes(include=['category']).columns:
        st.write(f"### {col}")
        st.write(df[col].value_counts())
        st.bar_chart(df[col].value_counts())

with tab2:
    st.subheader("📊 Distribution des âges")
    st.write("📈 Répartition des âges des employés"
             "\n🔴 18 - 25 ans, 🔵 26 - 35 ans, 🟢 36 - 45 ans, 🟡 46 - 55 ans, 🟣 56 - 65 ans")
    age_bins = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], precision=0, right=False)
    age_bins_str = age_bins.astype(str)
    age_distribution = age_bins_str.value_counts().sort_index()
    age_distribution.index = age_distribution.index.str.replace('[', '').str.replace(')', '').str.replace(',', ' -')
    st.bar_chart(age_distribution)


    # 📌 RÉPARTITION DES SALAIRES PAR TRANCHE
    st.subheader("💰 Répartition des salaires par tranche")
    salary_bins = pd.cut(df['MonthlyIncome'], bins=5, precision=0)
    salary_bins_str = salary_bins.astype(str)
    salary_distribution = salary_bins_str.value_counts().sort_index()
    salary_distribution.index = salary_distribution.index.str.replace('(', '').str.replace(']', '').str.replace(',', ' -')
    st.bar_chart(salary_distribution)

    st.subheader("📈 Répartition des années d'ancienneté par YearsAtCompany")
    # axe x : nombre d'années, axe y : nombre d'employés
    st.bar_chart(df['YearsAtCompany'].value_counts())


    st.subheader("📊 Répartition des niveaux de satisfaction"
                 "\n🔴 0 : Bas, 🔵 4 : Haut")
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
    for col in satisfaction_cols:
        st.write(f"### {col}")
        st.bar_chart(df[col].value_counts())

with tab3:
    st.subheader("📂 Aperçu des données")
    st.dataframe(df.head(20))

# 📌 MATRICE DE CORRÉLATION
st.subheader("📌 Matrice de Corrélation")

# Filtrer les données selon les variables sélectionnées
correlation_matrix = df[selected_features].corr()

# Affichage de la heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# 📌 TABLEAU DES CORRÉLATIONS
st.subheader("📊 Tableau des Corrélations")
st.write(correlation_matrix)

# 📌 ANALYSE DES DÉPARTS
st.subheader("📉 Analyse des employés ayant quitté l'entreprise")
col1, col2 = st.columns(2)

with col1:
    st.write("📌 **Moyenne d'âge des employés ayant quitté :**")
    st.write(f"➡️ {df[df['Attrition'] == 1]['Age'].mean():.1f} ans")

    st.write("📌 **Salaire moyen des employés ayant quitté :**")
    st.write(f"➡️ ${df[df['Attrition'] == 1]['MonthlyIncome'].mean():,.2f}")

with col2:
    st.write("📌 **Nombre moyen d'années dans l'entreprise avant de partir :**")
    st.write(f"➡️ {df[df['Attrition'] == 1]['YearsAtCompany'].mean():.1f} ans")

    st.write("📌 **Niveau moyen de satisfaction des employés ayant quitté :**")
    st.write(f"➡️ {df[df['Attrition'] == 1]['JobSatisfaction'].mean():.1f} / 4")

# Définition des tranches d'âge
def age_category(age):
    if age < 30:
        return "18-30 ans"
    elif age <= 45:
        return "30-45 ans"
    else:
        return "45+ ans"

# Appliquer la fonction aux données
df["AgeGroup"] = df["Age"].apply(age_category)

# Calcul du taux d'attrition par tranche d'âge
age_attrition = df.groupby("AgeGroup")["Attrition"].mean() * 100

# Afficher les résultats
print("Taux d'attrition par tranche d'âge (%)")
print(age_attrition)

# Visualisation avec un graphique à barres
plt.figure(figsize=(8,5))
sns.barplot(x=age_attrition.index, y=age_attrition.values, palette="coolwarm")
plt.xlabel("Tranche d'âge")
plt.ylabel("Taux d'attrition (%)")
plt.title("Taux d'attrition par tranche d'âge")
plt.show()

import numpy as np
from scipy.interpolate import make_interp_spline

# 📌 ANALYSE DES DÉPARTS PAR GROUPE DÉMOGRAPHIQUE
st.subheader("📊 Analyse de l'attrition par groupe démographique")

# Appliquer les transformations aux colonnes nécessaires
df["Gender"] = df["Gender"].map({1: "Homme", 0: "Femme"})
df["MaritalStatus"] = df["MaritalStatus"].map({0: "Célibataire", 1: "Marié", 2: "Divorcé"})

# Calcul des taux d'attrition
age_attrition = df.groupby("Age")["Attrition"].mean() * 100
gender_attrition = df.groupby("Gender")["Attrition"].mean() * 100
marital_attrition = df.groupby("MaritalStatus")["Attrition"].mean() * 100

# Sélection du graphique à afficher
option = st.selectbox("Choisissez l'analyse à afficher :", ["📈 Taux d'attrition par âge", "📊 Taux d'attrition par genre", "📉 Taux d'attrition par état matrimonial"])

# Fonction pour tracer un graphique avec lissage
def plot_line_chart(data, xlabel, title):
    fig, ax = plt.subplots(figsize=(10,6))

    # Lissage avec une moyenne mobile
    data_sorted = data.sort_index()
    smoothed_data = data_sorted.rolling(window=3, min_periods=1).mean()

    sns.lineplot(x=data_sorted.index, y=smoothed_data, marker="o", linestyle="-", color="b", ax=ax)

    # Personnalisation
    plt.xlabel(xlabel)
    plt.ylabel("Taux d'attrition (%)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Affichage dans Streamlit
    st.pyplot(fig)

# Affichage du graphique en fonction de la sélection
if option == "📈 Taux d'attrition par âge":
    plot_line_chart(age_attrition, "Âge", "Évolution du taux d'attrition par âge")
elif option == "📊 Taux d'attrition par genre":
    plot_line_chart(gender_attrition, "Genre", "Taux d'attrition par genre")
elif option == "📉 Taux d'attrition par état matrimonial":
    plot_line_chart(marital_attrition, "État matrimonial", "Taux d'attrition par état matrimonial")
