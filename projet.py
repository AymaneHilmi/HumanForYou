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
    hr_data['JobInvolvement'] = hr_data['JobInvolvement'].astype(int)
    hr_data['PerformanceRating'] = hr_data['PerformanceRating'].astype(int)
    hr_data['EnvironmentSatisfaction'] = hr_data['EnvironmentSatisfaction'].astype(int)
    hr_data['WorkLifeBalance'] = hr_data['WorkLifeBalance'].astype(int)

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

    hr_data["CareerGrowthRate"] = hr_data["JobLevel"] / (hr_data["TotalWorkingYears"] + 1)
    hr_data["PromotionRate"] = hr_data["YearsSinceLastPromotion"] / (hr_data["YearsAtCompany"] + 1)
    hr_data["ManagerChangeRate"] = hr_data["YearsAtCompany"] / (hr_data["YearsWithCurrManager"] + 1)
    hr_data["SatisfactionScore"] = (hr_data["JobSatisfaction"] + hr_data["EnvironmentSatisfaction"] + hr_data["WorkLifeBalance"]) / 3
    hr_data["SalarySatisfactionGap"] = hr_data["MonthlyIncome"] / (hr_data["JobSatisfaction"] + 1)
    hr_data["PerformanceInvolvementGap"] = hr_data["PerformanceRating"] - hr_data["JobInvolvement"]
    hr_data["AbsenceRate"] = hr_data["AbsenceDays"] / (hr_data["YearsAtCompany"] + 1)
    hr_data["TravelFatigue"] = hr_data["BusinessTravel"] * hr_data["DistanceFromHome"]

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
tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistiques détaillées", "📊 Graphiques", "📁 Données brutes", "📌 Indicateurs de Performance"])

with tab1:
    st.subheader("📌 Détails des statistiques par variable")
    st.dataframe(df.describe())

    st.subheader("📌 Répartition des employés par département"
                 )
    st.write(df['Department'].value_counts())


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

    st.subheader("📈 Répartition des années d'ancienneté")
    # axe x : nombre d'années, axe y : nombre d'employés
    st.bar_chart(df['YearsAtCompany'].value_counts())
    satisfaction_mapping = {
        'EnvironmentSatisfaction': 'Satisfaction de l\'environnement de travail',
        'JobSatisfaction': 'Satisfaction du travail',
        'WorkLifeBalance': 'Équilibre travail-vie personnelle'
    }
    st.subheader("📊 Répartition des niveaux de satisfaction"
                 "\n🔴 0 : Bas, 🔵 4 : Haut")
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
    for col in satisfaction_cols:
        st.write(f"### {satisfaction_mapping[col]}")
        st.bar_chart(df[col].value_counts())

with tab3:
    st.subheader("📂 Aperçu des données")
    st.dataframe(df.head(20))

# 📌 TAB 4 : INDICATEURS DE PERFORMANCE
with tab4:
    st.subheader("📌 Indicateurs de Performance et de Satisfaction")

    # 📌 FONCTION POUR AFFICHER LES INDICATEURS AVEC LABELS VISUELS
    def display_metric(label, value, low_threshold, high_threshold):
        """Affiche un KPI avec une évaluation visuelle : 🔴 Mauvais, 🟡 Moyen, 🟢 Bon"""
        if value < low_threshold:
            status = "🔴 Mauvais"
        elif value < high_threshold:
            status = "🟡 Moyen"
        else:
            status = "🟢 Bon"
        st.metric(label, f"{value:.2f}", status)

    # 📌 AFFICHAGE DES MÉTRIQUES AVEC INDICATEURS
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric("📈 Taux de Croissance de Carrière", df['CareerGrowthRate'].mean(), 0.1, 0.5)
        display_metric("📊 Taux de Promotion", df['PromotionRate'].mean(), 0.05, 0.2)
        display_metric("🔄 Changement de Manager", df['ManagerChangeRate'].mean(), 0.2, 0.8)

    with col2:
        display_metric("😊 Score Satisfaction", df['SatisfactionScore'].mean(), 2.0, 3.5)
        display_metric("💰 Écart Salaire/Satisfaction", df['SalarySatisfactionGap'].mean(), 3000, 8000)
        display_metric("📉 Performance - Implication", df['PerformanceInvolvementGap'].mean(), -1, 1)

    with col3:
        display_metric("🚪 Taux d'Absence", df['AbsenceRate'].mean(), 0.05, 0.2)
        display_metric("✈️ Fatigue liée au Voyage", df['TravelFatigue'].mean(), 5, 20)

# 📌 MATRICE DE CORRÉLATION
st.subheader("📌 Matrice de Corrélation")

# Filtrer les données selon les variables sélectionnées
correlation_matrix = df[selected_features].corr()

# Affichage de la heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

# 📌 ANALYSE DES DÉPARTS

# Appliquer les transformations aux colonnes nécessaires
df["Gender"] = df["Gender"].map({1: "Homme", 0: "Femme"})
df["MaritalStatus"] = df["MaritalStatus"].map({0: "Célibataire", 1: "Marié", 2: "Divorcé"})

# Calcul des taux d'attrition
age_attrition = df.groupby("Age")["Attrition"].mean() * 100
gender_attrition = df.groupby("Gender")["Attrition"].mean() * 100
marital_attrition = df.groupby("MaritalStatus")["Attrition"].mean() * 100

# Sélection du graphique à afficher
option = st.selectbox("Choisissez l'analyse à afficher :", ["📈 Taux d'attrition par âge", "📊 Taux d'attrition par genre", "📉 Taux d'attrition par état matrimonial"])

# Fonction pour afficher un graphique
def plot_bar_chart(data, xlabel, title):
    st.subheader(title)
    st.bar_chart(data)

# Affichage du graphique en fonction de la sélection
if option == "📈 Taux d'attrition par âge":
    # Groupement des tranches d'âge
    age_attrition = df.groupby("Age")["Attrition"].mean() * 100
    plot_bar_chart(age_attrition, "Âge", "Taux d'attrition par âge")

elif option == "📊 Taux d'attrition par genre":
    plot_bar_chart(gender_attrition, "Genre", "Taux d'attrition par genre")

elif option == "📉 Taux d'attrition par état matrimonial":
    plot_bar_chart(marital_attrition, "État matrimonial", "Taux d'attrition par état matrimonial")

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