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
        hr_data[col].fillna(hr_data[col].mean().round(), inplace=True)

    # Transformation des variables catégoriques
    hr_data['Attrition'] = hr_data['Attrition'].map({'Yes': 1, 'No': 0})
    hr_data['BusinessTravel'] = hr_data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    hr_data['Gender'] = hr_data['Gender'].map({'Male': 1, 'Female': 0})
    hr_data['MaritalStatus'] = hr_data['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    hr_data['PercentSalaryHike'] = hr_data['PercentSalaryHike'] / 100  # Mise à l'échelle
    
    return hr_data

# Charger les données
df = load_data()

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


# ONGLETS INTERACTIFS 
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
    st.bar_chart(df['Age'])

    st.subheader("💰 Répartition des salaires")
    st.line_chart(df['MonthlyIncome'])

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

# Définition des tranches d'âge
df["AgeGroup"] = df["Age"].apply(age_category)

# Calcul du taux d'attrition par tranche d'âge
age_attrition = df.groupby("AgeGroup")["Attrition"].mean() * 100

# Affichage dans Streamlit
st.subheader("📌 Taux d'attrition par tranche d'âge")
st.bar_chart(age_attrition)

# 📌 ANALYSE DE L'ATTRITION PAR GENRE
st.subheader("📊 Taux d'attrition par genre")

# Remplacement des valeurs de la colonne Gender
df["Gender"] = df["Gender"].map({1: "Homme", 0: "Femme"})

# Calcul du taux d'attrition par genre
gender_attrition = df.groupby("Gender")["Attrition"].mean() * 100

# Affichage des résultats sous forme de graphique
st.bar_chart(gender_attrition)

# 📌 FIN DU SCRIPT
st.success("🚀 Analyse terminée ! Sélectionnez des variables dans la sidebar pour explorer plus en détail. ")