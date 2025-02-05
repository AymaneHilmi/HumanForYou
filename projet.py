# Import des bibliothèques principales
import numpy as np
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# 📌 CONFIGURATION DE L'INTERFACE
st.set_page_config(page_title="HumanForYou", layout="wide")

# 📌 CHARGEMENT DES DONNÉES
@st.cache_data
def load_data():
    # Chargement des données
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
        hr_data[col] = hr_data[col].fillna(hr_data[col].median())

    # Transformation des variables catégoriques
    hr_data['Age'] = hr_data['Age'].astype(int)
    hr_data['Attrition'] = hr_data['Attrition'].map({'Yes': 1, 'No': 0})
    hr_data['BusinessTravel'] = hr_data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    hr_data['DistanceFromHome'] = hr_data['DistanceFromHome'].astype(int)
    hr_data['Education'] = hr_data['Education'].astype(int)
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
    print(hr_data['MonthlyIncome'])
    hr_data["MonthlyIncome"] = hr_data["MonthlyIncome"].apply(lambda x: round(x, -3)) 
    print(hr_data['MonthlyIncome'])
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
    categorical_columns = ['Departement', 'EducationField', 'JobRole']
    binary_columns = ['Attrition', 'Gender']
    numerical_columns = hr_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in categorical_columns + binary_columns]
    scaler = MinMaxScaler()
    normalized_df = hr_data.copy()
    normalized_df[numerical_columns] = scaler.fit_transform(hr_data[numerical_columns])

    return hr_data, absence_status, absence_days, normalized_df

# Charger les données
df, absence_status, absence_days, normalized_df = load_data()

page1, page2, page3, page4, page5 = st.tabs(["Accueil","Analyse Univariée", "Analyse Bivariée & Multivariée", "Analyse Avancée & Business Insights", "Prédiction"])

with page1 :
    # 📌 TITRE PRINCIPAL
    st.title("📊 HumanForYou - Dashboard")

    # 📌 STATISTIQUES GÉNÉRALES
    st.subheader("📌 Statistiques Clés")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌍 Nombre total d'employés", df.shape[0])
        st.metric("🚀 Taux d'attrition", f"{df['Attrition'].mean() * 100:.2f} %")
        st.metric("📊 Absence moyenne par employé", f"{absence_days['AbsenceDays'].mean():.1f} jours")

        
    with col2:
        st.metric("📈 Salaire moyen", f"${df['MonthlyIncome'].mean():,.2f}")
        st.metric("📅 Ancienneté moyenne", f"{df['YearsAtCompany'].mean():.1f} ans")
        
    with col3:
        st.metric("👨‍💼 % Hommes", f"{df[df['Gender'] == 1].shape[0] / df.shape[0] * 100:.1f} %")
        st.metric("👩 % Femmes", f"{df[df['Gender'] == 0].shape[0] / df.shape[0] * 100:.1f} %")



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
    st.subheader("📌 Indicateurs de Performance et de Satisfaction")

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

    # # 📌 STATISTIQUES D'ABSENTÉISME
    # st.subheader("📌 Statistiques d'Absentéisme")
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.metric("📊 Absence moyenne par employé", f"{absence_days['AbsenceDays'].mean():.1f} jours")

    # with col2:
    #     max_absences_employee = absence_days.loc[absence_days['AbsenceDays'].idxmax()]
    #     st.metric("👥 Employé avec le plus d'absences", f"ID :{max_absences_employee['EmployeeID']} avec {max_absences_employee['AbsenceDays']} jours")

with page2 :
    # 📌 TITRE PRINCIPAL
    st.title("📊 Analyse des Données")
    # 📌 ONGLETS INTERACTIFS 
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistiques détaillées", "📊 Graphiques", "📁 Données brutes", "📌 Indicateurs de Performance"])
    with tab1:
        st.markdown("## 📊 Analyse Univariée")
        st.markdown("#### Exploration des statistiques et répartition des données")

        # === Affichage des Statistiques Générales ===
        st.subheader("📌 Statistiques Générales")
        
        col1, col2 = st.columns([1, 2])  # Séparation en 2 colonnes
        with col1:
            # 📌 Transformer df.info() en DataFrame
            info_dict = {
                "Column": df.columns,
                "Non-Null Count": df.count().values,
                "Dtype": [df[col].dtype for col in df.columns]
            }
            df_info = pd.DataFrame(info_dict)

            # 📊 Affichage stylé
            st.dataframe(df_info, height=500) 
        with col2:
            st.dataframe(df.describe(), height=300)  # Affichage des stats descriptives

        st.markdown("---")

        # === Répartition des employés par département ===
        st.subheader("🏢 Répartition des employés par département")
        department_counts = df['Department'].value_counts()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(department_counts)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(y=department_counts.index, x=department_counts.values, palette="Blues_r", ax=ax)
            ax.set_xlabel("Nombre d'employés")
            ax.set_ylabel("Département")
            ax.set_title("📊 Répartition par Département")
            st.pyplot(fig)


        # === Répartition des valeurs manquantes ===
        st.subheader("🚨 Gestion des valeurs manquantes")

        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

        if missing_values.empty:
            st.success("✅ Aucune valeur manquante détectée ! Tout est propre 🎉")
        else:
            st.warning("⚠️ Certaines colonnes contiennent des valeurs manquantes.")

            # Affichage des valeurs manquantes sous forme de barplot
            st.subheader("📉 Distribution des valeurs manquantes")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette="Reds", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel("Nombre de valeurs manquantes")
            ax.set_title("🔍 Colonnes concernées")
            st.pyplot(fig)

            # Heatmap des valeurs manquantes
            st.subheader("🗺️ Carte de chaleur des valeurs manquantes")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cmap="Reds", cbar=False, yticklabels=False, ax=ax)
            ax.set_title("🔍 Heatmap des valeurs manquantes")
            st.pyplot(fig)

        st.markdown("---")


        # === Répartition des âges ===
        st.subheader("📊 Distribution des âges")
        st.write("📈 Répartition des âges des employés"
                "\n🔴 18 - 25 ans, 🔵 26 - 35 ans, 🟢 36 - 45 ans, 🟡 46 - 55 ans, 🟣 56 - 65 ans")
        age_bins = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], precision=0, right=False)
        age_bins_str = age_bins.astype(str)
        age_distribution = age_bins_str.value_counts().sort_index()
        age_distribution.index = age_distribution.index.str.replace('[', '').str.replace(')', '').str.replace(',', ' -')
        st.bar_chart(age_distribution)

        st.markdown("---")

        # === Répartition des années d'ancienneté ===
        st.subheader("📈 Répartition des années d'ancienneté")
        # axe x : nombre d'années, axe y : nombre d'employés
        st.bar_chart(df['YearsAtCompany'].value_counts())
        satisfaction_mapping = {
            'EnvironmentSatisfaction': 'Satisfaction de l\'environnement de travail',
            'JobSatisfaction': 'Satisfaction du travail',
            'WorkLifeBalance': 'Équilibre travail-vie personnelle'
        }
        st.markdown("---")


        # === Répartition des salaires ===
        st.subheader("💰 Répartition des salaires par tranche")
        salary_bins = pd.cut(df['MonthlyIncome'], bins=5, precision=0)
        salary_bins_str = salary_bins.astype(str)
        salary_distribution = salary_bins_str.value_counts().sort_index()
        salary_distribution.index = salary_distribution.index.str.replace('(', '').str.replace(']', '').str.replace(',', ' -')

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("💼 Distribution des salaires :")
            st.dataframe(salary_distribution)

        with col2:
            st.bar_chart(salary_distribution)

        st.markdown("---")

        

        # === Satisfaction des employés ===
        satisfaction_mapping = {
            'EnvironmentSatisfaction': 'Satisfaction de l\'environnement de travail',
            'JobSatisfaction': 'Satisfaction du travail',
            'WorkLifeBalance': 'Équilibre travail-vie personnelle'
        }

        st.subheader("😀 Satisfaction des employés")

        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']

        for col in satisfaction_cols:
            st.write(f"### 📊 {satisfaction_mapping[col]}")
            
            # Création des colonnes pour une meilleure disposition
            col1, col2 = st.columns([1, 2])

            with col1:
                st.write("📋 Répartition des niveaux :")
                st.dataframe(df[col].value_counts().rename_axis("Niveau").reset_index(name="Nombre d'employés"))

            with col2:
                st.write("📊 Distribution graphique :")
                st.bar_chart(df[col].value_counts())
        st.markdown("---")

    with tab2:
        st.subheader("2️⃣ Visualisation Dynamique des Variables")
        # Sélection de la variable à explorer
        selected_var = st.selectbox(
            "📊 Sélectionnez une variable numérique à analyser :",
            df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        )

        # Création des colonnes pour un affichage structuré
        col1, col2 = st.columns([1, 2])

        # Histogramme interactif
        st.write("📏 **Distribution des valeurs (Histogramme)**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df[selected_var], kde=True, color="royalblue", ax=ax)
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

        # Boxplot interactif

        st.write("📦 **Diagramme a Moustache (Boxplot)**")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[selected_var], color="lightcoral", ax=ax)
        ax.set_xlabel(selected_var)
        st.pyplot(fig)

        # KDE Plot interactif (Courbe de densité)
        st.write("📊 **Courbe de densité (KDE Plot)**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.kdeplot(df[selected_var], shade=True, color="green", ax=ax)
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Densité")
        st.pyplot(fig)

        st.markdown("---")

    with tab3:
        st.subheader("📂 Aperçu des données")
        st.dataframe(df.head(20))

with page3:
    with st.expander("🔎 Options d'analyse", expanded=False):
        selected_features = st.multiselect("Sélectionnez les variables à afficher dans la matrice de corrélation :", 
                                       df.select_dtypes(include=['int64', 'float64']).columns.tolist(), 
                                       default=["Attrition","JobLevel", "YearsAtCompany", "YearsWithCurrManager",
                                                "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
                                                "PercentSalaryHike", "StockOptionLevel", "JobSatisfaction", "WorkLifeBalance",
                                                "EnvironmentSatisfaction", "TrainingTimesLastYear", "BusinessTravel",
                                                "DistanceFromHome", "AbsenceDays", "TotalWorkingYears",
                                                "PerformanceRating", "JobInvolvement"])


    # 📌 MATRICE DE CORRÉLATION INTERACTIVE
    st.subheader("📌 Matrice de Corrélation Interactive")

    # Filtrer les données selon les variables sélectionnées
    correlation_matrix = normalized_df[selected_features].corr()

    # ✅ **Correction de l'ordre des indices pour la diagonale correcte**
    correlation_matrix = correlation_matrix.iloc[::-1]  # Inverser l'ordre des lignes pour la bonne orientation

    # Création de la figure Plotly avec des couleurs modernes
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index)[::-1],  # Inverser l'ordre des colonnes pour correspondre
        colorscale="RdBu",  # Palette de couleurs moderne
        annotation_text=np.round(correlation_matrix.values, 2),
        showscale=True,
        reversescale=True
    )

    # Mise en page optimisée
    fig.update_layout(
        title="Matrice de Corrélation",
        xaxis=dict(title="Variables"),
        yaxis=dict(title="Variables"),
        margin=dict(l=100, r=100, t=50, b=50),
        height=700
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # 📌 ANALYSE DES DÉPARTS (Comparaison Employés Partis vs. Restants)
    st.subheader("📌 Comparaison des Employés Partis vs. Restants")

    # Transformation des colonnes pour une meilleure lisibilité
    df["Gender"] = df["Gender"].map({1: "Homme", 0: "Femme"})
    df["MaritalStatus"] = df["MaritalStatus"].map({0: "Célibataire", 1: "Marié", 2: "Divorcé"})

    # 📌 COMPARAISON PAR FACTEUR CLÉ
    attrition_comparison = {
        "💰 Salaire Moyen": df.groupby("Attrition")["MonthlyIncome"].mean(),
        "🏢 Années dans l'Entreprise": df.groupby("Attrition")["YearsAtCompany"].mean(),
        "🚀 Dernière Augmentation (%)": df.groupby("Attrition")["PercentSalaryHike"].mean(),
        "🔄 Nombre d'Entreprises Précédentes": df.groupby("Attrition")["NumCompaniesWorked"].mean(),
        "📈 Niveau Hiérarchique": df.groupby("Attrition")["JobLevel"].mean(),
        "🏠 Distance Domicile-Travail (km)": df.groupby("Attrition")["DistanceFromHome"].mean(),
        "📊 Score Satisfaction": df.groupby("Attrition")["SatisfactionScore"].mean(),
        "📈 Taux de Promotion": df.groupby("Attrition")["PromotionRate"].mean(),
        "🚪 Taux d'Absence": df.groupby("Attrition")["AbsenceRate"].mean(),
    }

    # 📌 Sélection du critère de comparaison
    option = st.selectbox(
        "Choisissez un critère d'analyse :", 
        list(attrition_comparison.keys())
    )

    # 📊 Fonction pour afficher le graphique comparatif
    def plot_attrition_chart(data, title):
        st.subheader(title)
        st.bar_chart(data)

    # 📌 Affichage du graphique sélectionné
    plot_attrition_chart(attrition_comparison[option], option)

    # 📌 INTERPRÉTATION DES RÉSULTATS
    st.subheader("📌 Interprétation des Résultats")

    st.write("📌 **Employés Partis (Attrition = 1)**")
    st.write("📌 **Employés Restants (Attrition = 0)**")

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

with page4:
    #page 4
    st.write("📌 **Niveau moyen de satisfaction des employés ayant quitté :**")

with page5:
    # 📌 ONGLETS INTERACTIFS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Régression Logistique", "🧠 SVM", "🌲 Random Forest", "🔮 Aide à la Décision"])

    # 📌 VARIABLES À UTILISER DANS LES MODÈLES
    features = [
        "JobRole", "JobLevel", "YearsAtCompany", "YearsWithCurrManager",
        "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
        "PercentSalaryHike", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "TrainingTimesLastYear",
        "BusinessTravel",
        "AbsenceDays",
        "TotalWorkingYears",
        "Department"]

    target = "Attrition"


    def display_model_results(model, X_test, y_test, y_pred, y_proba, model_name):
        """
        Fonction pour afficher les résultats du modèle :
        - Statistiques principales
        - Matrice de confusion
        - Courbe ROC
        - Importance des variables
        """

        st.title(f"📊 Analyse de l'Attrition - {model_name}")

        ## 📊 Statistiques générales
        st.subheader("📌 Statistiques du Modèle")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Précision (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f} %")
        with col2:
            st.metric("🎯 Rappel (Recall)",
                      f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2f}")
        with col3:
            st.metric("✅ Score F1",
                      f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.2f}")

        # 📌 Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = ff.create_annotated_heatmap(
            z=cm[::-1],
            x=["Prédit : Non", "Prédit : Oui"],
            y=["Réel : Oui", "Réel : Non"],
            colorscale="RdBu",
            annotation_text=cm[::-1].astype(str),
            showscale=True,
            reversescale=True
        )
        st.subheader("📊 Matrice de confusion")
        fig_cm.update_layout(xaxis=dict(title="Classe Prédite"), yaxis=dict(title="Classe Réelle"))
        st.plotly_chart(fig_cm, use_container_width=True, key=f"confusion_matrix_{model_name}")

        # 📌 Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.2f})",
                       line=dict(color="darkorange", width=2))
        )
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Model", line=dict(color="navy", dash="dash"))
        )
        st.subheader("📉 Courbe ROC - Capacité de Prédiction du Modèle")
        st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_curve_{model_name}")

        # 📌 Récupération du modèle sous-jacent
        if isinstance(model, Pipeline):
            classifier = model.named_steps.get('classifier', model)
        else:
            classifier = model

        feature_names = X_test.columns
        coefficients = None

        # 📌 Cas spécifique pour Random Forest
        st.subheader("📊 Importance des Variables")
        if model_name == "Random Forest":
            coefficients = classifier.feature_importances_

        else:
            # 📌 Cas 1 : Modèles linéaires (Logistic Regression, SVM Linéaire)
            if hasattr(classifier, "coef_"):
                coefficients = classifier.coef_[0]

            # 📌 Cas 2 : Modèles sans coefficients (SVM avec Kernel, KNN, etc.)
            else:
                perm_importance = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42)
                coefficients = perm_importance.importances_mean  # Moyenne des impacts

        # 📌 Ajout du signe basé sur la corrélation avec y_proba
        correlations = np.array([np.corrcoef(X_test[col], y_proba)[0, 1] for col in feature_names])
        coefficients = coefficients * np.sign(correlations)  # Appliquer le signe

        # 📌 Création du DataFrame
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})

        # 📌 Trier les coefficients par ordre décroissant d'importance absolue
        feature_importance_df["Abs_Coefficient"] = feature_importance_df["Coefficient"].abs()
        feature_importance_df = feature_importance_df.sort_values(by="Abs_Coefficient", ascending=False).head(10).drop(
            columns=["Abs_Coefficient"])

        # 📌 Création du graphique avec Plotly
        fig_feature_imp = go.Figure()
        fig_feature_imp.add_trace(
            go.Bar(
                x=feature_importance_df["Feature"],
                y=feature_importance_df["Coefficient"],
                marker=dict(
                    color=feature_importance_df["Coefficient"],
                    colorscale="RdBu",
                    showscale=True
                ),
            )
        )

        # 📌 Mise en page optimisée
        fig_feature_imp.update_layout(
            title="📈 Top 10 Variables les Plus Influentes sur l'Attrition",
            yaxis=dict(title="Effet sur l'Attrition"),
            height=500
        )

        # 📌 Affichage dans Streamlit
        st.plotly_chart(fig_feature_imp, use_container_width=True, key=f"feature_importance_{model_name}")

    with tab1:
        # 📌 PRÉPARATION DES DONNÉES
        categorical_features = ["JobRole", "BusinessTravel", "Department"]
        numerical_features = [col for col in features if col not in categorical_features]

        # Encoder les variables catégoriques
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]),
                                  columns=encoder.get_feature_names_out(categorical_features))
        df_encoded.index = df.index

        # Normaliser les variables numériques
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]),
                                 columns=numerical_features)
        df_scaled.index = df.index

        # Combiner les données transformées
        df_final = pd.concat([df_encoded, df_scaled, df[target]], axis=1)

        # 📌 DIVISION DES DONNÉES EN TRAIN & TEST
        X = df_final.drop(columns=[target])
        y = df_final[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        # 📌 PRÉDICTION & AJUSTEMENT DU SEUIL
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.35  # Ajustement du seuil
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 📌 ÉVALUATION DU MODÈLE
        accuracy = accuracy_score(y_test, y_pred)

        display_model_results(model, X_test, y_test, y_pred, y_proba, "Régression Logistique")
        results_logistic = classification_report(y_test, y_pred, output_dict=True)

    with tab2:
        # ============================
        # 📌 MODÈLE DE PRÉDICTION SVM 
        # ============================
        # Créer une copie du dataframe pour le modèle SVM
        df_svm = df.copy()

        # Conversion des variables de satisfaction en entier
        df_svm["JobSatisfaction"] = df_svm["JobSatisfaction"].astype(int)
        df_svm["EnvironmentSatisfaction"] = df_svm["EnvironmentSatisfaction"].astype(int)
        df_svm["WorkLifeBalance"] = df_svm["WorkLifeBalance"].astype(int)
        df_svm["SatisfactionScore"] = (df_svm["JobSatisfaction"] + df_svm["EnvironmentSatisfaction"] + df_svm["WorkLifeBalance"]) / 3
        df_svm["SalarySatisfactionGap"] = df_svm["MonthlyIncome"] / (df_svm["JobSatisfaction"] + 1)
        # Calcul de la différence entre PerformanceRating et JobInvolvement
        df_svm["PerformanceInvolvementGap"] = df_svm["PerformanceRating"].astype(int) - df_svm["JobInvolvement"].astype(int)
        df_svm["AbsenceRate"] = df_svm["AbsenceDays"] / (df_svm["YearsAtCompany"] + 1)
        # Calcul de TravelFatigue avant modification de BusinessTravel
        df_svm["TravelFatigue"] = df_svm["BusinessTravel"] * df_svm["DistanceFromHome"]
        # Encoder BusinessTravel en tant que variable catégorielle
        df_svm["BusinessTravel"] = df_svm["BusinessTravel"].astype(str)

        X = df_svm[features].copy()
        y = df_svm[target]

        # --- Encodage des variables catégorielles ---
        # On encode "JobRole", "Department" et "BusinessTravel" via one-hot encoding
        X = pd.get_dummies(X, columns=["JobRole", "Department", "BusinessTravel"], drop_first=True)

        # --- Normalisation des données ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        # --- Séparation en ensembles d'entraînement et de test ---
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # --- Initialisation et entraînement du modèle SVM ---
        # On utilise 'class_weight' pour compenser le déséquilibre et se concentrer sur les départs (classe positive)
        # Les paramètres sont fixés pour réduire le temps d'exécution
        svm_model = SVC(probability=True, random_state=42, class_weight='balanced', kernel='rbf', C=1, gamma=0.1)
        svm_model.fit(X_train, y_train)

        # --- Prédictions ---
        y_pred = svm_model.predict(X_test)

        # --- Évaluation du modèle SVM ---
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"])

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        y_proba = svm_model.predict_proba(X_test)[:, 1]
        display_model_results(svm_model, X_test, y_test, y_pred, y_proba, "SVM")
        results_svm = classification_report(y_test, y_pred, output_dict=True)

    with tab3:
        # Préparation des données
        categorical_columns = ['Departement', 'EducationField', 'JobRole']
        binary_columns = ['Attrition', 'Gender']
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns if col not in categorical_columns + binary_columns]
        scaler = MinMaxScaler()
        normalized_df = df.copy()
        normalized_df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        df_encoded = pd.get_dummies(df[features])

        X = df_encoded
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 📌 Entraînement du modèle
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # 📌 Génération des bonnes prédictions pour Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe 1

        # 📌 Appel correct de display_model_results pour la Random Forest
        display_model_results(rf_model, X_test, y_test, rf_pred, rf_proba, "Random Forest")
        results_rf = classification_report(y_test, rf_pred, output_dict=True)

    with tab4:
        # 📌 TITRE PRINCIPAL
        col1, col2 = st.columns(2)
        with col1:
            # Création d'un DataFrame pour stocker les résultats (accuracy, recall, f1-score, etc.)
            df_results = pd.DataFrame({
                "Logistic Regression": results_logistic["1"],
                "SVM": results_svm["1"],
                "Random Forest": results_rf["1"]
            })
            # 📌 INTERPRÉTATION DES RÉSULTATS
            st.subheader("📊 Résultats des Modèles de Prédiction")
            st.dataframe(df_results)
        with col2:
            st.subheader("Lancer l'aide à la décision")
            st.write("Cliquez sur le bouton ci-dessous pour obtenir les résultats.")
            if st.button("🚀 Lancer l'aide à la décision"):
                # Animation de chargement
                with st.spinner("Calcul des résultats..."):
                    time.sleep(3)
                    best_model = df_results.idxmax(axis=1).value_counts().idxmax()
                    st.subheader("🏆 Meilleur Modèle de Prédiction")
                st.success(f"Le meilleur modèle est : **{best_model}**")
        