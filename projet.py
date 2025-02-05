# Import des bibliothèques principales
import numpy as np
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
    st.subheader("🚀 Un projet avancé d'exploration et de visualisation des données")

    # 📝 Présentation du projet
    st.markdown(
        """
        Ce tableau de bord a été conçu pour **analyser en profondeur les données RH** d’une entreprise et fournir des insights clés sur l’attrition, l’absentéisme et les facteurs influençant la satisfaction des employés.  
        
        💡 **Objectifs du projet** :
        - Explorer et comprendre les tendances des données RH.
        - Identifier les facteurs clés influençant le départ des employés.
        - Proposer des recommandations stratégiques basées sur une analyse avancée.
        
        📊 Grâce à des **visualisations interactives et dynamiques**, ce dashboard permet d’extraire des informations pertinentes pour une meilleure prise de décision.
        """
    )

    # 👥 Présentation des contributeurs
    st.subheader("👨‍💻 Équipe Projet")
    
    team_members = [
        {"name": "🔹 **Aymane Hilmi**", "role": "Data Analyst & Développeur Streamlit"},
        {"name": "🔹 **[Nom 2]**", "role": "Expert en Modélisation Statistique"},
        {"name": "🔹 **[Nom 3]**", "role": "Spécialiste en RH & Business Insights"}
    ]

    for member in team_members:
        st.markdown(f"{member['name']} - *{member['role']}*")

    # 🚀 Points forts du projet
    st.subheader("🔥 Pourquoi ce Dashboard est Innovant ?")
    st.markdown(
        """
        ✅ **Interface Interactive** : Navigation fluide et expérience utilisateur optimisée.  
        ✅ **Visualisations Avancées** : Graphiques détaillés pour une meilleure compréhension des données.  
        ✅ **Insights Stratégiques** : Analyse approfondie avec recommandations business.  
        ✅ **Technologies Modernes** : Utilisation de *Streamlit, Matplotlib, Seaborn, Pandas, et Scikit-Learn* pour des analyses puissantes.  
        """
    )

with page2 :
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
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Régression Logistique", "🧠 SVM", "🌲 Random Forest", "🌳 Decision Tree"])

    # 📌 VARIABLES À UTILISER DANS LES MODÈLES
    features = [
        "JobRole", "JobLevel", "YearsAtCompany", "YearsWithCurrManager",
        "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
        "PercentSalaryHike", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "TrainingTimesLastYear",
        "BusinessTravel",
        "AbsenceDays",
        "TotalWorkingYears",
        "Department"
    ]

    target = "Attrition"

    with tab1:
        # Séparation des données catégoriques et numériques
        categorical_features = ["JobRole", "BusinessTravel", "Department"]
        numerical_features = [col for col in features if col not in categorical_features]

        # Préparation des données
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

        # Pipeline de Modélisation
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=500))
        ])

        # Séparation en jeu de test et entraînement
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]


        # 📌 **Affichage dans Streamlit**
        st.title("📊 Analyse de l'Attrition - Régression Logistique")

        ## 📊 Statistiques générales
        st.subheader("📌 Statistiques du Modèle")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Précision (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f} %")
        with col2:
            st.metric("🎯 Rappel (Recall)",
                      f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2f}")
        with col3:
            st.metric("✅ Score F1", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.2f}")

        # 📌 Calcul de la Matrice de Confusion
        cm = confusion_matrix(y_test, y_pred)

        # 📌 Création d'une heatmap interactive avec Plotly
        fig_cm = ff.create_annotated_heatmap(
            z=cm[::-1],  # Inverser l'ordre des lignes pour correspondre au format
            x=["Prédit : Non", "Prédit : Oui"],
            y=["Réel : Oui", "Réel : Non"],  # Inversion pour correspondre à la diagonale correcte
            colorscale="RdBu",  # Palette moderne
            annotation_text=cm[::-1].astype(str),  # Ajouter les valeurs comme annotations
            showscale=True,
            reversescale=True
        )

        # 📌 Mise en page optimisée
        st.subheader("📊 Matrice de confusion")
        fig_cm.update_layout(
            xaxis=dict(title="Classe Prédite"),
            yaxis=dict(title="Classe Réelle"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500
        )

        # 📌 Affichage dans Streamlit
        st.plotly_chart(fig_cm, use_container_width=True)

        # 📌 Calcul des valeurs pour la courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        # 📌 Création du graphique avec Plotly
        fig_roc = go.Figure()

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_auc:.2f})",
                line=dict(color="darkorange", width=2)
            )
        )

        # 📌 Ajout de la ligne de référence (diagonale)
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Model",
                line=dict(color="navy", dash="dash")
            )
        )

        # 📌 Mise en page optimisée pour Streamlit
        fig_roc.update_layout(
            xaxis=dict(title="Taux de Faux Positifs (FPR)"),
            yaxis=dict(title="Taux de Vrais Positifs (TPR)"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500
        )

        # 📌 Affichage dans Streamlit
        st.subheader("📉 Courbe ROC - Capacité de Prédiction du Modèle")
        st.plotly_chart(fig_roc, use_container_width=True)

        # 📌 Ajout d'une analyse de l'importance des variables avec signe (positif/négatif)
        st.subheader("📈 Importance des Variables - Impact sur l'Attrition")

        # Récupérer les coefficients du modèle de régression logistique
        coefficients = model.named_steps['classifier'].coef_[0]

        # Associer les coefficients aux noms des features après transformation
        feature_names = preprocessor.get_feature_names_out()

        # Créer un DataFrame pour stocker les résultats
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefficients})

        # Trier les coefficients par ordre décroissant d'importance absolue
        feature_importance_df["Abs_Coefficient"] = feature_importance_df["Coefficient"].abs()
        feature_importance_df = feature_importance_df.sort_values(by="Abs_Coefficient", ascending=False).head(10).drop(
            columns=["Abs_Coefficient"])

        # Création du graphique avec Plotly pour afficher l'effet positif ou négatif
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
            yaxis=dict(title="Effet sur l'Attrition"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500
        )

        # 📌 Affichage dans Streamlit
        st.plotly_chart(fig_feature_imp, use_container_width=True)

    with tab2:
         # ============================
        # 📌 MODÈLE DE PRÉDICTION SVM 
        # ============================

        st.header("Modèle de Prédiction SVM - Optimisé pour détecter les départs")

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

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        y_proba = svm_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        st.title("📊 Analyse de l'Attrition - SVM")
        st.subheader("📌 Statistiques du Modèle")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Précision (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f} %")
        with col2:
            st.metric("🎯 Rappel (Recall)",
                      f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2f}")
        with col3:
            st.metric("✅ Score F1", f"{classification_report(y_test, y_pred, output_dict=True)['1']['f1-score']:.2f}")

        fig_cm = ff.create_annotated_heatmap(
            z=conf_matrix[::-1],  # Inversion des lignes pour le bon alignement
            x=["Prédit : Non", "Prédit : Oui"],
            y=["Réel : Oui", "Réel : Non"],
            colorscale="RdBu",
            annotation_text=conf_matrix[::-1].astype(str),
            showscale=True,
            reversescale=True
        )
        fig_cm.update_layout(
            xaxis=dict(title="Classe Prédite"),
            yaxis=dict(title="Classe Réelle"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500
        )
        st.subheader("📊 Matrice de confusion")
        st.plotly_chart(fig_cm, use_container_width=True)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.2f})",
            line=dict(color="darkorange", width=2)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Model",
            line=dict(color="navy", dash="dash")
        ))
        fig_roc.update_layout(
            title="Courbe ROC - SVM Optimisé pour Attrition",
            xaxis=dict(title="Taux de Faux Positifs (FPR)"),
            yaxis=dict(title="Taux de Vrais Positifs (TPR)"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500,
            template="plotly_white"
        )
        st.subheader("📉 Courbe ROC - Capacité de Prédiction du Modèle")
        st.plotly_chart(fig_roc, use_container_width=True)

        # Calcul de la corrélation entre chaque feature et la probabilité prédite d'attrition
        importances = {}
        for feature in X.columns:
            # Calcul de la corrélation de Pearson entre la feature et y_proba
            importances[feature] = np.corrcoef(X_test[feature], y_proba)[0, 1]

        importance_df = pd.DataFrame.from_dict(importances, orient='index', columns=['Correlation'])
        importance_df = importance_df.sort_values(by='Correlation', ascending=False)

        # 📌 Affichage de l'importance des variables sur l'attrition
        st.subheader("📈 Importance des Variables - Impact sur l'Attrition")

        # Récupérer les coefficients des features après transformation
        importances = {}
        for feature in X.columns:
            # Calcul de la corrélation de Pearson entre chaque feature et la probabilité d'attrition prédite
            importances[feature] = np.corrcoef(X_test[feature], y_proba)[0, 1]

        # Création d'un DataFrame pour stocker les résultats
        importance_df = pd.DataFrame.from_dict(importances, orient='index', columns=['Correlation'])

        # Trier les features par ordre décroissant d'importance absolue
        importance_df["Abs_Correlation"] = importance_df["Correlation"].abs()
        importance_df = importance_df.sort_values(by="Abs_Correlation", ascending=False).head(10).drop(
            columns=["Abs_Correlation"])

        # 📌 Création du graphique avec Plotly pour afficher l'effet positif ou négatif
        fig_feature_imp = go.Figure()

        fig_feature_imp.add_trace(
            go.Bar(
                x=importance_df.index,
                y=importance_df["Correlation"],
                marker=dict(
                    color=importance_df["Correlation"],
                    colorscale="RdBu",
                    showscale=True
                ),
            )
        )

        # 📌 Mise en page optimisée
        fig_feature_imp.update_layout(
            title="📈 Top 10 Variables les Plus Influentes sur l'Attrition",
            xaxis=dict(title="Variables", tickangle=-45),
            yaxis=dict(title="Effet sur l'Attrition"),
            margin=dict(l=100, r=100, t=50, b=50),
            height=500
        )

        # 📌 Affichage dans Streamlit
        st.plotly_chart(fig_feature_imp, use_container_width=True)

    with tab3:
        # Titre de l'application
        st.title("Prédiction de l'Attrition avec Random Forest")
        st.markdown("Ce modèle utilise un Random Forest pour prédire si un employé quittera l'entreprise (attrition).")
        # Préparation des données
        categorical_columns = ['Departement', 'EducationField', 'JobRole']
        binary_columns = ['Attrition', 'Gender']
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_columns = [col for col in numerical_columns if col not in categorical_columns + binary_columns]
        scaler = MinMaxScaler()
        normalized_df = df.copy()
        normalized_df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

        # Sélection des variables pour la prédiction
        features = [
            "JobRole", "JobLevel", "YearsAtCompany", "YearsWithCurrManager",
            "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
            "PercentSalaryHike", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
            "TrainingTimesLastYear",
            "BusinessTravel",
            "AbsenceDays",
            "TotalWorkingYears",
            "Department"]

        df_encoded = pd.get_dummies(df[features])

        X = df_encoded
        y = df['Attrition']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 📌 Entraînement du modèle
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # 📌 Évaluation du modèle
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)

        # Affichage des résultats
        st.subheader("📊 Résultats de la Prédiction")
        st.write(f"**Accuracy :** {rf_accuracy}")
        st.write("Prédiction sur l'ensemble de test :")
        st.write("🔴 0 : Non Attrition, 🟢 1 : Attrition")

        # Recall
        st.write("**Recall :**", recall_score(y_test, rf_pred))
        st.write("**Classification Report :**")
        st.text(classification_report(y_test, rf_pred))

        # Matrice de confusion
        st.write("**Confusion Matrix :**")
        cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap='Blues', alpha=0.7)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha='center', va='center', color='black')
        plt.xlabel('Prédictions')
        plt.ylabel('Réel')
        st.pyplot(fig)

        # 📌 Importance des Variables
        st.subheader("📈 Importance des Variables")
        feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        st.bar_chart(feature_importance.head(10))

        # 📌 Conclusion
        st.write("L'importance des variables montre quelles caractéristiques influencent le plus la prédiction d'attrition.")
        st.write("L'accuracy et le recall sont des métriques clés pour évaluer la performance du modèle.")

    with tab4:
        st.subheader("🌳 Prédiction avec Decision Tree")

        # Définition des features et de la target pour le Decision Tree
        features = [
            "JobRole",
            "JobLevel",
            "YearsAtCompany",
            "YearsWithCurrManager",
            "YearsSinceLastPromotion",
            "NumCompaniesWorked",
            "MonthlyIncome",
            "PercentSalaryHike",
            "JobSatisfaction",
            "WorkLifeBalance",
            "EnvironmentSatisfaction",
            "TrainingTimesLastYear",
            "BusinessTravel",
            "AbsenceDays",
            "TotalWorkingYears"
        ]
        target = "Attrition"

        # Séparation des variables catégoriques et numériques
        categorical_features = ["JobRole", "BusinessTravel", "Department"]
        numerical_features = [col for col in features if col not in categorical_features]

        # Création d'un préprocesseur avec ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

        # Transformation des données
        df_transformed = pd.DataFrame(preprocessor.fit_transform(df[categorical_features + numerical_features]),
                                      columns=preprocessor.get_feature_names_out(),
                                      index=df.index)
        # Combinaison avec la target
        df_final = pd.concat([df_transformed, df[target]], axis=1)

        # Division des données en ensembles d'entraînement et de test
        X = df_final.drop(columns=[target])
        y = df_final[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Définition de la grille de recherche pour le Decision Tree
        param_grid_dt = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2'],
            'class_weight': [None, 'balanced']
        }

        grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt,
                               cv=5, scoring='f1', n_jobs=-1)
        grid_dt.fit(X_train, y_train)
        best_dt = grid_dt.best_estimator_

        st.write("### Meilleurs paramètres pour Decision Tree")
        st.write(grid_dt.best_params_)

        # Prédiction avec le meilleur modèle
        y_pred = best_dt.predict(X_test)
        accuracy_dt = accuracy_score(y_test, y_pred)

        st.write(f"📌 **Précision du modèle Decision Tree optimisé :** {accuracy_dt * 100:.2f} %")

        # Calcul de la matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Affichage de la matrice de confusion sous forme de heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["Reste", "Part"], yticklabels=["Reste", "Part"], ax=ax_cm)
        ax_cm.set_xlabel("Prédiction")
        ax_cm.set_ylabel("Réel")
        ax_cm.set_title("Matrice de Confusion")
        st.pyplot(fig_cm)


        # Fonction pour afficher les statistiques du modèle
        def display_metrics(y_true, y_pred, model_name="Decision Tree"):
            st.subheader(f"📊 Performances du modèle : {model_name}")
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
            df_report = pd.DataFrame(class_report).transpose()
            st.dataframe(df_report)
            st.write(f"📌 **Précision globale (Accuracy) :** {class_report['accuracy'] * 100:.2f} %")
            st.write(f"📌 **Score F1 (moyenne pondérée) :** {class_report['weighted avg']['f1-score']:.2f}")
            st.write(f"📌 **Rappel (Recall, capacité à détecter les partants) :** {class_report['1']['recall']:.2f}")
            st.write(
                f"📌 **Précision (Précision sur les employés réellement partants) :** {class_report['1']['precision']:.2f}")


        # Affichage des métriques
        display_metrics(y_test, y_pred, model_name="Decision Tree Optimisé")