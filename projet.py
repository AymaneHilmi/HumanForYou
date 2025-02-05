import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff

from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(page_title="HumanForYou", layout="wide")

# -----------------------------------------------------------------------------
# Fonctions Utilitaires
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Charge et prépare les données
    
    Retourne :
        hr_data: DataFrame principal après fusion et transformation
        absence_status: DataFrame indiquant l'état (Absent/Present) pour chaque jour
        absence_days: DataFrame du nombre total de jours d'absence par employé
        normalized_df: hr_data avec les colonnes numériques normalisées (pour la corrélation)
    """
    try:
        # Chargement des fichiers CSV
        hr_data = pd.read_csv('./data/general_data.csv')
        survey_data = pd.read_csv('./data/employee_survey_data.csv')
        manager_data = pd.read_csv('./data/manager_survey_data.csv')
    except Exception as e:
        st.error("Erreur lors du chargement des fichiers CSV.")
        raise e

    # Fusion des datasets sur EmployeeID
    hr_data = hr_data.merge(survey_data, on='EmployeeID').merge(manager_data, on='EmployeeID')

    # Suppression des colonnes inutiles et lignes avec données manquantes
    hr_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)
    hr_data = hr_data.dropna(subset=['NumCompaniesWorked', 'TotalWorkingYears'])

    # Imputation des valeurs manquantes pour certaines colonnes
    for col in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
        hr_data[col] = hr_data[col].fillna(hr_data[col].median())

    # Transformation des colonnes numériques et catégoriques
    numeric_transformations = {
        'Age': int,
        'DistanceFromHome': int,
        'Education': int,
        'EmployeeID': int,
        'JobLevel': int,
        'MonthlyIncome': float,
        'NumCompaniesWorked': int,
        'StockOptionLevel': int,
        'TotalWorkingYears': int,
        'TrainingTimesLastYear': int,
        'YearsAtCompany': int,
        'YearsSinceLastPromotion': int,
        'YearsWithCurrManager': int,
        'JobInvolvement': int,
        'PerformanceRating': int,
        'EnvironmentSatisfaction': int,
        'WorkLifeBalance': int
    }
    for col, func in numeric_transformations.items():
        hr_data[col] = hr_data[col].apply(func)

    # Mappings pour les variables catégoriques
    hr_data['Attrition'] = hr_data['Attrition'].map({'Yes': 1, 'No': 0})
    hr_data['BusinessTravel'] = hr_data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    hr_data['Gender'] = hr_data['Gender'].map({'Male': 1, 'Female': 0})
    hr_data['MaritalStatus'] = hr_data['MaritalStatus'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    hr_data['EducationField'] = hr_data['EducationField'].astype('category')
    hr_data['JobRole'] = hr_data['JobRole'].astype('category')
    hr_data['Department'] = hr_data['Department'].astype('category')

    # Transformation en pourcentages
    hr_data['PercentSalaryHike'] = hr_data['PercentSalaryHike'].astype(float) / 100

    # Arrondi du salaire mensuel
    hr_data["MonthlyIncome"] = hr_data["MonthlyIncome"].apply(lambda x: round(x, -3))

    # Chargement des données d'absentéisme et renommage de la colonne d'index
    in_time_data = pd.read_csv('./data/in_time.csv').rename(columns={"Unnamed: 0": "EmployeeID"})
    out_time_data = pd.read_csv('./data/out_time.csv').rename(columns={"Unnamed: 0": "EmployeeID"})

    # Calcul de l'état d'absence : 
    # Si l'une des dates a une valeur manquante (NaN) dans in_time ou out_time, l'employé est considéré absent ce jour-là
    absence_bool = (in_time_data.iloc[:, 1:].isna() | out_time_data.iloc[:, 1:].isna())
    # Supprimer les colonnes (dates) pour lesquelles TOUS les employés sont absents
    # La méthode .all(axis=0) vérifie pour chaque colonne (en ignorant l'en-tête) si toutes les valeurs sont True
    absence_bool = absence_bool.loc[:, ~absence_bool.all(axis=0)]

    # Remplacer les valeurs booléennes par des chaînes de caractères pour faciliter l'affichage
    absence_status = absence_bool.replace({True: 'Absent', False: 'Present'})

    # Réinsérer la colonne 'EmployeeID' en première position
    absence_status.insert(0, 'EmployeeID', in_time_data['EmployeeID'])

    # Calculer le nombre de jours d'absence par employé
    absence_days = absence_status.iloc[:, 1:].apply(lambda x: (x == 'Absent').sum(), axis=1)
    absence_days = pd.DataFrame({'EmployeeID': absence_status['EmployeeID'], 'AbsenceDays': absence_days})

    # Enregistrement des données d'absence
    absence_days.to_csv('./data/absence_days.csv', index=False)
    absence_status.to_csv('./data/absence_status.csv', index=False)

    # Fusion du nombre de jours d'absence dans hr_data
    hr_data = hr_data.merge(absence_days, on='EmployeeID', how='left')

    # Création de variables supplémentaires 
    hr_data["CareerGrowthRate"] = hr_data["JobLevel"] / (hr_data["TotalWorkingYears"] + 1)
    hr_data["PromotionRate"] = hr_data["YearsSinceLastPromotion"] / (hr_data["YearsAtCompany"] + 1)
    hr_data["ManagerChangeRate"] = hr_data["YearsAtCompany"] / (hr_data["YearsWithCurrManager"] + 1)
    hr_data["SatisfactionScore"] = (hr_data["JobSatisfaction"] + hr_data["EnvironmentSatisfaction"] + hr_data["WorkLifeBalance"]) / 3
    hr_data["SalarySatisfactionGap"] = hr_data["MonthlyIncome"] / (hr_data["JobSatisfaction"] + 1)
    hr_data["PerformanceInvolvementGap"] = hr_data["PerformanceRating"] - hr_data["JobInvolvement"]
    hr_data["AbsenceRate"] = hr_data["AbsenceDays"] / (hr_data["YearsAtCompany"] + 1)
    hr_data["TravelFatigue"] = hr_data["BusinessTravel"] * hr_data["DistanceFromHome"]

    # Normalisation des colonnes pour la matrice de corrélation
    categorical_columns = ['Department', 'EducationField', 'JobRole']
    binary_columns = ['Attrition', 'Gender']
    numerical_columns = hr_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in categorical_columns + binary_columns]
    scaler = MinMaxScaler()
    normalized_df = hr_data.copy()
    normalized_df[numerical_columns] = scaler.fit_transform(hr_data[numerical_columns])

    return hr_data, absence_status, absence_days, normalized_df

def display_metric(label, value, low_threshold, high_threshold):
    """
    Affiche un KPI avec un indicateur visuel selon le seuil bas et haut.
    """
    if value < low_threshold:
        status = "🔴 Mauvais"
    elif value < high_threshold:
        status = "🟡 Moyen"
    else:
        status = "🟢 Bon"
    st.metric(label, f"{value:.2f}", status)

def display_model_results(model, X_test, y_test, y_pred, y_proba, model_name):
    """
    Affiche les résultats d’un modèle incluant :
        - Statistiques principales (accuracy, recall, F1)
        - Matrice de confusion
        - Courbe ROC
        - Importance des variables
    """
    st.title(f"📊 Analyse de l'Attrition - {model_name}")
    
    # Statistiques du modèle
    st.subheader("📌 Statistiques du Modèle")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Précision (Accuracy)", f"{accuracy_score(y_test, y_pred) * 100:.2f} %")
    with col2:
        report = classification_report(y_test, y_pred, output_dict=True)
        st.metric("🎯 Rappel (Recall)", f"{report['1']['recall']:.2f}")
    with col3:
        st.metric("✅ Score F1", f"{report['1']['f1-score']:.2f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm[::-1],
        x=["Positive predicted value", "Negative predicted value"],
        y=["Negative actual value", "Positive actual value"],
        colorscale="RdBu",
        annotation_text=cm[::-1].astype(str),
        showscale=True,
        reversescale=True
    )
    st.subheader("📊 Matrice de confusion")
    fig_cm.update_layout(xaxis=dict(title="Prédiction"), yaxis=dict(title="Réel"))
    st.plotly_chart(fig_cm, use_container_width=True, key=f"confusion_matrix_{model_name}")
    
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.2f})",
                   line=dict(color="darkorange", width=2))
    )
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Modèle aléatoire",
                   line=dict(color="navy", dash="dash"))
    )
    st.subheader("📉 Courbe ROC")
    st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_curve_{model_name}")
    
    # Importance des variables
    if model_name == "Random Forest":
        importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = importances.importances_mean.argsort()
        st.subheader("📈 Importance des Variables")
        fig_importance = go.Figure()
        fig_importance.add_trace(
            go.Bar(x=importances.importances_mean[sorted_idx], y=X_test.columns[sorted_idx],
                   orientation="h", marker=dict(color="royalblue"))
        )
        fig_importance.update_layout(title="Importance des Variables",
                                     xaxis_title="Importance",
                                     yaxis_title="Variables")
        st.plotly_chart(fig_importance, use_container_width=True, key=f"importance_{model_name}")
    # Autres modèles
    else:
        st.warning("ℹ️ L'importance des variables n'est disponible que pour le modèle Random Forest.")
        

def prepare_data_model(df, features, target, encode_cols=None, scaler=StandardScaler()):
    """
    Prépare les données pour l'entraînement d'un modèle.
    
    Paramètres :
        df         : DataFrame source
        features   : liste des colonnes à utiliser comme features
        target     : nom de la colonne cible
        encode_cols: liste des colonnes à encoder
        scaler     : objet scaler pour normaliser les variables numériques
    
    Retourne :
        X_train, X_test, y_train, y_test après traitement
    """
    df_model = df[features + [target]].copy()
    if encode_cols:
        df_model = pd.get_dummies(df_model, columns=encode_cols, drop_first=True)
    num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    df_model[num_cols] = scaler.fit_transform(df_model[num_cols])
    X = df_model.drop(columns=[target])
    y = df_model[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------------------------------------------------------
# Chargement des données
# -----------------------------------------------------------------------------
df, absence_status, absence_days, normalized_df = load_data()

# Création des onglets principaux
page1, page2, page3, page4, page5, page6 = st.tabs([
    "Accueil",
    "Analyse Univariée",
    "Analyse Bivariée & Multivariée",
    "Analyse Avancée & Business Insights",
    "Modélisation & Prédiction",
    "🔮 Aide à la Décision"
])

# -----------------------------------------------------------------------------
# Page 1 : Accueil
# -----------------------------------------------------------------------------
with page1:
    st.title("📊 HumanForYou - Dashboard")
    
    st.subheader("📌 Statistiques Clés")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🌍 Nombre total d'employés", df.shape[0])
        st.metric("🚀 Taux d'attrition", f"{df['Attrition'].mean() * 100:.2f} %")
        st.metric("📊 Absence moyenne par employé", f"{absence_days['AbsenceDays'].mean():.1f} jours")
    with col2:
        st.metric("📈 Salaire moyen", f"₹{df['MonthlyIncome'].mean():,.2f}")
        st.metric("📅 Ancienneté moyenne", f"{df['YearsAtCompany'].mean():.1f} ans")
    with col3:
        st.metric("👨‍💼 % Hommes", f"{df[df['Gender'] == 1].shape[0] / df.shape[0] * 100:.1f} %")
        st.metric("👩 % Femmes", f"{df[df['Gender'] == 0].shape[0] / df.shape[0] * 100:.1f} %")
    
    st.subheader("📌 Indicateurs de Performance et de Satisfaction")
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
        display_metric("✈️ Fatigue liée au Voyage", df['TravelFatigue'].mean(), 10, 30)

# -----------------------------------------------------------------------------
# Page 2 : Analyse Univariée
# -----------------------------------------------------------------------------
with page2:
    st.title("📊 Analyse des Données")
    tab1, tab2, tab3 = st.tabs([
        "📈 Statistiques détaillées",
        "📊 Graphiques",
        "📁 Données brutes",
    ])
    
    # Onglet 1 : Statistiques détaillées
    with tab1:
        st.markdown("## 📊 Analyse Univariée")
        st.markdown("#### Exploration des statistiques et répartition des données")

        st.subheader("📌 Statistiques Générales")
        col1, col2 = st.columns([1, 2])
        with col1:
            info_dict = {
                "Column": df.columns,
                "Non-Null Count": df.count().values,
                "Dtype": [df[col].dtype for col in df.columns]
            }
            df_info = pd.DataFrame(info_dict)
            st.dataframe(df_info, height=500)
        with col2:
            st.dataframe(df.describe(), height=300)

        st.markdown("---")
        
        st.subheader("🏢 Répartition des employés par département")
        department_counts = df['Department'].value_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(department_counts)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(
            x=department_counts.values,
            y=department_counts.index,
            hue=department_counts.index,  # Assigner la variable 'y' à hue
            palette="Blues_r",
            ax=ax,
            dodge=False )
            ax.set_xlabel("Nombre d'employés")
            ax.set_ylabel("Département")
            ax.set_title("📊 Répartition par Département")
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("🚨 Gestion des valeurs manquantes")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
        if missing_values.empty:
            st.success("✅ Aucune valeur manquante détectée !")
        else:
            st.warning("⚠️ Certaines colonnes contiennent des valeurs manquantes.")
            st.subheader("📉 Distribution des valeurs manquantes")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette="Reds", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel("Nombre de valeurs manquantes")
            ax.set_title("🔍 Colonnes concernées")
            st.pyplot(fig)
            
            st.subheader("🗺️ Heatmap des valeurs manquantes")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cmap="Reds", cbar=False, yticklabels=False, ax=ax)
            ax.set_title("🔍 Heatmap des valeurs manquantes")
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.subheader("📊 Distribution des âges")
        st.write("Répartition des âges des employés : 🔴 18-25, 🔵 26-35, 🟢 36-45, 🟡 46-55, 🟣 56-65")
        age_bins = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], right=False)
        age_distribution = age_bins.value_counts().sort_index()
        age_distribution.index = age_distribution.index.astype(str).str.replace('[', '').str.replace(')', '').str.replace(',', ' -')
        st.bar_chart(age_distribution)
        
        st.markdown("---")
        
        st.subheader("📈 Répartition des années d'ancienneté")
        st.bar_chart(df['YearsAtCompany'].value_counts())
        
        st.markdown("---")
        
        st.subheader("💰 Répartition des salaires par tranche")
        salary_bins = pd.cut(df['MonthlyIncome'], bins=5)
        salary_distribution = salary_bins.value_counts().sort_index()
        salary_distribution.index = salary_distribution.index.astype(str).str.replace('(', '').str.replace(']', '').str.replace(',', ' -')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("Distribution des salaires :")
            st.dataframe(salary_distribution)
        with col2:
            st.bar_chart(salary_distribution)
        
        st.markdown("---")
        
        satisfaction_mapping = {
            'EnvironmentSatisfaction': "Satisfaction de l'environnement de travail",
            'JobSatisfaction': "Satisfaction du travail",
            'WorkLifeBalance': "Équilibre travail-vie personnelle"
        }
        st.subheader("😀 Satisfaction des employés")
        satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
        for col in satisfaction_cols:
            st.write(f"### 📊 {satisfaction_mapping[col]}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("Répartition des niveaux :")
                st.dataframe(df[col].value_counts().rename_axis("Niveau").reset_index(name="Nombre d'employés"))
            with col2:
                st.write("Distribution graphique :")
                st.bar_chart(df[col].value_counts())
        st.markdown("---")
    
    # Onglet 2 : Graphiques
    with tab2:
        st.subheader("Visualisation Dynamique des Variables")
        selected_var = st.selectbox("Sélectionnez une variable numérique :", df.select_dtypes(include=['int64', 'float64']).columns.tolist())
        # Visualisation dynamique de la variable sélectionnée
        st.subheader("Visualisation de la variable sélectionnée : **{}**".format(selected_var))

        # Histogramme avec distribution et KDE
        st.markdown("**Histogramme avec distribution et KDE**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(df[selected_var], kde=True, color="royalblue", ax=ax)
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

        # Boxplot
        st.markdown("**Boxplot**")
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x=df[selected_var], color="lightcoral", ax=ax)
        ax.set_xlabel(selected_var)
        st.pyplot(fig)

        # Courbe de densité (KDE)
        st.markdown("**Courbe de densité (KDE)**")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.kdeplot(df[selected_var], fill=True, color="green", ax=ax)
        ax.set_xlabel(selected_var)
        ax.set_ylabel("Densité")
        st.pyplot(fig)
        st.markdown("---")
    
    # Onglet 3 : Données brutes
    with tab3:
        st.subheader("📂 Aperçu des données")
        st.dataframe(df.head(10))

# -----------------------------------------------------------------------------
# Page 3 : Analyse Bivariée & Multivariée
# -----------------------------------------------------------------------------
with page3:
    with st.expander("🔎 Options d'analyse", expanded=False):
        selected_features = st.multiselect(
            "Sélectionnez les variables pour la matrice de corrélation :",
            df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            default=["Attrition","JobLevel", "YearsAtCompany", "YearsWithCurrManager",
                     "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
                     "PercentSalaryHike", "StockOptionLevel", "JobSatisfaction", "WorkLifeBalance",
                     "EnvironmentSatisfaction", "TrainingTimesLastYear", "BusinessTravel",
                     "DistanceFromHome", "AbsenceDays", "TotalWorkingYears",
                     "PerformanceRating", "JobInvolvement"]
        )
    
    st.subheader("📌 Matrice de Corrélation")
    correlation_matrix = normalized_df[selected_features].corr()
    # Inversion pour une meilleure orientation
    correlation_matrix = correlation_matrix.iloc[::-1]
    fig_corr = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index)[::-1],
        colorscale="RdBu",
        annotation_text=np.round(correlation_matrix.values, 2),
        showscale=True,
        reversescale=True
    )
    fig_corr.update_layout(
        title="Matrice de Corrélation",
        xaxis=dict(title="Variables"),
        yaxis=dict(title="Variables"),
        margin=dict(l=100, r=100, t=50, b=50),
        height=700
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.subheader("📌 Comparaison des Employés Partis vs. Restants")
    # Transformation des variables pour une meilleure lisibilité
    df_comparison = df.copy()
    df_comparison["Gender"] = df_comparison["Gender"].map({1: "Homme", 0: "Femme"})
    df_comparison["MaritalStatus"] = df_comparison["MaritalStatus"].map({0: "Célibataire", 1: "Marié", 2: "Divorcé"})
    
    attrition_comparison = {
        "💰 Salaire Moyen": df_comparison.groupby("Attrition")["MonthlyIncome"].mean(),
        "🏢 Années dans l'Entreprise": df_comparison.groupby("Attrition")["YearsAtCompany"].mean(),
        "🚀 Dernière Augmentation (%)": df_comparison.groupby("Attrition")["PercentSalaryHike"].mean(),
        "🔄 Nombre d'Entreprises Précédentes": df_comparison.groupby("Attrition")["NumCompaniesWorked"].mean(),
        "📈 Niveau Hiérarchique": df_comparison.groupby("Attrition")["JobLevel"].mean(),
        "🏠 Distance Domicile-Travail": df_comparison.groupby("Attrition")["DistanceFromHome"].mean(),
        "📊 Score Satisfaction": df_comparison.groupby("Attrition")["SatisfactionScore"].mean(),
        "📈 Taux de Promotion": df_comparison.groupby("Attrition")["PromotionRate"].mean(),
        "🚪 Taux d'Absence": df_comparison.groupby("Attrition")["AbsenceRate"].mean()
    }
    
    option = st.selectbox("Choisissez un critère :", list(attrition_comparison.keys()))
    st.subheader(option)
    st.bar_chart(attrition_comparison[option])
    
    st.subheader("Interprétation des Résultats")
    col1, col2 = st.columns(2)
    with col1:
        st.write("📌 **Employés Partis (Attrition = 1)**")
        st.write(f"➡️ Moyenne d'âge : {df_comparison[df_comparison['Attrition'] == 1]['Age'].mean():.1f} ans")
        st.write(f"➡️ Salaire moyen : ₹{df_comparison[df_comparison['Attrition'] == 1]['MonthlyIncome'].mean():,.2f}")
    with col2:
        st.write("📌 **Employés Restants (Attrition = 0)**")
        st.write(f"➡️ Années moyennes dans l'entreprise : {df_comparison[df_comparison['Attrition'] == 0]['YearsAtCompany'].mean():.1f} ans")
        st.write(f"➡️ Satisfaction moyenne : {df_comparison[df_comparison['Attrition'] == 0]['JobSatisfaction'].mean():.1f} / 4")

# -----------------------------------------------------------------------------
# Page 4 : Analyse Avancée & Business Insights
# -----------------------------------------------------------------------------
with page4:
    st.title("Analyse Avancée & Business Insights")
    st.write("Ici aymane")

# -----------------------------------------------------------------------------
# Page 5 : Prédiction
# -----------------------------------------------------------------------------
with page5:
    tab_lr, tab_svm, tab_rf, tab_hyper = st.tabs([
        "📊 Régression Logistique",
        "🧠 SVM",
        "🌲 Random Forest",
        "⚙️ Hyperparamètres"
            ])
    
    # Variables pour la modélisation
    features = [
        "JobRole", "JobLevel", "YearsAtCompany", "YearsWithCurrManager",
        "YearsSinceLastPromotion", "NumCompaniesWorked", "MonthlyIncome",
        "PercentSalaryHike", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
        "TrainingTimesLastYear", "BusinessTravel", "AbsenceDays", "TotalWorkingYears",
        "Department"
    ]
    target = "Attrition"
    
    # --- Onglet Régression Logistique ---
    with tab_lr:
        st.subheader("Régression Logistique")
        # Encodage des variables catégorielles 
        encode_cols = ["JobRole", "BusinessTravel", "Department"]
        X_train, X_test, y_train, y_test = prepare_data_model(
            df, features, target, encode_cols=encode_cols, scaler=StandardScaler()
        )
        
        # Entraînement du modèle
        lr_model = LogisticRegression(max_iter=500)
        lr_model.fit(X_train, y_train)
        
        # Prédiction et ajustement du seuil
        y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
        threshold = 0.35  # Seuil ajusté
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Affichage des résultats
        display_model_results(lr_model, X_test, y_test, y_pred, y_pred_proba, "Régression Logistique")
        results_logistic = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Onglet SVM ---
    with tab_svm:
        st.subheader("Support Vector Machine (SVM)")
        # Encodage et normalisation pour SVM
        encode_cols = ["JobRole", "Department", "BusinessTravel"]
        X_train, X_test, y_train, y_test = prepare_data_model(
            df, features, target, encode_cols=encode_cols, scaler=StandardScaler()
        )
        
        svm_model = SVC(probability=True, random_state=42, class_weight='balanced', kernel='rbf', C=1, gamma=0.1)
        svm_model.fit(X_train, y_train)
        
        y_pred = svm_model.predict(X_test)
        y_proba = svm_model.predict_proba(X_test)[:, 1]
        
        display_model_results(svm_model, X_test, y_test, y_pred, y_proba, "SVM")
        results_svm = classification_report(y_test, y_pred, output_dict=True)
    
    # --- Onglet Random Forest ---
    with tab_rf:
        st.subheader("Random Forest")
        # Préparation des données avec encodage
        df_encoded = pd.get_dummies(df[features], drop_first=True)
        X = df_encoded
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        
        rf_pred = rf_model.predict(X_test)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        
        display_model_results(rf_model, X_test, y_test, rf_pred, rf_proba, "Random Forest")
        results_rf = classification_report(y_test, rf_pred, output_dict=True)
    # --- Onglet hyperparamètres ---
    with tab_hyper:
        st.subheader("Recherche d'hyperparamètres pour les 3 modèles")
        
        # Sélection de la méthode de recherche (grid ou random)
        search_method = st.selectbox(
            "Méthode de recherche :",
            ["random", "grid"]
        )
        
        # Dictionnaire qui contiendra les résultats de chaque modèle
        results_dict = {}

        ############################################################################
        # 1. Régression Logistique
        ############################################################################
        st.markdown("### Régression Logistique")
        grid_params_lr = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver" : ["liblinear", "saga", "lbfgs"],
            "max_iter": [100, 200, 300, 400, 500]
        }
        encode_cols_lr = ["JobRole", "BusinessTravel", "Department"]
        # Préparation des données avec prepare_data_model() pour LR
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = prepare_data_model(
            df, features, target, encode_cols=encode_cols_lr, scaler=StandardScaler()
        )
        model_lr = LogisticRegression(random_state=42)
        
        if search_method == "random":
            search_lr = RandomizedSearchCV(
                estimator=model_lr,
                param_distributions=grid_params_lr,
                n_iter=50,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1,
                random_state=42
            )
        else:
            search_lr = GridSearchCV(
                estimator=model_lr,
                param_grid=grid_params_lr,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1
            )
        search_lr.fit(X_train_lr, y_train_lr)
        st.write("**Régression Logistique – Meilleurs paramètres :**")
        st.write(search_lr.best_params_)
        st.write("**Régression Logistique – Meilleur F1 Score :**")
        st.write(f"{search_lr.best_score_ * 100:.2f} %")
        results_dict["Régression Logistique"] = {
            "best_params": search_lr.best_params_,
            "best_score": search_lr.best_score_
        }
        
        ############################################################################
        # 2. SVM
        ############################################################################
        st.markdown("### SVM")
        grid_params_svm = {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": [1e-3, 1e-2, 1e-1, 1],
            "kernel": ["rbf"]
        }
        encode_cols_svm = ["JobRole", "Department", "BusinessTravel"]
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = prepare_data_model(
            df, features, target, encode_cols=encode_cols_svm, scaler=StandardScaler()
        )
        model_svm = SVC(probability=True, random_state=42, class_weight='balanced')
        
        if search_method == "random":
            search_svm = RandomizedSearchCV(
                estimator=model_svm,
                param_distributions=grid_params_svm,
                n_iter=50,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1,
                random_state=42
            )
        else:
            search_svm = GridSearchCV(
                estimator=model_svm,
                param_grid=grid_params_svm,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1
            )
        search_svm.fit(X_train_svm, y_train_svm)
        st.write("**SVM – Meilleurs paramètres :**")
        st.write(search_svm.best_params_)
        st.write("**SVM – Meilleur F1 Score :**")
        st.write(f"{search_svm.best_score_ * 100:.2f} %")
        results_dict["SVM"] = {
            "best_params": search_svm.best_params_,
            "best_score": search_svm.best_score_
        }
        
        ############################################################################
        # 3. Random Forest
        ############################################################################
        st.markdown("### Random Forest")
        grid_params_rf = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
            "class_weight": ["balanced", None]
        }
        df_encoded_rf = pd.get_dummies(df[features], drop_first=True)
        X_rf = df_encoded_rf
        y_rf = df[target]
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
            X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
        )
        model_rf = RandomForestClassifier(random_state=42)
        
        if search_method == "random":
            search_rf = RandomizedSearchCV(
                estimator=model_rf,
                param_distributions=grid_params_rf,
                n_iter=100,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1,
                random_state=42
            )
        else:
            search_rf = GridSearchCV(
                estimator=model_rf,
                param_grid=grid_params_rf,
                cv=5,
                scoring={'f1': 'f1', 'accuracy': 'accuracy', 'recall': 'recall'},
                refit='f1',
                n_jobs=-1
            )
        search_rf.fit(X_train_rf, y_train_rf)
        st.write("**Random Forest – Meilleurs paramètres :**")
        st.write(search_rf.best_params_)
        st.write("**Random Forest – Meilleur F1 Score :**")
        st.write(f"{search_rf.best_score_ * 100:.2f} %")
        results_dict["Random Forest"] = {
            "best_params": search_rf.best_params_,
            "best_score": search_rf.best_score_
        }
        
        ############################################################################
        # Affichage du récapitulatif pour les 3 modèles
        ############################################################################
        st.subheader("Récapitulatif des résultats")
        df_results = pd.DataFrame.from_dict(results_dict, orient='index')
        df_results.index.name = "Modèle"
        st.dataframe(df_results)

        
        st.subheader("Comparaison des modèles")

        # Appliquer le thème Streamlit via Seaborn
        sns.set_theme(style="whitegrid", palette="viridis")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=df_results.index,
            y=df_results["best_score"] * 100,
            ax=ax
        )
        ax.set_ylabel("F1 Score (%)")
        ax.set_title("📊 Comparaison des modèles")
        st.pyplot(fig)
        st.success(f"Le meilleur modèle est {df_results['best_score'].idxmax()} avec un F1 Score de {df_results['best_score'].max() * 100:.2f} %")
# -----------------------------------------------------------------------------
# Page 6 : Predire un départ d'employé
# -----------------------------------------------------------------------------
with page6:
    st.title("🔮 Aide à la Décision")