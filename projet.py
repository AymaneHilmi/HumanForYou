#!/usr/bin/env python
# coding: utf-8

# In[96]:


# Age : Integer (Années)
# Attrition : Boolean
# BusinessTravel : Categorical (Non-Travel, Travel_Rarely, Travel_Frequently)
# DistanceFromHome : Integer (km)
# Education : Categorical (Avant College, College, Bachelor, Master, PhD)
# EducationField : Categorical (Domaines d’étude)
# EmployeeId : Integer (Identifiant unique)
# Gender : Boolean (Homme/Femme)
# JobLevel : Integer (1-5, niveau hiérarchique)
# JobRole : Categorical (Intitulé des postes)
# MaritalStatus : Categorical (Single, Married, Divorced)
# MonthlyIncome : Float (Salaire mensuel)
# NumCompaniesWorked : Integer (Nombre d’entreprises)
# PercentSalaryHike : Float (Pourcentage)
# StockOptionLevel : Integer (Niveau d’actions : 0, 1, 2, 3)
# TotalWorkingYears : Integer (Années)
# TrainingTimesLastYear : Integer (Jours)
# YearsAtCompany : Integer (Années)
# YearsSinceLastPromotion : Integer (Années)
# YearsWithCurrentManager : Integer (Années)
# JobInvolvement : Integer (1-4, Niveau d’implication)
# PerformanceRating : Integer (1-4, Niveau de performance)
# EnvironmentSatisfaction : Integer ou Null (1-4, Niveau de satisfaction, “NA” pour non-réponse)
# JobSatisfaction : Integer ou Null (1-4, Niveau de satisfaction, “NA” pour non-réponse)
# WorkLifeBalance : Integer ou Null (1-4, Niveau de satisfaction, “NA” pour non-réponse)


# In[97]:


# Import des bibliothèques principales
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


# In[98]:


# Charger les données
hr_data = pd.read_csv('general_data.csv')
survey_data = pd.read_csv('employee_survey_data.csv')
manager_data = pd.read_csv('manager_survey_data.csv')
in_time_data = pd.read_csv('in_time.csv')
out_time_data = pd.read_csv('out_time.csv')

# Aperçu des données
#print(in_time_data.head())
#print(out_time_data.head())

# Voir les donnees manquantes
print(hr_data.isnull().sum())
print(survey_data.isnull().sum())
print(manager_data.isnull().sum())

# Supprimer la colonne EmployeeCount, StandardHours, Over18
hr_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

# Supprimer les lignes pour NumCompaniesWorked vide 
hr_data.dropna(subset=['NumCompaniesWorked'], inplace=True)

# Supprimer les lignes pour TotalWorkingYears vide
hr_data.dropna(subset=['TotalWorkingYears'], inplace=True)


# In[99]:


# Renommer la première colonne pour qu'elle devienne "EmployeeID" dans in_time et out_time
in_time_data.rename(columns={"Unnamed: 0": 'EmployeeID'}, inplace=True)
out_time_data.rename(columns={"Unnamed: 0": 'EmployeeID'}, inplace=True)
# Definir le format en date de toutes les colonnes (type) pour in_time et out_time sauf la colonne EmployeeID
in_time_data.iloc[:, 1:] = in_time_data.iloc[:, 1:].apply(pd.to_datetime)
out_time_data.iloc[:, 1:] = out_time_data.iloc[:, 1:].apply(pd.to_datetime)

# Fusionner les données de chaque fichier
hr_data = hr_data.merge(survey_data, on='EmployeeID')
hr_data = hr_data.merge(manager_data, on='EmployeeID')
#hr_data = hr_data.merge(in_time_data, on='EmployeeID')
#hr_data = hr_data.merge(out_time_data, on='EmployeeID')

# Moyenne sans decimales de EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance pour les valeurs manquantes
hr_data['EnvironmentSatisfaction'].fillna(hr_data['EnvironmentSatisfaction'].mean().round(), inplace=True)
hr_data['JobSatisfaction'].fillna(hr_data['JobSatisfaction'].mean().round(), inplace=True)
hr_data['WorkLifeBalance'].fillna(hr_data['WorkLifeBalance'].mean().round(), inplace=True)

# Definir le format de chaque colonne (type)
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
hr_data['JobSatisfaction'] = hr_data['JobSatisfaction'].astype('category')
hr_data['WorkLifeBalance'] = hr_data['WorkLifeBalance'].astype('category')

# Sauvegarder les données fusionnées
hr_data.to_csv('merged_data.csv', index=False)


# In[100]:


# Verification du type de chaque colonne
print(hr_data.dtypes)

# Voir les donnees manquantes
print(hr_data.isnull().sum())
print(survey_data.isnull().sum())
print(manager_data.isnull().sum())


# In[101]:


# Moyenne d'age des employés qui ont quitté l'entreprise
st.title('Analyse des données RH')
st.write('Moyenne d\'age des employés qui ont quitté l\'entreprise')
st.write(hr_data[hr_data['Attrition'] == 1]['Age'].mean())

# Moyenne d'age des employés qui sont restés dans l'entreprise
st.write('Moyenne d\'age des employés qui sont restés dans l\'entreprise')
st.write(hr_data[hr_data['Attrition'] == 0]['Age'].mean())

# Repartition hommes/femmes
st.write('Repartition hommes/femmes')
st.write(hr_data[hr_data['Gender'] == 1].shape[0])

# Taux de satisfaction en fonction du niveau d'implication
st.write('Taux de satisfaction en fonction du niveau d\'implication')

# Histogramme de l'age des employés
st.write('Histogramme de l\'age des employés')
st.bar_chart(hr_data['Age'])


# Calculer la matrice de corrélation uniquement pour les colonnes numériques
numeric_data = hr_data.select_dtypes(include=['int', 'float'])
correlation_matrix = numeric_data.corr()

# Matrice de correlation des données
st.write('Matrice de correlation des données')
st.write(correlation_matrix)



# In[44]:


# Analyse univariée
# Afficher les répartitions des données

import pandas as pd
# enlever les warnings
import warnings
warnings.filterwarnings('ignore')
# Charger les données fusionnées
hr_data = pd.read_csv('merged_data.csv')

# Créer un dictionnaire pour stocker les statistiques
statistics = []

# Calculer les statistiques pour chaque colonne
def calculate_statistics(data, statistics):
    for column in data.columns:
        col_data = data[column]
        if col_data.dtypes == 'object':  # Colonnes catégorielles
            value_counts = col_data.value_counts().to_dict()
            statistics.append({
                'Column': column,
                'Type': 'Categorical',
                'Value Counts': value_counts
            })
        else:  # Colonnes numériques
            statistics.append({
                'Column': column,
                'Type': 'Numeric',
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'StdDev': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                'Unique Values': col_data.nunique()
            })

# Appeler la fonction de calcul des statistiques
calculate_statistics(hr_data, statistics)

# Convertir les statistiques en DataFrame pour l'export
statistics_df = pd.DataFrame(statistics)

# Sauvegarder les statistiques dans un fichier CSV
statistics_df.to_csv('statistics-data.csv', index=False)

# Afficher les statistiques dans la console
# print(statistics_df)


# In[45]:


# Caractéristiques des employés qui ont quitté l'entreprise
# Séparer les employés ayant quitté l'entreprise
attrition_yes = hr_data[hr_data['Attrition'] == 'Yes']
# Calculer les statistiques pour les employés ayant quitté l'entreprise
statistics_attrition = []
calculate_statistics(attrition_yes, statistics_attrition)
statistics_attrition_df = pd.DataFrame(statistics_attrition)
statistics_attrition_df.to_csv('statistics-attrition.csv', index=False)
#print(statistics_attrition_df)



