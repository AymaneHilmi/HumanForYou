
# Import des bibliothèques principales
import pandas as pd
import streamlit as st


# In[105]:


# Charger les données
hr_data = pd.read_csv('./data/general_data.csv')
survey_data = pd.read_csv('./data/employee_survey_data.csv')
manager_data = pd.read_csv('./data/manager_survey_data.csv')
in_time_data = pd.read_csv('./data/in_time.csv')
out_time_data = pd.read_csv('./data/out_time.csv')

# Supprimer la colonne EmployeeCount, StandardHours, Over18
hr_data.drop(['EmployeeCount', 'StandardHours', 'Over18'], axis=1, inplace=True)

# Supprimer les lignes pour NumCompaniesWorked vide 
hr_data.dropna(subset=['NumCompaniesWorked'], inplace=True)

# Supprimer les lignes pour TotalWorkingYears vide
hr_data.dropna(subset=['TotalWorkingYears'], inplace=True)


# In[106]:


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


####### Ne pas lancer cette section avec Jupyter #######

# Moyenne d'age des employés qui ont quitté l'entreprise
####### Ne pas lancer cette section avec Jupyter #######

# Moyenne d'age des employés qui ont quitté l'entreprise
st.title('Analyse des données RH')
st.write('Moyenne d\'age des employés qui ont quitté l\'entreprise')
st.write(hr_data[hr_data['Attrition'] == 1]['Age'].mean())

# Moyenne d'age des employés qui sont restés dans l'entreprise
st.write('Moyenne d\'age des employés qui sont restés dans l\'entreprise')
st.write(hr_data[hr_data['Attrition'] == 0]['Age'].mean())

# Repartition hommes/femmes
st.write('Repartition hommes/femmes')
st.write(hr_data[hr_data['Gender'] == 1].shape[0]/hr_data.shape[0] * 100 , '% d\'hommes')

# Taux de satisfaction en fonction du niveau d'implication
st.write('Taux de satisfaction en fonction du niveau d\'implication')

# Histogramme de l'age des employés
st.write('Histogramme de l\'age des employés')
st.bar_chart(hr_data['Age'])


# Calculer la matrice de corrélation uniquement pour les colonnes numériques (utiliser les donnees normalisées)
normalized_data = pd.read_csv('merged_data_normalized.csv')
numeric_data = normalized_data.select_dtypes(include=['int', 'float'])
correlation_matrix = numeric_data.corr()

# Matrice de correlation des données
st.write('Matrice de correlation des données')
st.write(correlation_matrix)