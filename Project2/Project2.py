import numpy as np
import pandas as pd

df = pd.read_csv("Project2/Titanic.csv")

age_median = df['Age'].median()

df['Age'] = df['Age'].fillna(age_median)

embarked_mode = df['Embarked'].mode()[0]
# Why the 0? Because mode returns a series and you just need the first value

df['Embarked'] = df['Embarked'].fillna(embarked_mode)

df = df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)

empty_columns = df.isnull().sum()

tot_survival_rate = df['Survived'].sum() / len(df)
# len(df) is the count for the length of df in rows/passengers

pclass_survival_rate = df.groupby('Pclass')['Survived'].mean()
# First grouping by class and then only calculating the survived column

gender_survival_rate = df.groupby('Sex')['Survived'].mean()

pclass_avg_fare = df.groupby('Pclass')['Fare'].mean()

survival_age_distribution = df.groupby('Survived')['Age'].mean()

age_means = np.mean(df['Age'])
fare_means = np.mean(df['Fare'])

age_medians = np.median(df[['Age']])
fare_medians = np.median(df[['Fare']])

age_standard_devs = np.std(df[['Age']])
fare_standard_devs = np.std(df[['Fare']])

summary = pd.DataFrame({
    'Metric' : ['Total Survival Rate', 'Class 1 Survival', 'Class 2 Survival', 'Class 3 Survival', 
                'Female Survival','Male Survival', 'Class 1 Avg Fare', 'Class 2 Avg Fare', "Class 3 Avg Fare",
                  'Avg Age of Non-Survivors', 'Avg Age of Survivors'],
    'Value' : [tot_survival_rate, pclass_survival_rate[1], pclass_survival_rate[2], 
               pclass_survival_rate[3], gender_survival_rate['female'], gender_survival_rate['male'], 
               pclass_avg_fare[1], pclass_avg_fare[2], pclass_avg_fare[3], survival_age_distribution[0], 
               survival_age_distribution[1]]

})

summary.to_csv('titanic_summary.csv')

print(summary)


