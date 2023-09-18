# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

sns.set_theme(style="whitegrid")
df = pd.read_csv("heart.csv")

fig, axes = plt.subplots(3, 3, figsize=(18, 10))

# Output gender relationship
output_gender_relationship = pd.concat([df["sex"],df["output"]],axis=1 )

male_output = output_gender_relationship.loc[output_gender_relationship["sex"] == 1]
female_output = output_gender_relationship.loc[output_gender_relationship["sex"] == 0]

positive_results_on_male_outputs = male_output[male_output["output"] == 1]
positive_results_on_female_outputs = female_output[female_output["output"] == 1]

male_val = positive_results_on_male_outputs.count(axis = 0)[0]
female_val = positive_results_on_female_outputs.count(axis = 0)[0]


# output age relationship
output_age_relationship = pd.concat([df["sex"],df["age"],df["output"]],axis=1 )

male_output = output_age_relationship.loc[output_age_relationship["sex"] == 1]
female_output = output_age_relationship.loc[output_age_relationship["sex"] == 0]

# cofiguration of male dataset
male_output_hist_positive = male_output.loc[male_output["output"] == 1]
male_output_hist_positive = male_output_hist_positive.groupby("age").size()
male_output_hist_positive = pd.DataFrame({'age':male_output_hist_positive.index, 'output':male_output_hist_positive.values})
# configuration of female dataset

female_output_hist_positive = female_output.loc[female_output["output"] == 1]
female_output_hist_positive = female_output.groupby("age").size()
female_output_hist_positive = pd.DataFrame({'age':female_output_hist_positive.index, 'output':female_output_hist_positive.values})

female_output_hist_positive['gender']= "female"
male_output_hist_positive['gender']= "male"

res=pd.concat([male_output_hist_positive,female_output_hist_positive])

chol = pd.concat([df["chol"],df["output"]],axis=1 )
chest_pain =pd.concat([df["cp"],df["output"]],axis=1 )

sns.barplot(ax=axes[0, 0], data=df, x='sex', y='age', hue = "cp")
sns.lineplot(ax=axes[0, 1], data=df , x='cp', y='age', hue = "sex")
sns.boxplot(ax=axes[0, 2], data=df, x='sex', y='age', hue = "output")

sns.boxplot(ax=axes[1, 0], data=df, x='sex', y='chol', hue = "output")
sns.scatterplot(ax=axes[1, 1], data=df, x='age', y='chol', hue = "output", size = "cp")
sns.boxplot(ax=axes[1, 2], data=df, x='cp', y='chol', hue = "sex")

sns.violinplot(ax=axes[2, 0], data=df, x='cp', y='trtbps', hue = "sex",split=True, inner="quart")
sns.lineplot(ax=axes[2, 1], data=df, x='fbs', y='trtbps', hue = "output")
sns.histplot(ax=axes[2, 2], data=df, x='chol', y='trtbps', hue = "output", multiple="stack", palette="light:m_r")

plt.show()

X = df.drop(['output'],axis=1)
Y = df['output']

# loading data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
Dtc = DecisionTreeClassifier()
Rfc = RandomForestClassifier()
Knn = KNeighborsClassifier()

# fitting the models
Dtc.fit(X_train,Y_train)
Rfc.fit(X_train,Y_train)
Knn.fit(X_train,Y_train)

# Predictions
cross_validation_Dtc = cross_val_score(Dtc,X_train,Y_train,cv = 5, scoring = 'accuracy')
cross_validation_Rfc = cross_val_score(Rfc,X_train,Y_train,cv = 5, scoring = 'accuracy')
cross_validation_Knn = cross_val_score(Knn,X_train,Y_train,cv = 5, scoring = 'accuracy')

print("Decision Tree Classifier Cross Validation Scores:")
print(cross_validation_Dtc)
print("Random Forest Classifier Cross Validation Scores:")
print(cross_validation_Rfc)
print("K Neighbors Classifier Cross Validation Scores:")
print(cross_validation_Knn)

y_pred_dtc = Dtc.predict(X_test)
y_pred_rfc = Rfc.predict(X_test)
y_pred_knn = Knn.predict(X_test)

accuracy_dtc = accuracy_score(Y_test, y_pred_knn)
accuracy_rfc = accuracy_score(Y_test, y_pred_rfc)
accuracy_knn = accuracy_score(Y_test, y_pred_dtc)


print("Decision Tree Classifier , accuracy score: ")
print(accuracy_dtc)

print("Random Fores Classifier , accuracy score: ")
print(accuracy_rfc)

print("K Neighbor Classifier , accuracy score: ")
print(accuracy_knn)
