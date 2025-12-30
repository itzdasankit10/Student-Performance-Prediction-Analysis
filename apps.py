import numpy as np
import matplotlib.pyplot as pp
import pandas as pb
import seaborn as ss
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pb.read_csv("students.csv")

##DELETING SL.NO UNNAMED COLUMN
df = df.drop("Unnamed: 0", axis=1)
print(df.head())

##CHANGING WEEKLY STUDY HOURS
df["WklyStudyHours"] = df["WklyStudyHours"].str.replace("05-Oct", "5-10")
print(df["WklyStudyHours"].head(50))

##DRAWING THE GRAPH OF GENDER VALUES & GOT TO KNOW THAT NO FEMALE > NO OF MALES
pp.figure(figsize=(4,4))
ax  = ss.countplot(data = df, x = "Gender")
ax.bar_label(ax.containers[0 ])
pp.show()

##PARENT EDUCATION IMPACT ON SCORES
group = df.groupby("ParentEduc").aggregate({"MathScore":'mean', "ReadingScore": 'mean', "WritingScore":'mean'})
print(group)

##HIGHER THE DEGREE OF PARENTS BETTER THE MARKS
pp.figure(figsize=(14,4))
ss.heatmap(group, annot=True)
pp.title("PARENT'S EDUCATION IMPACT")
pp.show()

## ANALYSING DATA BY PARENT MARITAL STATUS
group = df.groupby("ParentMaritalStatus").agg({"MathScore":"mean", "ReadingScore":"mean", "WritingScore":"mean"})
print(group)

##FROM HERE WE GET THAT NOT MUCH EFFECT IS THERE ON STUDENT'S SCORE BASED ON PARENT'S MARITAL STATUS
pp.figure(figsize=(14,4))
ss.heatmap(group, annot=True)
pp.title("PARENT'S MARITAL STATUS IMPACT")
pp.show()


##ANALYSING DATA BY CHECKING HOW MUCH SPORTS ACTIVITES AFFECT THE STUDENT'S
group = df.groupby("PracticeSport").agg({"MathScore":"mean", "ReadingScore":"mean", "WritingScore":"mean"})
print(group)


##VERY LITTLE EFFECT IS SHOWN BY SPORTS  ACTIVITES
pp.figure(figsize=(14,4))
ss.heatmap(group, annot=True)
pp.title("SPORT ACTIVITES IMPACT")
pp.show()


##ANALYSING DATA BY CHECKING IMPACT OF WEEKLY STUDY HOURS OVER CHILDREN
group = df.groupby("WklyStudyHours").agg({"MathScore":"mean", "ReadingScore":"mean", "WritingScore":"mean"})
print(group)

## STUDY HOURS IS EFFECTING QUITE A LOT FOR THE CHILDREN
pp.figure(figsize=(13,4))
ss.heatmap(group, annot=True)
pp.title("WEEKLY STUDY HOURS EFFECT")
pp.show()


print(df.isnull().sum())

##HANDLING THE MISSING DATAS
columns = ["EthnicGroup", "ParentEduc", "TestPrep", "ParentMaritalStatus", "PracticeSport", "IsFirstChild", 
           "TransportMeans", "WklyStudyHours"]
df[columns] = df[columns].fillna("Unknown")

df["NrSiblings"] = df["NrSiblings"].fillna(df["NrSiblings"].median())
print(df["WklyStudyHours"].value_counts())

##PERFORMANCE LABELS WILL NOW BE DONE
df["AvgScore"] = (df["MathScore"]+df["ReadingScore"]+df["WritingScore"]) / 3
print(df["AvgScore"])

def LABEL(score):
    if score<40:
        return "LOW"
    elif score<=75:
        return "Average"
    else:
        return "High"
    
df["Performance"] = df["AvgScore"].apply(LABEL)
print(df["Performance"].value_counts())

##DROPPING THE FEATURES USED AND OTHER FEATURES TO CONVERT THEM BY ENCODING
drops = df.drop(["MathScore", "ReadingScore", "WritingScore", "AvgScore", "Performance"], axis=1)
de = df["Performance"]
drops = pb.get_dummies(drops, drop_first=True)
print(drops.shape)
print(de.shape)

##SPLITING DATA INTO TRAIN-TEST
a_train, a_test, b_train, b_test = train_test_split(drops, de, test_size=0.2, random_state=42, stratify=de)
print(b_train.value_counts())
print(b_test.value_counts())


##IMPLEMENTING LOGISTIC REGRESSION TO TRAIN THE DATASET
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(a_train, b_train)
y_pred = model.predict(a_test)
print(classification_report(b_test, y_pred))

param_grid = {
    "C":[0.01, 0.1, 1, 5, 10],
    "solver":["lbfgs"] 
}

##USING CROSS-VALIDATION TECHNIQUE FOR HYPERPARAMETER TUNING USING GRID-SEARCH
grid = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=1
)

grid.fit(a_train, b_train)
best_model = grid.best_estimator_
print("THE BEST PARAMETERS = ", grid.best_params_)

##EVALUATION OF THE TUNED MODEL
y_pred = best_model.predict(a_test)
print("FOR LOGISTIC REGRESSION-------------")
print(classification_report(b_test, y_pred))


## AS DATA RELATIONSHIPS ARE NON-LINEAR IN NATURE SO LOGISTIC REGRESSION CAN NOT DELIVER MORE ACCURATE RESULTS

## WILL BE USING RANDOM FOREST FOR NON-LINEAR RELATIONSHIP
random= RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced", 
    max_depth=20,
    min_samples_leaf=20,
    n_jobs=1
)
random.fit(a_train, b_train)

prediction = random.predict(a_test)
print("FOR RANDOM FOREST--------------")
print(classification_report(b_test, prediction))

##SAVING THE ML MODEL
jb.dump(random, "model.pkl")
jb.dump(drops.columns, "model_cols.pkl")