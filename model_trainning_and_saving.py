import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

df = df.drop_duplicates()

# replacing zero value
df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].median())
df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].median())
df['BMI'] = df['BMI'].replace(0,df['BMI'].median())
df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].median())
df['Age'] = df['Age'].replace(0,df['Age'].median())

# divide the dataset into independent and dependent dataset
X = df.drop(columns='Outcome')
y = df['Outcome']

data_columns = X.columns

numerical_pipeline = Pipeline(
        steps = [
            ('imputer',SimpleImputer(strategy = 'median')),
            ('scaler',StandardScaler())
        ]
)

preprocessor = ColumnTransformer([
    ('numerical_pipeline',numerical_pipeline,data_columns)
])

# splitting the data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 20)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

models = {
    'LogisticRegression':LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=200),

    'RandomForestClassifier':RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        oob_score=True)
}

def model_evaluation(X_train,X_test,y_train,y_test,models):
    reports = {}

    for model_name,model in models.items():
        model.fit(X_train,y_train)
        y_test_pred = model.predict(X_test)

        accuracyScore = accuracy_score(y_test,y_test_pred)
        confusionMatrix = confusion_matrix(y_test,y_test_pred)
        classificationReport = classification_report(y_test,y_test_pred)

        reports[model_name] = [
            accuracyScore,
            confusionMatrix,
            classificationReport
        ]

    return reports

reports = model_evaluation(X_train,X_test,y_train,y_test,models)
for model_name,model_data in reports.items():
    print(f'Model : {model_name}')
    print(f'Accuracy score : {model_data[0]}')
    print(f'Confusion Matrix : \n{model_data[1]}')
    print(f'Classification Report : \n{model_data[2]}')



model_objects = {
    'preprocessor': preprocessor,
    'models': models
}

file = open('model_objects.pkl','wb')
pickle.dump(model_objects,file)

print("Model object saved in model_objects.pkl")

