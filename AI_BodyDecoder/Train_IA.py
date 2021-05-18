
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle

df = pd.read_csv('./data/coords/coords.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

fit_models = {}

for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

accuracy_list = []

for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, yhat))
    print(algo, accuracy_score(y_test, yhat))

max_index = accuracy_list.index(max(accuracy_list))
pipe_list = list(pipelines)
best_algo = str(pipe_list[max_index])
print("The best algorithm was " + best_algo)

with open('./data/models/body_language.pkl', 'wb') as f:
    pickle.dump(fit_models[best_algo], f)