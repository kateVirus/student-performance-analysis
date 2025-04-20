#Load the dataset:
import pandas as pd
data = pd.read_csv(r"C:\Users\User\Downloads\GuidedProject_MLG382_2025\Student_performance_data .csv")
print(data.head())

#Check for missing values:
print(data.isnull().sum())

#Analyze data types:
print(data.info())

#5. Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(data['GPA'], kde=True)
plt.title("Distribution of GPA")
plt.show()

#ii. Perform Bivariate Analysis
sns.boxplot(x='GradeClass', y='StudyTimeWeekly', data=data)
plt.title("Study Time vs Grade Class")
plt.show()

#6. Missing Value and Outlier Treatment
data.fillna(data.mean(), inplace=True)

#Handle Outliers:
Q1 = data['GPA'].quantile(0.25)
Q3 = data['GPA'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['GPA'] >= Q1 - 1.5 * IQR) & (data['GPA'] <= Q3 + 1.5 * IQR)]

#7. Evaluation Metrics for Classification Problem
#Choose metrics like accuracy, precision, recall, F1-score, and AUC-ROC for evaluation.

#8. Feature Engineering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['StudyTimeWeekly', 'Absences']] = scaler.fit_transform(data[['StudyTimeWeekly', 'Absences']])

#9. Model Building: Part 1 (Baseline ML Models)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

X = data.drop(columns=['GradeClass'])
y = data['GradeClass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

#10. Model Building: Part 2 (Deep Learning Model)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax') # Output layer for GradeClass (5 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)

loss, accuracy = model.evaluate(X_test, y_test)
print("Deep Learning Model Accuracy:", accuracy)

#11. Model Deployment - Dash App on Render.com
import dash
from dash import dcc, html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Student Performance Prediction"),
    dcc.Input(id="input-study-time", type="number", placeholder="Enter Study Time"),
    dcc.Input(id="input-parental-support", type="number", placeholder="Enter Parental Support"),
    html.Button("Predict", id="predict-button"),
    html.Div(id="output-prediction")
])

@app.callback(
    dash.dependencies.Output("output-prediction", "children"),
    [dash.dependencies.Input("predict-button", "n_clicks")],
    [dash.dependencies.State("input-study-time", "value"), dash.dependencies.State("input-parental-support", "value")]
)
def predict_grade(n_clicks, study_time, parental_support):
    if n_clicks is None:
        return ""
    # Example prediction logic (replace with actual model prediction):
    grade_class = rf.predict([[study_time, parental_support]])
    return f"Predicted Grade Class: {grade_class[0]}"

if __name__ == '__main__':
    app.run_server(debug=True)








