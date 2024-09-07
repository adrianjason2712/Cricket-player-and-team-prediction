from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle

app = Flask(__name__, template_folder='templates')

# Load datasets at the start to avoid reloading every time
batting_ODI = pd.read_csv('static/data/odb.csv')
batting_test = pd.read_csv('static/data/tb.csv')
batting_t20 = pd.read_csv('static/data/twb.csv')
data = pd.concat([batting_ODI, batting_t20, batting_test])

# Prepare features (X) and target variable (y)
player_names = data['Player']
X = data.drop(['Player', 'Span'], axis=1)
data['Top n'] = (data['Ave'].rank(method='first', ascending=False) <= 11).astype(int)  # Adjust as needed
y = data['Top n']

# Handle missing or non-numeric values in X
X.replace('-', np.nan, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the hyperparameters for tuning
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],  # 1: Manhattan, 2: Euclidean
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize KNN
knn = KNeighborsClassifier()

# Define Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Grid Search with Stratified K-Fold Cross-Validation
grid_search = GridSearchCV(knn, param_grid, cv=skf, scoring='accuracy')
grid_search.fit(X_scaled, y)

# Get the best KNN model after tuning
best_knn = grid_search.best_estimator_

# Save the best KNN model to a file (optional, for reusability)
with open('best_knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)

@app.route('/')
def home():
    return render_template('cricket.html') 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        num_players = int(request.form['num_players'])

        # Check if the number of players is a multiple of 11
        if num_players % 11 != 0:
            return render_template('error.html', message="The number of players must be a multiple of 11.")

        num_teams = num_players // 11

        # Predict probabilities using the best KNN model
        y_prob = best_knn.predict_proba(X_scaled)[:, 1]
        y_pred = best_knn.predict(X_scaled)

        # Identify the top n predicted players by probability
        top_n_indices = np.argsort(y_prob)[-num_players:]
        top_n_names = player_names.iloc[top_n_indices]

        # Split top n players into teams of 11
        teams = [top_n_names[i:i + 11].tolist() for i in range(0, num_players, 11)]

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        report = classification_report(y, y_pred, output_dict=True)
        metrics_df = pd.DataFrame(report).transpose().drop(index=['accuracy', 'macro avg', 'weighted avg'])

        return render_template('result.html', num_players=num_players, teams=teams,
                               accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                               metrics_df=metrics_df.to_html())
    except Exception as e:
        return render_template('error.html', message=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)