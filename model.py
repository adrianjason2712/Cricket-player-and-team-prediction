import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Step 1: Load the datasets
batting_ODI = pd.read_csv('static/data/odb.csv')
batting_test = pd.read_csv('static/data/tb.csv')
batting_t20 = pd.read_csv('static/data/twb.csv')

# Step 2: Combine the datasets
data = pd.concat([batting_ODI, batting_t20, batting_test])

# Step 3: Prepare features (X) and target variable (y)
# Store player names and span separately
player_names = data['Player']
span = data['Span']

# Drop 'Player' and 'Span' as they are not needed for prediction
X = data.drop(['Player', 'Span'], axis=1)

# Prompt user to enter the number of players to select
n = int(input("Enter the number of players to select (must be a multiple of 11): "))

# Check if n is a multiple of 11
if n % 11 != 0:
    raise ValueError("The number of players must be a multiple of 11.")

# Number of teams
num_teams = n // 11

# Create the target variable 'Top n'
data['Top n'] = (data['Ave'].rank(method='first', ascending=False) <= n).astype(int)
y = data['Top n']

# Step 4: Handle missing or non-numeric values in X
X.replace('-', np.nan, inplace=True)
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5)  # 5-fold cross-validation

# Set up the KNN model and grid search parameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'p': [1, 2],  # p=1 for Manhattan, p=2 for Euclidean
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

best_knn = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Step 7: Predict probabilities using the best KNN model
y_prob = best_knn.predict_proba(X_scaled)[:, 1]

# Step 8: Calculate accuracy, precision, recall, and F1 score
y_pred = best_knn.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Step 9: Identify the top n predicted players by probability
top_n_indices = np.argsort(y_prob)[-n:]
top_n_names = player_names.iloc[top_n_indices]

print(f"\nNames of top {n} predicted players:\n")
print(top_n_names.tolist())

# Step 10: Split top n players into teams of 11
teams = [top_n_names[i:i + 11].tolist() for i in range(0, n, 11)]

# Display the teams
for i, team in enumerate(teams, 1):
    print(f"\nTeam {i}: {team}")

# Step 11: Generate precision, recall, and F1 score plots
report = classification_report(y, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

# Filter out support and average metrics
metrics_df = metrics_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])

# Plot metrics
metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(10, 6))
plt.title('Precision, Recall, and F1 Score for Each Class')
plt.ylabel('Score')
plt.xlabel('Class')
plt.show()

# Optional: Save the model
with open('best_knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)