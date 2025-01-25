import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('match_data_with_wins.csv')

#Preprocess the data
df['result'] = df['result'].apply(lambda x: 1 if x == 'Win' else (0 if x == 'Loss' else x))

#Drop unnecessary columns
columns_to_drop = ['score', 'goals', 'goals conceded', 'shots conceded', 'match_id']
df_filtered = df.drop(columns=columns_to_drop)

#Ensure data is numerical
df_numeric = df_filtered.select_dtypes(include=['number'])

#Define predictors (X) and target (y)
X = df_numeric.drop(columns=['result'])
y = df_numeric['result']

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Predict on the test set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

#Moidel Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_prob):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.show()

#Coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(coefficients)

#Savings models and Data
plt.savefig('confusion_matrix.png')
plt.savefig('roc_curve.png')
df_numeric['predictions'] = model.predict(X)
df_numeric.to_csv('updated_data_with_predictions.csv', index=False)