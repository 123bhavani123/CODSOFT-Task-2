import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
movie_data = pd.read_csv("C:\\Users\Bhavani\OneDrive\Desktop\durga\Project\IMDb Movies India.csv",encoding='latin1')
print(movie_data.info())
print(movie_data.describe())
sns.histplot(movie_data['Rating'], bins=20, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
movie_data.dropna(inplace=True)
X = movie_data[['Genre', 'Director', 'Actor 1','Actor 2','Actor 3']]
y = movie_data['Rating']
X_encoded = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
coefficients = pd.DataFrame({'Feature': X_encoded.columns, 'Coefficient': model.coef_})
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients)
plt.title('Coefficients of Linear Regression Model')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()
