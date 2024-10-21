import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Extract
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
wine_data = pd.read_csv(url, sep=';')

# Step 2: Handling Missing Values
print(wine_data.isnull().sum())
wine_data.fillna(wine_data.mean(), inplace=True)

# Step 3: Outlier Detection
sns.boxplot(x=wine_data['alcohol'])
plt.title('Boxplot of Alcohol Content')
plt.show()

Q1 = wine_data['alcohol'].quantile(0.25)
Q3 = wine_data['alcohol'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

wine_data['alcohol'] = np.where(
    wine_data['alcohol'] > upper_bound, 
    wine_data['alcohol'].median(), 
    wine_data['alcohol']
)

# Step 4: Data Transformation
scaler = StandardScaler()
numerical_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                      'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                      'pH', 'sulphates', 'alcohol']

wine_data[numerical_features] = scaler.fit_transform(wine_data[numerical_features])

# Step 5: Final Data Cleaning
wine_data.to_csv('cleaned_wine_data.csv', index=False)
print(wine_data.head())
