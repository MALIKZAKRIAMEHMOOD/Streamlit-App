import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('K-Nearest Neighborsmodel.pkl')

#Load the test data
test_data = pd.read_csv('mobile_price_range_data.csv')

#Assuming the last column in the target
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

#Generate Test Predictions
y_pred = model.predict(x_test)

#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

#Save accuracy to file
with open('accuracy.txt', 'w') as file:
    file.write(f'Accuracy: {accuracy}')

print(f'Accuracy: {accuracy}')
