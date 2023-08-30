import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.model_selection import train_test_split

import h2o
from h2o.frame import H2OFrame
from h2o.estimators.random_forest import H2ORandomForestEstimator


data = pd.read_csv('D:\Machine Learning\ML TP Random Forest\data\Fraud\Fraud_Data.csv', parse_dates=['signup_time', 'purchase_time'])
print(data.head())

address2country = pd.read_csv('D:\Machine Learning\ML TP Random Forest\data\Fraud\IpAddress_to_Country.csv')
print(address2country.head())

# Merge the two datasets and print the first 5 rows
merged_data = pd.merge(data, address2country, 
                       left_on='ip_address', 
                       right_on='lower_bound_ip_address', 
                       how='left')

print("merged data : ", merged_data.head())

# Time diff
merged_data['time_diff'] = (merged_data['purchase_time'] - merged_data['signup_time']).dt.total_seconds() / 3600  # Time difference in hours
# Check user number for unique devices
merged_data['device_num'] = merged_data.groupby('device_id')['user_id'].transform('nunique')
# Check user number for unique ip_address
merged_data['ip_num'] = merged_data.groupby('ip_address')['user_id'].transform('nunique')
# Signup day and week
merged_data['signup_day'] = merged_data['signup_time'].dt.dayofweek  
merged_data['signup_week'] = merged_data['signup_time'].dt.isocalendar().week
# Purchase day and week
merged_data['purchase_day'] = merged_data['purchase_time'].dt.dayofweek 
merged_data['purchase_week'] = merged_data['purchase_time'].dt.isocalendar().week

print("With new features :", merged_data.head())

# Define features and target to be used
final_data = merged_data.drop(columns=['signup_time', 'purchase_time', 'device_id', 'ip_address', 'user_id'])
cols_order = ['signup_day', 'signup_week', 'purchase_day', 'purchase_week', 'purchase_value', 'source', 'browser', 'sex', 'age', 'country', 'time_diff', 'device_num', 'ip_num', 'class']
final_data = final_data[cols_order]
#print(final_data.head())

X = final_data.drop(columns=['class'])
y = final_data['class']


# Split into 70% training and 30% test dataset
# Define features and target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test :", y_test.shape)

# Build random forest model
h2o.init()
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
predictors = X_train.columns.tolist()
response = 'class'
drf = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)
drf.train(x=predictors, y=response, training_frame=train)

feature_importances = drf.varimp(use_pandas=True)

# seaborn barplot for feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='scaled_importance', y='variable', data=feature_importances.sort_values(by='scaled_importance', ascending=False))
plt.title("Feature Importance")
plt.xlabel("Scaled Importance")
plt.ylabel("Feature")
plt.savefig("feature_barplot.png")

# Classification report
predictions_h2o = drf.predict(test)
predictions_df = predictions_h2o.as_data_frame()
threshold = 0.5 
predictions_df['predict'] = (predictions_df['predict'] > threshold).astype(int)
classification_rep = classification_report(y_test, predictions_df['predict'])

print("Classification Report:")
print(classification_rep)

# plot ROC curve and calculate AUC
fpr, tpr, thresholds = roc_curve(y_test, predictions_df['predict'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("ROC_curve.png")

print("AUC:", roc_auc)