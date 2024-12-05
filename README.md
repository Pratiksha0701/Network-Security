# Network-Security


Network Security with Machine Learning


The most likely way for attackers to gain access to your infrastructure is through the network. Network security is the general practice of protecting computer networks and devices accessible to the network against malicious intent, misuse and denial. In this article, I will take you through techniques of Network Security Analysis with Machine Learning.
 


 
Exploring patterns is one of the main strengths of machine learning, and there are many inherent patterns to discover in the network traffic data. At first glance, network packet capture data may appear sporadic and random, but most communication flows follow strict network protocol.
Live network data capture is the primary way to record network activity for online or offline analysis. Like a CCTV camera at a traffic intersection, packet analyzers intercept and record network traffic. Now let’s create a network attack classifier from scratch using machine learning.
Building a Predictive Model to Classify Network Security Attacks
The dataset we will be using is the NSLKDD dataset, which is an improvement over traditional network intrusion detection. This dataset widely used by security data science professionals to classify problems of Network Security. You can download the dataset from here. Let’s start this task by importing some necessary libraries:
import os
 from collections import defaultdict
 import pandas as pd
 import numpy as np
import matplotlib.pyplot as plt


Data Exploration:
Let’s look at the preliminary data to get some insight into the data. Let’s take a look at the breakdown of categories first:



Code :

dataset_root = 'datasets/nsl-kdd' train_file = os.path.join(dataset_root, 'KDDTrain+.txt') test_file = os.path.join(dataset_root, 'KDDTest+.txt') header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred'] col_names = np.array(header_names) nominal_idx = [1, 2, 3] binary_idx = [6, 11, 13, 14, 20, 21] numeric_idx = list(set(range(41)).difference(nominal_idx).difference(binary_idx)) nominal_cols = col_names[nominal_idx].tolist() binary_cols = col_names[binary_idx].tolist() numeric_cols = col_names[numeric_idx].tolist() category = defaultdict(list) category['benign'].append('normal') with open('datasets/training_attack_types.txt', 'r') as f: for line in f.readlines(): attack, cat = line.strip().split(' ') category[cat].append(attack) attack_mapping = dict((v,k) for k in category for v in category[k])



Now, here is the data that we will be using:


train_df = pd.read_csv(train_file, names=header_names) train_df['attack_category'] = train_df['attack_type'] \ .map(lambda x: attack_mapping[x]) train_df.drop(['success_pred'], axis=1, inplace=True) test_df = pd.read_csv(test_file, names=header_names) test_df['attack_category'] = test_df['attack_type'] \ .map(lambda x: attack_mapping[x]) test_df.drop(['success_pred'], axis=1, inplace=True) train_attack_types = train_df['attack_type'].value_counts() train_attack_cats = train_df['attack_category'].value_counts() test_attack_types = test_df['attack_type'].value_counts() test_attack_cats = test_df['attack_category'].value_counts() train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)



train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=30)




Data Preparation
The NSL-KDD dataset is a useful dataset for education and experimentation with data mining and machine learning classification because it strikes a balance between simplicity and sophistication.
Let’s start by splitting the test and training DataFrames into data and labels:
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)
In typical cases, we will have complete knowledge of all categorical variables either because we defined them or because the dataset provided this information. In this case of Network Security Analysis, the dataset is not provided with a list of possible values of each categorical variable, so we can preprocess as follows:
 

 
combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))
Now let’s apply the Standard Scalar Algorithm on this data to scale the dataset:
# Experimenting with StandardScaler on the single 'duration' feature
from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)
pd.Series(scaled_durations.flatten()).describe()
count    1.259730e+05
mean     2.549477e-17
std      1.000004e+00
min     -1.102492e-01
25%     -1.102492e-01
50%     -1.102492e-01
75%     -1.102492e-01
max      1.636428e+01
dtype: float64
You can choose to use MinMaxScaler on StandardScaler if you want the scaling operation to keep the small standard deviations of the original series, or if you want to keep zero entries in the sparse data. Here’s how MinMaxScaler transforms the duration function:
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
pd.Series(min_max_scaled_durations.flatten()).describe()
count    125973.000000
mean          0.006692
std           0.060700
min           0.000000
25%           0.000000
50%           0.000000
75%           0.000000
max           1.000000
dtype: float64
Outliers in your data can seriously and negatively skew the results of standard scaling and normalization. If the data contains outliers, sklearn.preprocessing.RobustScaler will be more suitable for this problem of Network Security. RobustScaler uses robust estimates such as median and quartile ranges, so it won’t be affected as much by outliers:
from sklearn.preprocessing import RobustScaler

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
pd.Series(robust_scaled_durations.flatten()).describe()
count    125973.00000
mean        287.14465
std        2604.51531
min           0.00000
25%           0.00000
50%           0.00000
75%           0.00000
max       42908.00000
dtype: float64
We complete the data preprocessing phase by standardizing training and test data:
 

 
standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])
Classification for Network Security Analysis
By applying the default or initial best guess parameters to the algorithm, we can quickly get initial classification results for Network Security. While these results may not be close to the accuracy of our goal, they will usually give us a rough indication of the potential of the algorithm.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(train_x, train_Y)
pred_y = classifier.predict(test_x)
results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
[[9365   56  289    1    0]
 [1541 5998   97    0    0]
 [ 675  220 1528    0    0]
 [2278    1   14  277    4]
 [ 179    0    5    5   11]]
0.238245209368


