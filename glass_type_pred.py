# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
@st.cache()
def prediction(model,RI, Na, Mg, Al, Si, K, Ca, Ba ,Fe):
  glass_type = model.predict([[RI, Na, Mg, Al, Si, K, Ca, Ba ,Fe]])
  glass_type = glass_type[0]
  if glass_type == 1:
    return "building windows float processed"
  elif glass_type == 2:
    return "building windows non float processed"
  elif glass_type == 3:
    return "vehicle windows float processed"
  elif glass_type == 4:
    return "vehicle windows non float processed"
  elif glass_type == 5:
    return "containers"
  else:
    return "tableware"
st.title('GLASS TYPE PREDICTOR')
st.sidebar.title('EXPLORATORY DATA ANALYSIS')
if st.sidebar.checkbox('Show raw data '):
  st.header('RAW Data')
  st.subheader('full data set')
  st.dataframe(glass_df)
st.sidebar.subheader('Scatter plot')
features_list = st.sidebar.multiselect('select x-axis values:',('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
st.set_option('deprecation.showPyplotGlobalUse', False)
for i in features_list:
  st.header(f'Scatter plot between {i} and GlassType')
  plt.figure(figsize=(20,5))
  plt.scatter(glass_df[i],glass_df['GlassType'])
  st.pyplot()
st.sidebar.subheader('Visualisation selector')
plot_types = st.sidebar.multiselect('select the chart or plot',('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))
if 'Histogram' in plot_types:
  st.subheader('Histogram')
  columns = st.sidebar.selectbox('Select a column to create a histogram:',['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
  plt.figure(figsize=(20,5))
  plt.title(f'Histogram for {columns}')
  plt.hist(glass_df[columns],bins = 'sturges',edgecolor = 'g')
  st.pyplot()
if 'Box Plot' in plot_types:
  st.subheader('Box Plot')
  columns = st.sidebar.selectbox('Select a column to create a Box Plot:',['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])
  plt.figure(figsize=(20,5))
  plt.title(f'Box Plot for {columns}')
  sns.boxplot(glass_df[columns])
  st.pyplot()
if 'Count Plot' in plot_types:
  st.header('Count Plot')
  plt.figure(figsize=(20,5))
  sns.countplot(glass_df['GlassType'])
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
if 'Pie Chart' in plot_types:
  st.header('Pie Chart')
  plt.figure(dpi = 100 )
  plt.pie(glass_df['GlassType'].value_counts(),autopct = '%1.2f%%',labels = glass_df['GlassType'].value_counts().index)
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if 'Correlation Heatmap' in plot_types:
  st.header('Correlation Heatmap')
  plt.figure(figsize=(20,5))
  sns.heatmap(glass_df.corr(),annot = True)
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
if 'Pair Plot' in plot_types:
  st.header('Pair Plot')
  plt.figure(figsize=(20,5))
  sns.pairplot(glass_df)
  st.pyplot()
st.sidebar.subheader('select your input:')
ri = st.sidebar.slider('Input Ri:',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
na = st.sidebar.slider('Input Na:',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
mg = st.sidebar.slider('Input Mg:',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
al = st.sidebar.slider('Input Al:',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
si = st.sidebar.slider('Input Si:',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
k = st.sidebar.slider('Input K:',float(glass_df['K'].min()),float(glass_df['K'].max()))
ba = st.sidebar.slider('Input Ba:',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
ca = st.sidebar.slider('Input Ca:',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
fe = st.sidebar.slider('Input Fe:',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
st.sidebar.subheader('Choose Classifier')
classifier = st.sidebar.selectbox('Classifier',('Support Vector Machine','Randomforest Classifier','Logistic Regression'))
from sklearn.metrics import plot_confusion_matrix
if classifier == 'Support Vector Machine':
  st.sidebar.subheader('Model Hyper Parameter')
  c_value = st.sidebar.number_input('C (Error Rate)',1,100,step = 1)
  kernel_value = st.sidebar.radio('Kernel',('linear','rbf','poly'))
  gamma_value = st.sidebar.number_input('Gamma Value',1,100,step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svc_model = SVC(C = c_value , kernel = kernel_value, gamma = gamma_value)
    svc_model.fit(X_train,y_train)
    y_predict = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test,y_test)
    predicted = prediction(svc_model,ri, na, mg, al, si, k, ca, ba ,fe)
    st.write('Type of Glass Predicted is : ',predicted)
    st.write('Accuracy is : ',accuracy)
    plot_confusion_matrix(svc_model,X_test,y_test)
    st.pyplot()
if classifier == 'Randomforest Classifier':
  st.sidebar.subheader('Model Hyper Parameter')
  estimator_inp = st.sidebar.number_input('Number of Trees : ',100,5000,step = 10)
  max_dept_inp = st.sidebar.number_input('Maximum depth of Tree : ',1,100,step = 1)
  if st.sidebar.button('Classify'):
    st.subheader('Randomforest Classifier')
    rfc_model = RandomForestClassifier(n_estimators= estimator_inp,max_depth = max_dept_inp)
    rfc_model.fit(X_train,y_train)
    y_predict = rfc_model.predict(X_test)
    accuracy = rfc_model.score(X_test,y_test)
    predicted = prediction(rfc_model,ri, na, mg, al, si, k, ca, ba ,fe)
    st.write('Type of Glass Predicted is : ',predicted)
    st.write('Accuracy is : ',accuracy)
    plot_confusion_matrix(rfc_model,X_test,y_test)
    st.pyplot()
if classifier == 'Logistic Regression':
  st.sidebar.subheader('Model Hyper Parameter')
  c_value = st.sidebar.number_input('C : ',1,100,step = 1)
  max_iteration = st.sidebar.number_input('Maximum iteration : ',10,1000,step = 10)
  if st.sidebar.button('Classify'):
    st.subheader('Logistic Regression')
    lr_model = LogisticRegression(C = c_value,max_iter = max_iteration)
    lr_model.fit(X_train,y_train)
    y_predict = lr_model.predict(X_test)
    accuracy = lr_model.score(X_test,y_test)
    predicted = prediction(lr_model,ri, na, mg, al, si, k, ca, ba ,fe)
    st.write('Type of Glass Predicted is : ',predicted)
    st.write('Accuracy is : ',accuracy)
    plot_confusion_matrix(lr_model,X_test,y_test)
    st.pyplot()
