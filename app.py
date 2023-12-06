import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Display the Iris dataset
st.markdown('<h1 style="color:orange;">Iris Dataset </h1>', unsafe_allow_html=True)
st.write('**This is iris dataset**')
st.write(df)

st.markdown('<h1 style="color:green;">Charts View </h1>', unsafe_allow_html=True)
st.write('**This a multiple charts use function. It is very simple. you select chart and visualize data in liked chart.**')
# # Select chart type
chart_type = st.selectbox('Select Chart Type', ['Bar Chart', 'Line Chart', 'Scatter Plot'])


if chart_type == 'Bar Chart':
    fig = px.bar(df, x='sepal_length', y='sepal_width', title='Bar Chart')
elif chart_type == 'Line Chart':
    fig = px.line(df, x='sepal_length', y='sepal_width', title='Line Chart')
elif chart_type == 'Scatter Plot':
    fig = px.scatter(df, x='sepal_length', y='sepal_width', title='Scatter Plot')
st.plotly_chart(fig)

# Display a summary table of the sorted dataset
st.markdown('<h1 style="color:blue;">Discription Summary Title</h1>', unsafe_allow_html=True)

st.write("**This is a full summary of dataset. It is a giving you multiple things like: Count, Mean, Std, Min, 25%, 50%, 75%, Max**")

st.table(df.describe())

# Encode the 'species' column
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# Split the data into features (X) and target (y)
X = df.drop(['species', 'species_encoded'], axis=1)
y = df['species_encoded']

# Train a simple Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit app
st.markdown('<h1 style="color:red;">Iris Dataset Prediction App</h1>', unsafe_allow_html=True)

# Sidebar with user input
st.sidebar.header('Input Parameters')

sepal_length = st.sidebar.slider('Sepal Length', min_value=4.0, max_value=8.0, value=5.0)
sepal_width = st.sidebar.slider('Sepal Width', min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.sidebar.slider('Petal Length', min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.sidebar.slider('Petal Width', min_value=0.1, max_value=2.5, value=1.3)

# Make prediction
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(user_input)

# Decode the predicted class
predicted_species = le.inverse_transform(prediction)[0]

# Display the prediction
st.write('Prediction:', predicted_species)

# Display model performance (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.markdown('<h1 style="color:gray;">Model Performance </h1>', unsafe_allow_html=True)
st.write('**This is dataset model performence and it is a prediction of dataset. It is a full report of classification and accuracy.**')
st.write(f'Accuracy: {accuracy:.2f}')
st.text('Classification Report:')
st.code(report)

st.title("Pair Plot with Streamlit")

# Sidebar for customization
features = st.sidebar.multiselect("Select features:", df.columns)

# Create pair plot
if features:
    st.write(f"Creating pair plot for selected features: {features}")
    pair_plot = sns.pairplot(df[features])
    st.pyplot(pair_plot)
else:
    st.warning("Please select at least one feature.")
