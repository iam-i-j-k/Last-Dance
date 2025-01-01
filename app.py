import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np

# Title of the App
st.title("MULTIMODAL ML APP")

# Step 1: Upload Dataset
if 1:
    data = pd.read_csv('asf.csv')
    st.write("Preview of the dataset:")
    st.write(data.head())

    # Step 2: Select Features and Target
    st.write("Step 2: Define Features and Target")
    feature_cols = data[['area','sex ratio','r cases','crimes','wine shops','men literacy','porn access','psych cases','desserted area','ring roads','slum areas','season','time of visit']]
    target_col = data['class']
    
    if 1:
        X = feature_cols
        Y = target_col

        # Step 3: Input Data for Prediction
        st.write("Step 3: Input Data for Prediction")
        default_inputs = {col: float(data[col].mean()) for col in feature_cols.columns}
        inputs = {}
        for col in feature_cols:
            inputs[col] = st.number_input(f"Input {col}", value=default_inputs[col])
        input_data = np.array([list(inputs.values())])

        # Step 4: Choose Algorithm
        st.write("Step 4: Choose Algorithm")
        algorithms = {
            "Linear Regression": LinearRegression(),
            "Logistic Regression": LogisticRegression(),
            "Support Vector Regression (SVR)": SVR(kernel='rbf'),
            "Support Vector Classifier (SVC)": SVC(kernel='poly'),
            "Naive Bayes": GaussianNB(),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "K-Means Clustering": KMeans(n_clusters=4)
        }
        algorithm = st.selectbox("Select an Algorithm", list(algorithms.keys()))

        if st.button("Predict"):
            model = algorithms[algorithm]
            if "Classifier" in algorithm or "Regression" in algorithm:
                model.fit(X, Y)
                prediction = model.predict(input_data)
                st.write(f"Prediction: {prediction[0]}")
                st.write(f"Accuracy: {model.score(X, Y) * 100:.2f}%")

                # Visualize Results
                st.write("Visualization:")
                plt.scatter(X.iloc[:, 0], Y, color='b', label='Data')
                plt.plot(X.iloc[:, 0], model.predict(X), color='r', label='Prediction')
                plt.xlabel(feature_cols.columns[0])
                plt.ylabel('Outcome')
                plt.legend()
                st.pyplot(plt)

            elif "K-Means" in algorithm:
                model.fit(X)
                st.write(f"Cluster Centers: {model.cluster_centers_}")
                prediction = model.predict(input_data)
                st.write(f"Prediction: Cluster {prediction[0]}")

                # Plot Clusters
                labels = model.labels_
                colors = ['blue', 'red', 'green', 'black', 'purple']
                plt.figure()
                for idx, label in enumerate(labels):
                    plt.scatter(X.iloc[idx, 0], X.iloc[idx, 1], color=colors[label])
                for center in model.cluster_centers_:
                    plt.scatter(center[0], center[1], color='yellow', s=200, marker='X', edgecolor='black')
                st.pyplot(plt)
