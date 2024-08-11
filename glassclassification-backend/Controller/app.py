from flask import Flask,Response
from werkzeug.utils import secure_filename
from flask import request
from flask import render_template
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
from flask import  jsonify
import sys
import joblib
import json
import requests
import pandas
import codecs
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import spacy
import tensorflow as tf
sys.path.append("..")
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app,resources={r'/ml/*': {'origins':['http://localhost:3000','https://main--glassclassificationanalysis090523.netlify.app']}})

data = pd.read_csv('/Users/vinaya/Desktop/UNT/SPRING-2024/SDAI/GlassClassification/glassclassification-backend/glass.csv')
plt.switch_backend('Agg')

X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, stratify=Y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# A helper function to reduce the code for training each model separately
# It takes model, model name, X and Y (both train and test) and trains the model.
def base(model, X, Y, model_name):
    model.fit(X[0], Y[0])
    results = cross_val_score(model, X[0], Y[0], cv=10)
    return results


# Initializing the required models and variables for the training and validation process.
# 7 different models have been tested with one model having different variations (Random Forest).

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Adaboost Classifier": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300),
    "Extra Tree Classifier": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest 50": RandomForestClassifier(n_estimators=50),
    "Random Forest 75": RandomForestClassifier(n_estimators=75),
    "Random Forest 100": RandomForestClassifier(n_estimators=100),
}
X = [X_train, X_test]
Y = [Y_train, Y_test]


cv_results_train = dict()
for model_name, model in models.items():
    result = base(model, X, Y, model_name)
    cv_results_train[model_name] = result

# Save the trained model
best_model_name = max(cv_results_train, key=lambda k: max(cv_results_train[k]))
best_model = models[best_model_name]
best_model.fit(X[0], Y[0])

print(f"The best model is {best_model} with a score of")

@app.route('/ml/MLalgo', methods=["GET"])
def getData():
    print(data.describe())
    response=data.describe()
    statistics_json = response.to_json(orient='index')
    resp=Response(statistics_json)
    resp.headers['Access-Control-Allow-Credentials']='true'
    resp.headers['Access-Control-Allow-Headers']= "Origin, X-Requested-With, Content-Type, Accept"
    resp.headers['Access-Control-Allow-Methods']="GET, POST, PUT, DELETE, OPTIONS"
    return resp

@app.route('/ml/heatmap', methods=["GET"])
def getHeatMaps():
        # Set up the figure and axis
    plt.figure(figsize=(8, 8))

    # Generate the heatmap
    sns.heatmap(data.corr(), annot=True)

    # Save the plot to a BytesIO object
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)
    plt.close()
     # Return the plot as an image response
    return Response(img_bytesio.getvalue(), mimetype='image/png')

# Endpoint to get and send the distribution plot for each feature
@app.route('/ml/distributionplot', methods=['GET'])
def distribution_plot():
    plt.figure(figsize=(15, 15))
    plt.tight_layout()

    for i, feature in enumerate(data.columns.tolist()[:-1]):
        plt.subplot(int(f"33{i+1}"))
        sns.distplot(data[feature], kde=True)

    plt.suptitle("Distribution plot for each feature")

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # Send the image file directly
    return Response(img_buffer, mimetype='image/png')


@app.route('/ml/boxquartiles', methods=['GET'])
def box_quartile_plot():
    plt.figure(figsize=(15, 10))
    plt.title("Box plot for each feature")
    sns.boxplot(data=data.iloc[:, :-1], orient='h')

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # Send the image file directly
    return Response(img_buffer.getvalue(), mimetype='image/png')


@app.route('/ml/pairplot', methods=["GET"])
def getPairplots():
        # Set up the figure and axis
    plt.figure(figsize=(8, 8))

    # Generate the pairplot
    sns.pairplot(data, hue="Type")

    # Save the plot to a BytesIO object
    img_bytesio = BytesIO()
    plt.savefig(img_bytesio, format='png')
    img_bytesio.seek(0)
    plt.close()
     # Return the plot as an image response
    return Response(img_bytesio.getvalue(), mimetype='image/png')

# Endpoint to check the count of each class type
@app.route('/ml/checkclasscounts', methods=['GET'])
def check_class_counts():
    types,counts = np.unique(data['Type'].values, return_counts=True)

    # Create a bar plot
    plt.bar(types, counts)
    plt.title("Checking count of each class type")
    plt.xlabel("Class Type")
    plt.ylabel("Count")

    # Save the plot to a BytesIO object
    img_buffer2 = BytesIO()
    plt.savefig(img_buffer2, format='png')
    img_buffer2.seek(0)

    # Send the image file directly
    return Response(img_buffer2.getvalue(), mimetype='image/png')


# API endpoint for box plot visualization
@app.route('/ml/boxplot', methods=['GET'])
def box_plot():
    plt.figure(figsize=(25, 15))
    plt.boxplot(cv_results_train.values(), labels=cv_results_train.keys())
    plt.xticks(rotation='vertical')
    plt.title("Comparing performance of different models")
    plt.savefig("box_plot.png")  # Save the box plot image
      # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    return Response(img_buffer.getvalue(), mimetype='image/png')


# API endpoint for confusion matrix heatmap
@app.route('/ml/trainconfusionmatrix', methods=['GET'])
def confusion_matrix_heatmap_train():
    # Generate confusion matrix
    print("best_model",best_model_name)
    cm = confusion_matrix(Y[0], best_model.predict(X[0]))
    print(classification_report(Y[0], best_model.predict(X[0])))
    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='g')
    plt.title("Train Confusion Matrix")

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Send the image file directly
    return Response(img_buffer, mimetype='image/png')



# API endpoint for confusion matrix heatmap
@app.route('/ml/testconfusionmatrix', methods=['GET'])
def confusion_matrix_heatmap_test():
    # Generate confusion matrix
    print("best_model",best_model_name)
    cm = confusion_matrix(Y[1], best_model.predict(X[1]))
    print(classification_report(Y[1], best_model.predict(X[1])))
    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='coolwarm', fmt='g')
    plt.title("Test Confusion Matrix")

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Send the image file directly
    return Response(img_buffer, mimetype='image/png')

# Load your trained model (replace with the path to your model)
model = tf.keras.models.load_model('/Users/vinaya/Desktop/UNT/SPRING-2024/SDAI/GlassClassification/glassclassification-backend/glass_type_classifier.h5')

@app.route('/ml/classify-image', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    # Make sure to read the image in the correct color mode if your model expects grayscale or RGB
    image = Image.open(BytesIO(file.read())).convert('RGB')
    
    # Preprocess the image to fit your model's input requirements (size, shape, scaling, etc.)
    image = image.resize((150, 150))  # Example size, change to what your model expects
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # Use the model to predict the class
    predictions = model.predict(image_array)
    print("ppp",predictions)
    # Assuming your model outputs one prediction per class in the order of ['Tempered Glass', 'Gorilla Glass', 'Lead Glass']
    predicted_classes = ['Gorilla Glass', 'Lead Glass', 'Tempered Glass']
    predicted_class = predicted_classes[np.argmax(predictions)]
    # Display predicted probabilities alongside class names
    predicted_probabilities = dict(zip(predicted_classes, predictions))
    # Output the predicted class with the highest probability and all class probabilities
    return jsonify({
    "predicted_class": predicted_class
   # "probabilities": predicted_probabilities
})

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the trained classifier and vectorizer
classifier = joblib.load('/Users/vinaya/Desktop/UNT/SPRING-2024/SDAI/GlassClassification/glassclassification-backend/trained_classifier.pkl')
vectorizer = joblib.load('/Users/vinaya/Desktop/UNT/SPRING-2024/SDAI/GlassClassification/glassclassification-backend/vectorizer.pkl')
# Define a dictionary mapping glass types to key characteristics
characteristics_map = {
    'Tempered Glass': ['Far Stronger', 'Small Granules', 'Used in car windows'],
    'Gorilla Glass': ['Scratch Resistant', 'Used in smartphones', 'Thin and light'],
    'Lead Glass': ['High Refractive Index', 'Used in camera lenses', 'Denser than normal glass']
}
# Define a function for text preprocessing
def preprocess_text(text):
    doc = nlp(text)
    return ' '.join(token.lemma_ for token in doc if not token.is_punct and not token.is_stop)

@app.route('/ml/classify-text', methods=['POST'])
def classify_glass_from_text():
    data = request.json
    text_description = data.get('description', '')
    # Preprocess the text
    processed_text = preprocess_text(text_description)

    # Vectorize the processed text
    vectorized_text = vectorizer.transform([processed_text])

    # Predict the glass type
    prediction = classifier.predict(vectorized_text)
    # Get the key characteristics for the predicted glass type
    predicted_class = prediction[0]
    key_characteristics = characteristics_map.get(predicted_class, ["No characteristics found"])

        
    # You can also output probabilities or other model outputs as needed
    # probabilities = classifier.predict_proba(vectorized_text)

    # Return the prediction as a JSON response
    return jsonify({
            "description": text_description,
            "predicted_class": prediction[0],
            "key_characteristics": key_characteristics
        })
    


if __name__ == '__main__':
    app.run(debug=True, port=8001)




