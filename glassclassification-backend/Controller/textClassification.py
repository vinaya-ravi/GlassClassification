import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import spacy

# Load spaCy's English-language model
nlp = spacy.load('en_core_web_sm')

# Your dataset should be in a CSV file with two columns: 'description' and 'glass_type'
# dataset = pd.read_csv('your_dataset.csv')

# For demonstration purposes, let's create a mock dataset
data = {
   'description': [
        'This glass type is engineered for safety and is far stronger than standard glass. When broken, it crumbles into small granules instead of sharp shards, making it ideal for use in car windows and some types of shower doors.',
        'Known for its high resistance to scratches, this glass is commonly used in smartphones and other touchscreen devices. It is thin and light, yet offers excellent durability and damage resistance, making it a top choice for mobile and portable electronics.',
        'This glass has a high refractive index and is much denser than normal glass, which makes it perfect for optical uses like camera lenses and spectacles.',
        'Safety glass that shatters into blunt pieces is perfect for public areas and home bathrooms to ensure accidents dont cause serious injuries.',
        'Advanced glass used in the latest consumer electronics is resistant to scratches and pressure, ideal for screens that undergo frequent use and rough handling.',
        'Optical grade glass that provides superior clarity for high-quality lenses, utilized in professional photography and precision instruments.',
        'Tempered glass used in storm doors can withstand high impacts and severe weather conditions, ensuring safety and durability in extreme environments.',
        'Smartphone screens now use a special type of glass that is engineered to be resistant to drops and scratches, making them durable against everyday accidents.',
        'Lead glass is used in radiation shielding windows, offering high density to protect against X-rays in medical facilities and laboratories.',
        'Engineered for enhanced optical performance, this glass type is favored in scientific applications where clarity and precision are crucial.'
    ],
    'glass_type': [
        'Tempered Glass',
        'Gorilla Glass',
        'Lead Glass',
        'Tempered Glass',
        'Gorilla Glass',
        'Lead Glass',
        'Tempered Glass',
        'Gorilla Glass',
        'Lead Glass',
        'Lead Glass'
    ]
}

df = pd.DataFrame(data)
print(df['glass_type'].value_counts())

# Define a function to preprocess text using spaCy
def preprocess_text(text):
    doc = nlp(text)
    # Example of a simple preprocessing step: lemmatization
    # You could also remove stop words, perform entity recognition, etc.
    return ' '.join(token.lemma_ for token in doc if not token.is_stop)

# Apply the preprocessing function to the descriptions
df['processed_description'] = df['description'].apply(preprocess_text)

# Vectorization with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['processed_description'])
y = df['glass_type']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)

# Save the classifier and vectorizer to a file
joblib.dump(classifier, 'trained_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'vectorizer.pkl')

# Provide the paths to the saved files
classifier_path = 'trained_classifier.pkl'
vectorizer_path = 'vectorizer.pkl'

classifier_path, vectorizer_path
