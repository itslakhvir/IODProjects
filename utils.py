# def create_label(row):
#     if row['subreddit'] in ['ptsd', 'assistance', 'relationships', 'survivorsofabuse', 'domesticviolence']:
#         if row['confidence'] >= 0.8:
#             return 1  # Stress
#         else:
#             return 0  # No stress
#     elif row['subreddit'] in ['anxiety', 'homeless', 'stress', 'almosthomeless', 'food_pantry']:
#         if row['confidence'] >= 0.6:
#             return 1  # Stress
#         else:
#             return 0  # No stress
#     else:
#         return 0  # Default to No stress
# data['label'] = data.apply(create_label, axis=1)
# print(data)

# # Defining a function to create the label based on both the confidence score and the subreddit
# def create_stress_label(row, confidence_threshold=0.8, stress_subreddits={'ptsd', 'anxiety', 'domesticviolence'}):
#     # Label as stress if the confidence is above the threshold and the subreddit is one of the stress-related ones
#     if row['confidence'] > confidence_threshold and row['subreddit'] in stress_subreddits:
#         return 1
#     else:
#         return 0

# # Apply the function to the dataset
# data['label'] = data.apply(create_stress_label, axis=1)

# # Display the first few rows with the new 'label' column
# data.head()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import nlp
from sklearn.model_selection import LearningCurveDisplay,train_test_split,GridSearchCV
import plotly
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re 
import string
import plotly.express as px
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
from transformers import BertTokenizer
from sklearn.naive_bayes import BernoulliNB



# Step 1: Text Preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def remove_stopwords(text):
    stpw = set(stopwords.words('english'))
    filtered_text = [word for word in text if word not in stpw]
    return filtered_text

def lemmatize_words(text):
    lemmer = WordNetLemmatizer()
    lemmatized_text = [lemmer.lemmatize(word, pos='v') for word in text]
    return lemmatized_text

# Step 7: Classification Report and Confusion Matrix



# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Models and their predictions



models = [RandomForestClassifier(), SVC(), KNeighborsClassifier()]
model_names = ['RandomForestClassifier', 'SVC', 'KNeighborsClassifier']

# Iterate through models and plot confusion matrix
for model, model_name in zip(models, model_names):
    pipeline = make_pipeline(TfidfVectorizer(), model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)




# Assuming you have the word_tokenize, remove_stopwords, and lemmatize_words functions defined
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_words(tokens)
    return ' '.join(tokens)

data=pd.read_csv(r"stress.csv")

# Split your data into training and testing sets
from sklearn.model_selection import train_test_split,GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Assuming you have a trained pipeline with SVC
pipeline = make_pipeline(TfidfVectorizer(), SVC())
pipeline.fit(X_train, y_train)

# Assuming X_valid is your new data
X_valid = input['text'].apply(preprocess_text)

# Predict on the new data
input['predicted'] = pipeline.predict(X_valid)
input['predicted'] = input['predicted'].apply(lambda x: 'Stress' if x == 1 else 'Not Stress')
