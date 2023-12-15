import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import nlp
from sklearn.model_selection import LearningCurveDisplay,train_test_split,GridSearchCV
from sklearn.svm import SVC
import plotly
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import re 
import string
import plotly.express as px
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import classification_report
from transformers import BertTokenizer


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

data=pd.read_csv(r"C:\Users\itsla\Downloads\CapstoneProject\Deploymentfiles\Stress.csv")

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Define the pipeline with the TfidfVectorizer and the classifier
pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())

# Define the parameter grid for GridSearchCV
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'tfidfvectorizer__max_features': [None, 1000, 5000],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate the performance on the test set
test_score = grid_search.score(X_test, y_test)
print("Test set accuracy:", test_score)


from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Step 5: Visualizing Decision Regions with PCA
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Convert the sparse matrix to dense array
X_train_dense = X_train_tfidf.todense()

# Convert the dense matrix to NumPy array
X_train_array = np.array(X_train_dense)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_array)

for model in models:
    pipeline = make_pipeline(TfidfVectorizer(), model)
    pipeline.fit(X_train, y_train)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='viridis', legend='full')
    plt.title(f"Decision Regions - {model.__class__.__name__}")
    plt.show()

# Step 6: Visualizing Feature Importance
for model in models:
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = TfidfVectorizer().fit(X_train).get_feature_names_out()
        feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title(f"Feature Importance - {model.__class__.__name__}")
        plt.show()

# Step 7: Classification Report and Confusion Matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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


# Step 8: Prediction on Trained Data
chosen_model = RandomForestClassifier(n_estimators=100)
final_pipeline = make_pipeline(TfidfVectorizer(), chosen_model)
final_pipeline.fit(X, y)

X_valid=vect.fit_transform(input['text'])

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming you have the word_tokenize, remove_stopwords, and lemmatize_words functions defined
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_words(tokens)
    return ' '.join(tokens)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Assuming you have a trained pipeline with SVC
pipeline = make_pipeline(TfidfVectorizer(), SVC())
pipeline.fit(X_train, y_train)

# Assuming X_valid is your new data
X_valid = input['text'].apply(preprocess_text)

# Predict on the new data
input['predicted'] = pipeline.predict(X_valid)
input['predicted'] = input['predicted'].apply(lambda x: 'Stress' if x == 1 else 'Not Stress')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor

# Assuming X_train and X_test are your text data
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Convert the sparse matrices to dense arrays
X_train_dense = X_train_tfidf.todense()
X_test_dense = X_test_tfidf.todense()

# Convert dense arrays to Pandas DataFrames
X_train_df = pd.DataFrame(X_train_dense, columns=tfidf.get_feature_names_out())
X_test_df = pd.DataFrame(X_test_dense, columns=tfidf.get_feature_names_out())

# LazyPredict with DataFrame data
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_df, X_test_df, y_train, y_test)

# Display the results
print(models)
