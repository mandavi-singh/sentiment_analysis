# -*- coding: utf-8 -*-
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import random
from nltk.corpus import wordnet
import nltk

# Download WordNet for synonym replacement
nltk.download('wordnet')
nltk.download('omw-1.4')

# ======================
# 📌 Load Data
# ======================
sentiment = pd.read_csv("C:\\Users\\singh\\Downloads\\sentiment_analysis\\3) Sentiment dataset.csv")
data = sentiment[['Text','Sentiment']].dropna()

# ======================
# 📌 Clean Text
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

data['Clean_Text'] = data['Text'].apply(clean_text)
data['Sentiment'] = data['Sentiment'].str.strip().str.lower()

# ======================
# 📌 Map Sentiment to Labels
# ======================
def map_sentiment(s):
    if s == "positive":
        return 1
    elif s == "negative":
        return 0
    else:
        return 2  # Neutral

data['label'] = data['Sentiment'].apply(map_sentiment)
print("Class Distribution:\n", data['label'].value_counts())

# ======================
# 📌 Text Augmentation: Synonym Replacement for Negative Class
# ======================
def synonym_replacement(sentence, n=2):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for word in random_word_list:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            if synonym != word:
                new_words = [synonym if w==word else w for w in new_words]
                num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

# Augment Negative samples
negative_samples = data[data['label']==0]['Clean_Text'].tolist()
augmented_negatives = [synonym_replacement(text) for text in negative_samples for _ in range(5)]  # 5x augmentation

augmented_labels = [0]*len(augmented_negatives)
augmented_df = pd.DataFrame({'Clean_Text': augmented_negatives, 'label': augmented_labels})

# Combine with original dataset
data_aug = pd.concat([data[['Clean_Text','label']], augmented_df], ignore_index=True)

# ======================
# 📌 Train/Test Split
# ======================
X_train, X_test, Y_train, Y_test = train_test_split(
    data_aug['Clean_Text'], data_aug['label'], test_size=0.2, random_state=42, stratify=data_aug['label']
)

# ======================
# 📌 Vectorization
# ======================
vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# ======================
# 📌 Handle Class Imbalance (SMOTE)
# ======================
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_res, Y_train_res = smote.fit_resample(X_train_vec, Y_train)

print("Before SMOTE:", Y_train.value_counts())
print("After SMOTE:", pd.Series(Y_train_res).value_counts())

# ======================
# 📌 Compute Class Weights for Resampled Data
# ======================
classes = np.unique(Y_train_res)
weights = compute_class_weight('balanced', classes=classes, y=Y_train_res)
class_weights = dict(zip(classes, weights))
sample_weights = np.array([class_weights[i] for i in Y_train_res])

# ======================
# 📌 Train XGBoost Model with Class Weights
# ======================
model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train_res, Y_train_res, sample_weight=sample_weights)

# ======================
# 📌 Predictions
# ======================
Y_pred = model.predict(X_test_vec)
Y_prob = model.predict_proba(X_test_vec)

# ======================
# 📌 Evaluation
# ======================
print("\nAccuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n",
      classification_report(Y_test, Y_pred, labels=[0,1,2],
                            target_names=["Negative", "Positive", "Neutral"], zero_division=0))
print("\nPrecision (macro):", precision_score(Y_test, Y_pred, average="macro", zero_division=0))
print("Recall (macro):", recall_score(Y_test, Y_pred, average="macro", zero_division=0))
print("ROC AUC (ovr):", roc_auc_score(Y_test, Y_prob, multi_class="ovr"))
print("ROC AUC (ovo):", roc_auc_score(Y_test, Y_prob, multi_class="ovo"))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# ======================
# 📌 Save Model & Vectorizer
# ======================
with open("sentiment_model_xgb_augmented.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vec, f)

print("\n✅ Augmented XGBoost Model and Vectorizer saved successfully!")

