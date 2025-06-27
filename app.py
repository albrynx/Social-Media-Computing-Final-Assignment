import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import spacy
from collections import Counter, defaultdict
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import io
import numpy as np # Import numpy for confusion matrices

# Download NLTK resources (only run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Define the emotion categories based on your dataset description
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
# Ordered list of emotion names for consistent plotting
EMOTION_NAMES_SORTED = [EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]

st.set_page_config(layout="wide")

st.title("Emotion and Aspect-Based Sentiment Analysis Dashboard")
st.markdown("---")

# --- Load and Cache Data ---
@st.cache_data
def load_data(file_path):
    """
    Loads the dataset from a CSV file.
    Assumes a format similar to 'emotion.csv' with 'text' and 'label' columns.
    """
    try:
        data = pd.read_csv(file_path, index_col=0, nrows=4000)
        # Ensure 'text' column is string type and handle NaNs for processing
        data['text'] = data['text'].astype(str).fillna('')
        data['emotion_name'] = data['label'].map(EMOTION_LABELS) # Map labels to names
        return data
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found. Please upload it or ensure it's in the correct directory.")
        return pd.DataFrame({'text': [], 'label': []})

data = load_data("emotion.csv")

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        nlp_model = spacy.load("en_core_web_lg")
        return nlp_model
    except OSError:
        st.warning("spaCy model 'en_core_web_lg' not found. Attempting to download or use 'en_core_web_sm'. "
                   "Run 'python -m spacy download en_core_web_lg' for better performance.")
        try:
            nlp_model = spacy.load("en_core_web_sm")
            return nlp_model
        except OSError:
            st.error("Neither 'en_core_web_lg' nor 'en_core_web_sm' found. ABSA functionality will be limited.")
            return None

nlp = load_spacy_model()

# --- Preprocessing Function ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, return_tokens=False):
    """
    Preprocesses text by lowercasing, removing punctuation, tokenizing,
    removing stopwords, and lemmatizing.
    """
    lowered = text.lower()
    no_punct = re.sub(f"[{re.escape(string.punctuation)}]", "", lowered)
    tokens = word_tokenize(no_punct)
    no_stopwords = [word for word in tokens if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in no_stopwords]
    if return_tokens:
        return lemmatized
    else:
        return " ".join(lemmatized)

# --- Emotion Classification Model Evaluation Results (Hardcoded from final assignment.ipynb) ---
@st.cache_resource
def get_model_evaluation_results():
    """
    Provides hardcoded classification reports and confusion matrices for
    Logistic Regression, BiLSTM, and BERT, based on the provided results.
    """

    # --- Logistic Regression Results (from user's input) ---
    lr_report = """
    Logistic Regression Classification Report:
                     precision    recall  f1-score   support

           sadness       0.86      0.90      0.88       231
               joy       0.79      0.97      0.87       270
              love       0.93      0.61      0.73        61
             anger       0.92      0.75      0.83       114
              fear       0.77      0.65      0.71        94
          surprise       0.86      0.40      0.55        30

          accuracy                           0.83       800
         macro avg       0.85      0.71      0.76       800
      weighted avg       0.84      0.83      0.82       800
    """
    lr_cm = np.array([
        [207, 13, 0, 5, 6, 0],   # sadness
        [5, 262, 2, 0, 1, 0],    # joy
        [1, 23, 37, 0, 0, 0],    # love
        [16, 9, 0, 86, 3, 0],    # anger
        [10, 18, 1, 2, 61, 2],   # fear
        [2, 8, 0, 0, 8, 12]      # surprise
    ])

    # --- BiLSTM Results (from user's input) ---
    bilstm_report = """
    BiLSTM Classification Report:
                     precision    recall  f1-score   support

           sadness       0.90      0.94      0.92       231
               joy       0.90      0.93      0.92       270
              love       0.87      0.67      0.76        61
             anger       0.89      0.85      0.87       114
              fear       0.81      0.88      0.84        94
          surprise       0.82      0.60      0.69        30

          accuracy                           0.88       800
         macro avg       0.86      0.81      0.83       800
      weighted avg       0.88      0.88      0.88       800
    """
    bilstm_cm = np.array([
        [217, 4, 0, 5, 5, 0],   # sadness
        [7, 251, 6, 2, 2, 2],   # joy
        [2, 17, 41, 1, 0, 0],   # love
        [10, 2, 0, 97, 5, 0],   # anger
        [4, 2, 0, 3, 83, 2],    # fear
        [1, 2, 0, 1, 8, 18]     # surprise
    ])

    # --- BERT Results (from user's input) ---
    bert_report = """
    BERT Classification Report:
                     precision    recall  f1-score   support

           sadness       0.95      0.94      0.95       231
               joy       0.94      0.91      0.93       270
              love       0.74      0.75      0.75        61
             anger       0.91      0.93      0.92       114
              fear       0.84      0.94      0.88        94
          surprise       0.82      0.77      0.79        30

          accuracy                           0.91       800
         macro avg       0.87      0.87      0.87       800
      weighted avg       0.91      0.91      0.91       800
    """
    bert_cm = np.array([
        [217, 2, 4, 5, 3, 0],   # sadness
        [4, 246, 12, 2, 2, 4],  # joy
        [3, 10, 46, 1, 1, 0],   # love
        [1, 1, 0, 106, 6, 0],   # anger
        [2, 1, 0, 2, 88, 1],    # fear
        [1, 1, 0, 0, 5, 23]     # surprise
    ])

    return {
        "Logistic Regression": {"report": lr_report, "cm": lr_cm, "model_classes": list(range(len(EMOTION_LABELS)))},
        "BiLSTM": {"report": bilstm_report, "cm": bilstm_cm, "model_classes": list(range(len(EMOTION_LABELS)))},
        "BERT": {"report": bert_report, "cm": bert_cm, "model_classes": list(range(len(EMOTION_LABELS)))}
    }

# Get all model evaluation results
all_model_results = get_model_evaluation_results()


# --- ABSA Functions ---
def get_sentiment_polarity(text):
    """
    Determines sentiment polarity (positive, negative, neutral) using VADER.
    """
    if not text:
        return "neutral"
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def perform_absa(texts, nlp_model):
    """
    Perform Aspect-Based Sentiment Analysis on a list of texts.
    Extracts aspects, their associated opinion words, and the sentiment polarity of those opinion words.
    """
    aspect_sentiment_data = []
    aspect_counts = Counter()

    if nlp_model is None:
        st.warning("spaCy model not loaded, skipping ABSA analysis.")
        return [], Counter()

    for doc in nlp_model.pipe(texts, disable=["ner"]):
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"]:
                aspect = token.text.lower()
                for child in token.children:
                    if child.pos_ == "ADJ":
                        opinion_word = child.text.lower()
                        polarity = get_sentiment_polarity(opinion_word)
                        aspect_sentiment_data.append({
                            'aspect': aspect,
                            'opinion_word': opinion_word,
                            'polarity': polarity,
                            'source_text': doc.text
                        })
                        aspect_counts[aspect] += 1

            if token.dep_ in ("nsubj", "nsubjpass") and token.head and token.head.pos_ == "ADJ":
                aspect = token.text.lower()
                opinion_word = token.head.text.lower()
                polarity = get_sentiment_polarity(opinion_word)
                aspect_sentiment_data.append({
                    'aspect': aspect,
                    'opinion_word': opinion_word,
                    'polarity': polarity,
                    'source_text': doc.text
                })
                aspect_counts[aspect] += 1
    return aspect_sentiment_data, aspect_counts

def visualize_absa(df_results, aspect_counts, top_n_aspects=15):
    """
    Visualizes ABSA results: Top N aspects and their sentiment distribution.
    """
    if df_results.empty:
        st.write("No aspect-sentiment data to visualize.")
        return

    st.subheader(f"Top {top_n_aspects} Most Frequent Aspects")
    top_aspects_list = aspect_counts.most_common(top_n_aspects)
    if top_aspects_list:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[count for _, count in top_aspects_list],
                    y=[aspect for aspect, _ in top_aspects_list],
                    palette="viridis", ax=ax1)
        ax1.set_title(f"Top {top_n_aspects} Most Frequent Aspects (across all sentiments)")
        ax1.set_xlabel("Frequency of Mentions")
        ax1.set_ylabel("Aspect")
        plt.tight_layout()
        st.pyplot(fig1)
    else:
        st.write("No frequent aspects found for visualization.")

    st.subheader(f"Sentiment Distribution for Top {top_n_aspects} Aspects")
    top_aspect_names = [aspect for aspect, _ in aspect_counts.most_common(top_n_aspects)]
    df_top_absa = df_results[df_results['aspect'].isin(top_aspect_names)].copy()

    if df_top_absa.empty:
        st.write(f"No sentiment data for top {top_n_aspects} aspects to visualize the distribution.")
        return

    sentiment_order = ['negative', 'neutral', 'positive']
    sentiment_colors = {'negative': 'salmon', 'neutral': 'lightgray', 'positive': 'lightgreen'}

    sentiment_distribution = df_top_absa.groupby(['aspect', 'polarity']).size().unstack(fill_value=0)
    sentiment_distribution = sentiment_distribution.reindex(columns=sentiment_order, fill_value=0)
    sentiment_distribution['Total'] = sentiment_distribution.sum(axis=1)
    sentiment_distribution = sentiment_distribution.sort_values(by='Total', ascending=True)
    sentiment_distribution.drop(columns='Total', inplace=True)

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sentiment_distribution.plot(kind='barh', stacked=True, ax=ax2,
                                color=[sentiment_colors[col] for col in sentiment_distribution.columns])
    ax2.set_title(f"Sentiment Distribution for Top {top_n_aspects} Aspects")
    ax2.set_xlabel("Number of Opinion Mentions")
    ax2.set_ylabel("Aspect")
    ax2.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig2)


# --- Dashboard Sections ---
st.sidebar.header("Navigation")
analysis_options = ["Data Overview", "Emotion Distribution", "Word Clouds", "Model Evaluation", "ABSA"]
selected_analysis = st.sidebar.radio("Go to", analysis_options)

if data.empty:
    st.warning("Please ensure 'emotion.csv' is available to proceed with analysis.")
else:
    if selected_analysis == "Data Overview":
        st.header("Dataset Overview")
        st.write("First 5 rows of the dataset:")
        st.dataframe(data.head())
        st.write(f"Total number of entries: {len(data)}")
        st.write("Dataset Information:")
        buffer = io.StringIO()
        data.info(verbose=True, show_counts=True, buf=buffer)
        s = buffer.getvalue()
        st.code(s)
        st.write("Descriptive Statistics:")
        st.dataframe(data.describe())

    elif selected_analysis == "Emotion Distribution":
        st.header("Emotion Distribution")
        emotion_counts = data['label'].map(EMOTION_LABELS).value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette='viridis', ax=ax)
        ax.set_title("Distribution of Emotions in the Dataset")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Number of Tweets")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    elif selected_analysis == "Word Clouds":
        st.header("Word Clouds per Emotion")
        for label_id, emotion_name in EMOTION_LABELS.items():
            st.subheader(f"Word Cloud for: {emotion_name.capitalize()}")
            emotion_texts = data[data['label'] == label_id]['text']
            preprocessed_emotion_texts = emotion_texts.apply(preprocess_text)
            long_string = " ".join(preprocessed_emotion_texts.tolist())

            if long_string:
                wordcloud = WordCloud(width=800, height=400, background_color="white",
                                      colormap='viridis', max_words=100).generate(long_string)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.write(f"No text available for '{emotion_name}' to generate a word cloud.")

    elif selected_analysis == "Model Evaluation":
        st.header("Emotion Classification Model Evaluation")
        
        # Dropdown for model selection
        model_options = list(all_model_results.keys())
        selected_model = st.selectbox("Select a Model for Evaluation:", model_options)

        # Retrieve results for the selected model
        model_data = all_model_results.get(selected_model)

        if model_data:
            st.write(f"#### Classification Report ({selected_model})")
            st.code(model_data["report"])

            st.write(f"#### Confusion Matrix ({selected_model})")
            cm = model_data["cm"]
            model_classes = model_data["model_classes"]
            
            if len(model_classes) > 0 and cm.shape[0] == len(model_classes):
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=[EMOTION_LABELS[i] for i in model_classes],
                            yticklabels=[EMOTION_LABELS[i] for i in model_classes])
                ax_cm.set_xlabel('Predicted Label')
                ax_cm.set_ylabel('True Label')
                ax_cm.set_title(f'Confusion Matrix for {selected_model}')
                plt.tight_layout()
                st.pyplot(fig_cm)
            else:
                st.warning(f"Could not display confusion matrix for {selected_model}. Data might be incomplete or dimensions mismatch.")
                st.dataframe(pd.DataFrame(cm))
        else:
            st.warning(f"No evaluation data available for {selected_model}.")

    elif selected_analysis == "ABSA":
        st.header("Aspect-Based Sentiment Analysis")
        st.write("This section identifies specific aspects (nouns/topics) mentioned in tweets and analyzes the sentiment (positive, negative, neutral) expressed towards them.")

        sample_texts_for_absa = data['text'].sample(n=min(1000, len(data)), random_state=42).tolist()
        
        if nlp:
            absa_results, absa_counts = perform_absa(sample_texts_for_absa, nlp)
            df_absa_results = pd.DataFrame(absa_results)
            visualize_absa(df_absa_results, absa_counts)
            
            st.subheader("Raw ABSA Results Sample")
            st.dataframe(df_absa_results.head(10))
        else:
            st.error("SpaCy model could not be loaded. ABSA functionality is unavailable.")
