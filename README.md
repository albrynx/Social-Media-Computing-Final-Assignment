# Social-Media-Computing-Final-Assignment: Emotion Classification on Twitter Messages

Made by:
| Role    | Name                              | Student ID |
| :------ | :---------------------------------| :--------- |
| Student | Muhammad Muzaffar bin Mazlan      | 1211103184 |
| Student | Muhammad Haikal Afiq bin Rafingei | 1211103141 |

## Project Overview

This project focuses on developing an end-to-end Natural Language Processing (NLP) pipeline for the fine-grained task of emotion classification on Twitter messages. Utilizing a dataset annotated with six primary emotions (sadness, joy, love, anger, fear, and surprise), the primary goal is to accurately identify and categorize the specific emotional states conveyed in short, informal text.

The project encompasses several key stages:

* **Data Preprocessing**: Cleaning and preparing raw text data for analysis.

* **Feature Engineering**: Converting textual data into numerical representations suitable for machine learning models (e.g., TF-IDF).

* **Model Implementation**: Applying machine learning algorithms (e.g., Logistic Regression) and discussing advanced deep learning architectures (e.g., BiLSTM, BERT).

* **Aspect-Based Sentiment Analysis (ABSA)**: Identifying specific aspects within texts and determining the sentiment expressed towards them.

* **Visualization and Insight Generation**: Creating an interactive Streamlit dashboard to present analysis results, including emotion distributions, word clouds, model evaluation metrics, and ABSA insights.

## Dataset

The project utilizes the ["Emotions"](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) dataset, a collection of English Twitter messages meticulously annotated with six fundamental emotions: anger, fear, joy, love, sadness, and surprise.

* **File Name**: `emotion.csv`

* **Columns**:

    * (Unnamed: 0 / Index): Original index.

    * `text`: The Twitter message.

    * `label`: The predominant emotion conveyed, mapped as follows:

        * 0: sadness

        * 1: joy

        * 2: love

        * 3: anger

        * 4: fear

        * 5: surprise

This dataset serves as a valuable resource for understanding and analyzing the diverse spectrum of emotions expressed in short-form text on social media. For this project, a subset of `nrows=4000` is used for demonstration and computational efficiency.

## Features

The Streamlit dashboard (`app.py`) provides the following interactive features:

* **Data Overview**: Displays sample data, basic information (e.g., data types, non-null counts), and descriptive statistics of the loaded dataset.

* **Emotion Distribution**: Visualizes the frequency and percentage distribution of the six emotions across the dataset using bar and pie charts.

* **Word Clouds**: Generates word clouds for each emotion category, highlighting the most frequent terms associated with specific emotional states.

* **Model Evaluation**: Allows users to select between Logistic Regression, BiLSTM (Placeholder), and BERT (Placeholder) to view their:

    * Classification Report (precision, recall, f1-score, support).

    * Confusion Matrix (visualizing true vs. predicted labels).

    * *(Note: For BiLSTM and BERT, pre-computed results are shown as placeholders, as the full models are not directly loaded in the dashboard for simplicity and size considerations.)*

* **ABSA (Aspect-Based Sentiment Analysis)**:

    * Displays the top N most frequent opinion targets (aspects) found in the dataset.

    * Shows the sentiment distribution (positive, negative, neutral) for these top aspects, providing granular insights into what specific topics evoke certain sentiments.

    * Offers an interactive text area to input custom text and extract aspect-opinion-sentiment triples in real-time.

## Project Structure
```bash
├── app.py                                       # Streamlit dashboard application
├── emotion.csv                                  # Dataset file
├── final assignment.ipynb                       # Jupyter Notebook for detailed analysis (preprocessing, model training, evaluation)
├── BERT.ipynb                                   # Jupyter Notebook for BERT-specific training/tuning
├── Social Media Assignment Report.docx          # Project report document
├── Corrected Assignment Guidelines Jan 2025.pdf # Assignment guidelines
└── README.md                                    # This file
```

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone [https://github.com/your-username/Social-Media-Computing-Final-Assignment.git](https://github.com/your-username/Social-Media-Computing-Final-Assignment.git)
    cd Social-Media-Computing-Final-Assignment
    ```

    (Replace `your-username` with your actual GitHub username if you fork it)

2.  **Create a virtual environment (recommended)**:

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    Install all required Python libraries.

    ```bash
    pip install streamlit pandas matplotlib seaborn wordcloud spacy nltk scikit-learn numpy
    ```

4.  **Download NLTK Data**:
    The application uses NLTK's `stopwords`, `punkt`, `wordnet`, and `vader_lexicon`. These are automatically downloaded by `app.py` if not found, but you can also download them manually:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    ```

5.  **Download spaCy Model**:
    The ABSA component relies on a spaCy English language model.

    ```bash
    python -m spacy download en_core_web_lg
    ```

    (If `en_core_web_lg` fails, `app.py` will attempt to use `en_core_web_sm` as a fallback, which you can download via `python -m spacy download en_core_web_sm`).

6.  **Place the Dataset**:
    Ensure `emotion.csv` is in the root directory of your project (where `app.py` is located).

## How to Run the Dashboard

Once all dependencies are installed and the `emotion.csv` file is in place, run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

## Results & Visualizations

This section presents the key visualizations generated by our NLP pipeline, showcasing the dataset characteristics and model performance.

### Emotion WordCloud

![visualizations/emotion wordcloud.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/emotion%20wordcloud.png)

### Emotion Distribution in Dataset

![visualizations/emotion distribution in dataset.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/emotion%20distribution%20in%20dataset.png)

### Top 10 Opinion Targets & Words

![visualizations/top 10 opinion targets.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/top%2010%20opinion%20targets.png)
![visualizations/top 10 opinion words.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/top%2010%20opinion%20words.png)

### Top 15 Most Frequent Aspects

![visualizations/top 15 most frequent aspects.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/top%2015%20most%20frequent%20aspects.png)

### Sentiment Distribution for Top 15 Aspects

![visualizations/sentiment distribution for top 15 aspects.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/sentiment%20distribution%20for%20top%2015%20aspects.png)

### Model Evaluation: Logistic Regression, BiLSTM, BERT

![visualizations/confusion matrix logistic regression.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/confusion%20matrix%20logistic%20regression.png)
![visualizations/confusion matrix bilstm.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/confusion%20matrix%20bilstm.png)
![visualizations/confusion_matrix_bert.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/confusion%20matrix%20bert.png)

### Comparison of Accuracy between Models

![visualizations/model accuracy comparison.png](https://github.com/albrynx/Social-Media-Computing-Final-Assignment/blob/main/visualizations/model%20accuracy%20comparison.png)

## Future Work

This project has established a foundational NLP pipeline for emotion and aspect-based sentiment analysis. To further enhance its capabilities and practical utility, we propose the following future work:

* **Live Model Integration & Prediction Interface:** Implement and integrate the fully trained BiLSTM and BERT models into the Streamlit dashboard, replacing current placeholders. This will enable real-time emotion and aspect-based sentiment prediction on user-inputted text, providing an interactive demonstration of their advanced capabilities.

* **Advanced ABSA Implementation:** Explore and integrate more sophisticated Aspect-Based Sentiment Analysis techniques, such as fine-tuned transformer models specifically designed for ABSA (e.g., BERT-ABSA). This would allow for more nuanced contextual sentiment analysis per aspect and potentially the categorization of aspects into broader themes.

* **Real-time Data Ingestion & Monitoring:** Integrate the dashboard with social media APIs (e.g., Twitter API, if accessible) to facilitate real-time data ingestion. This would enable live monitoring of emotional trends and aspect-level sentiments, offering dynamic insights for various applications.

* **Deployment & Scalability:** Deploy the Streamlit application to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS) to make it publicly accessible and to ensure scalability for handling larger volumes of data.

* **User Feedback & Model Improvement:** Incorporate a mechanism for users to provide feedback on the model's predictions. This feedback loop could be utilized to continuously refine and improve the model's accuracy over time.

* **Multilingual Support:** Extend the current framework to support emotion and ABSA in multiple languages, broadening the application's global relevance.
