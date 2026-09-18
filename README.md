# Twitter Sentiment Analysis

A machine learning project for classifying tweets as **positive** or
**negative** using the **Sentiment140 dataset**. The project converts
tweet text into TF-IDF features and compares three classification
models: **Bernoulli Naive Bayes, Linear Support Vector Machine
(LinearSVC), and Logistic Regression**.

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to
determine the sentiment expressed in text.

This project implements a simple text-classification pipeline:

``` text
Sentiment140 Dataset
        ↓
Select Polarity and Tweet Text
        ↓
Remove Neutral Tweets
        ↓
Map Labels: 0 → Negative, 4 → Positive
        ↓
Basic Text Cleaning
        ↓
Train-Test Split (80:20)
        ↓
TF-IDF Vectorization
        ↓
 ┌──────────────┬──────────────┬──────────────────┐
 │ Bernoulli NB │   Linear SVM  │ Logistic         │
 │              │   (LinearSVC) │ Regression       │
 └──────────────┴──────────────┴──────────────────┘
        ↓
Accuracy & Classification Reports
        ↓
Custom Tweet Predictions
```

## Dataset

The project uses the **Sentiment140** dataset.

The notebook loads:

``` text
training.1600000.processed.noemoticon.csv.zip
```

using Latin-1 encoding.

The dataset contains tweet polarity and tweet text. The notebook
selects:

-   Column `0` → `polarity`
-   Column `5` → `text`

Only positive and negative tweets are retained:

``` text
0 → Negative
4 → Positive
```

Neutral tweets (`2`) are removed.

## Text Preprocessing

The notebook applies a basic preprocessing step to the tweet text:

``` python
def clean_text(text):
    return text.lower()
```

Thus, the text is converted to lowercase before feature extraction.

## Train-Test Split

The dataset is divided into training and testing sets using:

``` python
train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)
```

Therefore:

-   **80%** of the data is used for training.
-   **20%** is used for testing.
-   `random_state=42` is used for reproducibility.

## Feature Extraction

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)**
to convert tweets into numerical feature vectors.

The vectorizer is configured as:

``` python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)
```

### Configuration

  Parameter                           Value
  -------------------- --------------------
  Maximum features                    5,000
  N-grams                Unigrams + Bigrams
  Feature extraction                 TF-IDF

### Unigrams and Bigrams

The model considers:

-   **Unigrams** --- individual words
-   **Bigrams** --- pairs of consecutive words

For example:

``` text
"I love this movie"
```

can produce features such as:

``` text
i
love
this
movie
i love
love this
this movie
```

## Machine Learning Models

Three classification algorithms are trained and compared.

### 1. Bernoulli Naive Bayes

``` python
bnb = BernoulliNB()
```

The trained model predicts whether a tweet belongs to the negative or
positive sentiment class.

### 2. Linear Support Vector Machine

The project uses `LinearSVC`:

``` python
svm = LinearSVC(max_iter=1000)
```

The classifier is trained on the TF-IDF representation of the tweets.

### 3. Logistic Regression

The project also trains:

``` python
logreg = LogisticRegression(max_iter=100)
```

It predicts the sentiment class from the TF-IDF features.

## Model Evaluation

The models are evaluated using:

### Accuracy

Accuracy measures the proportion of correctly classified tweets.

``` text
Accuracy = Correct Predictions / Total Predictions
```

The notebook calculates accuracy using:

``` python
accuracy_score(y_test, predictions)
```

### Classification Report

For each model, the notebook generates a classification report
containing:

-   Precision
-   Recall
-   F1-score
-   Support

The reports are generated using:

``` python
classification_report(y_test, predictions)
```

## Custom Tweet Prediction

After training, the models are tested on custom tweets:

``` python
sample_tweets = [
    "I love this!",
    "I hate that!",
    "It was okay, not great."
]
```

The tweets are transformed using the same TF-IDF vectorizer and
predictions are produced by all three trained models.

## Technologies Used

-   Python
-   Google Colab
-   Pandas
-   Scikit-learn
-   TF-IDF
-   Bernoulli Naive Bayes
-   Linear SVM (`LinearSVC`)
-   Logistic Regression
-   Sentiment140 Dataset

## Installation

Install the required Python packages with:

``` bash
pip install pandas scikit-learn
```

## How to Run

### 1. Open the notebook

Open:

``` text
sentimentanalysis.ipynb
```

in Google Colab or Jupyter Notebook.

### 2. Upload the dataset

The first notebook cell uses Google Colab's file upload functionality:

``` python
from google.colab import files
uploaded = files.upload()
```

Upload:

``` text
training.1600000.processed.noemoticon.csv.zip
```

### 3. Run the notebook

Execute the cells sequentially.

The notebook will:

1.  Load the dataset.
2.  Select polarity and tweet text.
3.  Remove neutral tweets.
4.  Convert labels to negative/positive classes.
5.  Convert tweets to lowercase.
6.  Split the data into training and testing sets.
7.  Generate TF-IDF features.
8.  Train the three classifiers.
9.  Calculate accuracy.
10. Generate classification reports.
11. Predict sentiment for custom tweets.

## Project Structure

``` text
twitter-sentiment-analysis/
│
├── sentimentanalysis.ipynb
├── README.md
└── training.1600000.processed.noemoticon.csv.zip
```

The dataset can be omitted from the repository if its redistribution is
not appropriate. In that case, download/obtain the dataset separately
and upload it through the notebook when running the project.

## Reproducibility

The train-test split uses:

``` python
random_state=42
```

This makes the split reproducible when the same data and environment are
used.

## Limitations

The current notebook uses relatively simple text preprocessing. The
cleaning function only converts text to lowercase; it does not perform
additional operations such as stemming, lemmatization, stop-word
removal, or explicit URL/mention/hashtag processing.

The project also uses TF-IDF features with a maximum of 5,000 features
and compares traditional machine-learning classifiers rather than
deep-learning or transformer-based NLP models.

## Future Improvements

Possible extensions include:

-   More advanced tweet preprocessing.
-   Hyperparameter tuning.
-   Larger or different TF-IDF feature spaces.
-   Confusion-matrix visualization.
-   Cross-validation.
-   Word embeddings.
-   LSTM-based sentiment classification.
-   Transformer-based models such as BERT.
-   Model and vectorizer serialization for deployment.
-   A web interface for real-time sentiment prediction.

## Author

**Suva Pratyush Tripathy**

B.Tech --- Electrical and Electronics Engineering\
IIIT Bhubaneswar
