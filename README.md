# Customer Review Sentiment Analysis System

## 📌 Project Overview

**Customer Review Sentiment Analysis System** is a Natural Language Processing (NLP) and Machine Learning project that analyzes customer/user reviews and predicts their sentiment.

The system processes textual reviews, cleans the text, converts the reviews into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**, and uses a **Logistic Regression** machine learning model to classify reviews according to their sentiment.

The project focuses on three main sentiment categories:

* 🟢 Positive
* 🔴 Negative
* 🟡 Neutral

The complete workflow is implemented in a **Jupyter Notebook** using Python and popular data science and machine learning libraries.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze textual customer reviews using NLP techniques.
2. Prepare and clean raw review data.
3. Standardize sentiment labels from different datasets.
4. Combine suitable review datasets into a common format.
5. Remove duplicate and empty reviews.
6. Convert text into numerical features using TF-IDF.
7. Train a Machine Learning classification model.
8. Predict sentiment for unseen reviews.
9. Evaluate model performance using accuracy and classification metrics.
10. Visualize model performance using a confusion matrix.

---

## 🧠 Technologies Used

| Technology          | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| Python              | Main programming language                      |
| Pandas              | Data loading and manipulation                  |
| NumPy               | Numerical operations                           |
| Matplotlib          | Data visualization                             |
| Seaborn             | Statistical visualization and confusion matrix |
| Scikit-learn        | Machine Learning and evaluation                |
| NLP                 | Text processing and sentiment analysis         |
| TF-IDF              | Text feature extraction                        |
| Logistic Regression | Sentiment classification                       |
| Jupyter Notebook    | Development environment                        |

---

## 📂 Datasets Used

Five CSV datasets were initially explored during the project.

### Dataset 1

**Airbnb Reviews Data Dictionary**

This dataset contains field names and descriptions rather than review text and sentiment labels. Therefore, it was not used for sentiment model training.

### Dataset 2

**Amazon Ratings**

This dataset contains:

* User-ID
* ISBN
* Book-Rating

It does not contain textual review data, so it was not used for the NLP sentiment model.

### Dataset 3

**IMDB Dataset**

Contains:

* `review`
* `sentiment`

The dataset contains:

* 50,000 reviews
* 25,000 Positive reviews
* 25,000 Negative reviews

This dataset provides labeled textual reviews suitable for sentiment analysis.

### Dataset 4

**Twitter Training Dataset**

Contains:

* ID
* Topic
* Sentiment
* Review

The original sentiment categories include:

* Positive
* Negative
* Neutral
* Irrelevant

The dataset was loaded using `header=None` and its columns were standardized before processing.

### Dataset 5

**Women's Clothing E-Commerce Reviews**

Contains customer review information including:

* Review Text
* Rating
* Recommended IND
* Product/category information

The `Review Text` column was used for NLP processing.

Since this dataset does not originally contain a sentiment column, sentiment labels were derived from the rating:

* Rating 1–2 → Negative
* Rating 3 → Neutral
* Rating 4–5 → Positive

This rating-to-sentiment mapping is a derived labeling rule used for this project.

---

## 🔄 Project Workflow

```text
          Review Datasets
                 ↓
        Dataset Exploration
                 ↓
       Data Cleaning & Preparation
                 ↓
       Standardize Sentiment Labels
                 ↓
       Combine Suitable Datasets
                 ↓
        Remove Duplicate Reviews
                 ↓
         Text Preprocessing
                 ↓
             TF-IDF
                 ↓
        Train-Test Split
                 ↓
       Logistic Regression
                 ↓
       Sentiment Prediction
                 ↓
     Model Performance Evaluation
                 ↓
      New Review Classification
```

---

## 🧹 Data Preprocessing

The textual review data is cleaned before applying Machine Learning.

The preprocessing includes:

* Converting text to lowercase
* Removing HTML tags
* Removing special characters
* Removing unnecessary spaces
* Removing empty reviews
* Standardizing sentiment labels
* Removing duplicate reviews

Example:

```text
Original:
A wonderful little production. <br /><br />The story was amazing!

Cleaned:
a wonderful little production the story was amazing
```

---

## 🔤 TF-IDF Feature Extraction

Machine Learning algorithms cannot directly understand raw text.

Therefore, the cleaned reviews are converted into numerical feature vectors using **TF-IDF**.

TF-IDF gives higher importance to words that are useful for distinguishing documents while reducing the importance of words that occur very frequently across documents.

The project uses:

```python
TfidfVectorizer(max_features=10000)
```

The TF-IDF vectorizer is fitted on the training data and then used to transform both training and testing data.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as the primary classification algorithm.

Logistic Regression is suitable for this project because it is:

* Simple to understand
* Efficient for text classification
* Suitable for high-dimensional TF-IDF features
* Commonly used for sentiment classification
* Easy to train and evaluate

The model is trained using the TF-IDF features extracted from the training reviews.

---

## 📊 Model Evaluation

The trained model is evaluated using standard classification metrics.

### Accuracy

Accuracy measures the percentage of correctly classified reviews.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Classification Report

The project evaluates:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

A confusion matrix is used to understand how many reviews were correctly and incorrectly classified for each sentiment category.

The matrix helps identify cases where:

* Positive reviews are predicted as Negative
* Negative reviews are predicted as Positive
* Neutral reviews are predicted as Positive or Negative

---

## 🧪 Testing with New Reviews

After training, the model can classify new unseen reviews.

Example:

```text
"The product is amazing and works perfectly."
→ Positive

"The quality is terrible and disappointing."
→ Negative

"The product is okay, nothing special."
→ Neutral
```

The actual prediction depends on the trained model and its learned vocabulary/features.

---

## 📈 Key Data Preparation

The project combined the prepared review datasets into a common structure:

```text
review | sentiment
```

This common structure makes it possible to apply the same preprocessing and Machine Learning pipeline to different review sources.

Duplicate reviews were removed during preparation.

The notebook shows a combined dataset of **147,323 records before duplicate removal** and **141,708 records after duplicate removal** at that stage of processing.

---

## 🗂️ Project Structure

Recommended GitHub repository structure:

```text
Customer-Review-Sentiment-Analysis/
│
├── # Customer Review Sentiment Analysis.ipynb
│
├── datasets/
│   ├── IMDB-Dataset.csv
│   ├── twitter_training.csv
│   └── Womens Clothing E-Commerce Reviews.csv
│
├── images/
│   ├── sentiment_distribution.png
│   ├── confusion_matrix.png
│   └── predictions.png
│
├── README.md
│
└── requirements.txt
```

> Dataset files may be omitted from GitHub if they are very large or have redistribution restrictions. In that case, provide the original dataset source/reference in the README.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Customer-Review-Sentiment-Analysis.git
```

Move into the project folder:

```bash
cd Customer-Review-Sentiment-Analysis
```

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Minor_Project.ipynb
```

Run the notebook cells in order.

---

## 📦 Requirements

The main Python libraries used are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 💡 Applications

Sentiment analysis can be useful for:

* Customer feedback analysis
* Product review monitoring
* Brand reputation analysis
* Social media analysis
* Customer satisfaction monitoring
* Product improvement
* Service quality analysis
* Market research

Companies can use sentiment information to understand whether customers are generally satisfied, dissatisfied, or neutral toward their products or services.

---

## 👥 End Users

Potential end users include:

* Businesses
* E-commerce companies
* Product managers
* Marketing teams
* Customer support teams
* Market researchers
* Data analysts
* Business analysts

---

## ⚠️ Limitations

This project is a basic academic NLP and Machine Learning implementation.

Some limitations include:

* Sentiment prediction depends on the quality of the training data.
* Sarcasm can be difficult to identify.
* Context-dependent statements may be incorrectly classified.
* Mixed-sentiment reviews can be challenging.
* Dataset domains are different, such as movies, social media, and clothing reviews.
* Dataset 5 sentiment labels are derived from ratings rather than originally provided sentiment labels.
* The model uses traditional TF-IDF features rather than advanced transformer-based NLP models.

---

## 🚀 Future Improvements

The project can be extended with:

1. Naive Bayes model comparison.
2. Support Vector Machine classification.
3. Hyperparameter tuning.
4. N-gram based TF-IDF features.
5. Word embeddings.
6. Advanced NLP models such as BERT.
7. Sentiment prediction through a web application.
8. Real-time review sentiment analysis.
9. Interactive dashboards.
10. Deployment using Streamlit or Flask.

---

## 📚 Learning Outcomes

Through this project, the following concepts were practiced:

* Data loading using Pandas
* Dataset exploration
* Missing-value analysis
* Data cleaning
* Duplicate removal
* Text preprocessing
* Sentiment label standardization
* Natural Language Processing
* TF-IDF feature extraction
* Train-test splitting
* Logistic Regression
* Classification
* Accuracy calculation
* Classification reports
* Confusion matrices
* Prediction on new text

---

## 🏁 Conclusion

The Customer Review Sentiment Analysis System demonstrates how Natural Language Processing and Machine Learning can be used to automatically analyze textual reviews. Multiple review datasets were explored and suitable datasets were transformed into a common format. Text preprocessing was performed to remove unnecessary elements and prepare the reviews for modeling. TF-IDF was then used to convert textual data into numerical features, followed by Logistic Regression for sentiment classification. The model was evaluated using accuracy, classification metrics, and a confusion matrix. The project provides a basic foundation for building more advanced customer feedback and sentiment analysis systems.

---

## 👨‍💻 Author

**Niraj Khankari**

### Project

**Customer Review Sentiment Analysis System**

### Domain

**Natural Language Processing | Machine Learning | Data Science**

