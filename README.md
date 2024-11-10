# 📰 True Fake News Detection using LSTM

This project is a machine learning application that uses Natural Language Processing (NLP) and Long Short-Term Memory (LSTM) networks to classify news articles as **True** or **Fake**. The project provides a user-friendly interface built with **Streamlit** where users can input text to get a real-time prediction.

## 🚀 Project Overview

This project tackles the challenge of detecting fake news using advanced Natural Language Processing (NLP) and deep learning techniques. Leveraging a dataset of 40,000 rows containing both true and fake news articles, the model classifies input text as either True or Fake news with high accuracy.

The solution is powered by a Long Short-Term Memory (LSTM) neural network, which excels at capturing patterns in sequential data, making it a suitable choice for text classification tasks. Additionally, a user-friendly interface built with Streamlit allows users to input any news article and get a real-time prediction on its authenticity.

<div align="center">
    <img src="https://github.com/user-attachments/assets/8d9af2b2-e635-413f-9b05-cdc2ee628fa1" alt="n 1" width="900"/>
    <img src="https://github.com/user-attachments/assets/69971ead-1dcb-4e8e-9c31-f64440d4c1dc" alt="n 2" width="400"/>
    <img src="https://github.com/user-attachments/assets/c8f1e095-a341-42a2-b0da-e4a973c92581" alt="n 3" width="400"/>
</div>

## 📂 Datasets Used

- **True.csv**: Contains articles labeled as true news.
- **Fake.csv**: Contains articles labeled as fake news.
- Both datasets are combined into a single DataFrame with a `category` column (1 for True, 0 for Fake).

## 🧰 Technologies & Libraries

- **Python**: Programming language.
- **TensorFlow/Keras**: For building the LSTM model.
- **Streamlit**: For creating the interactive user interface (`app.py`).
- **Numpy & Pandas**: Data manipulation and analysis.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **BeautifulSoup**: For cleaning HTML content from text data.
- **Scikit-learn**: For model evaluation and data splitting.
- **Matplotlib & Seaborn**: For data visualization.

## 🛠️ Project Workflow

1. **Data Preprocessing**:
    - Removed HTML tags, URLs, stopwords, and punctuation from the text.
    - Used NLTK for tokenization, stopword removal, and text cleaning.
    - Applied **Lemmatization** to reduce words to their base form.

2. **Data Preparation**:
    - Labeled the dataset with 1 (True) and 0 (Fake).
    - Split the data into training and testing sets using `train_test_split`.
    - Tokenized the text using Keras `Tokenizer` and padded sequences to a maximum length of 300.

3. **Model Architecture**:
    - Used an **Embedding Layer** to convert text data into dense vectors.
    - Added two **LSTM layers**:
        - First LSTM layer with 64 units and `return_sequences=True`.
        - Second LSTM layer with 32 units for deeper learning.
    - Added **Dropout layers** to prevent overfitting.
    - Final **Dense layer** with a sigmoid activation function for binary classification.
    ```python
    model = Sequential()
    model.add(Embedding(input_dim=max_feature, output_dim=128, input_length=maxlen))
    model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
    model.add(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    ```

4. **Model Compilation & Training**:
    - Compiled with `adam` optimizer and `binary_crossentropy` loss.
    - Added a `ReduceLROnPlateau` callback to reduce learning rate on plateau.
    - Trained for 5 epochs with a batch size of 64.

5. **Model Evaluation**:
    - Evaluated the model's performance using accuracy, precision, recall, F1-score, and confusion matrix.

## 📊 Evaluation Metrics

The model achieved impressive performance on the test set, with the following evaluation metrics:

|                | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| **Fake**       | 1.00      | 0.98   | 0.99     | 5858    |
| **Not Fake**   | 0.98      | 1.00   | 0.99     | 5367    |
| **Accuracy**   |           |        | 0.99     | 11225   |
| **Macro Avg**  | 0.99      | 0.99   | 0.99     | 11225   |
| **Weighted Avg** | 0.99    | 0.99   | 0.99     | 11225   |

These metrics demonstrate that the model is highly accurate in distinguishing between fake and true news articles.

## 📈 Results

- **Accuracy**: 99%
- The model shows strong generalization on unseen data, with balanced precision and recall for both classes.

## 💻 How to Run the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/true-fake-news-detection.git
    cd true-fake-news-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

4. **Navigate to** `http://localhost:8501` in your web browser.

## 📝 Future Improvements

- Implement **BERT** or other transformer-based models for better performance.
- Integrate additional datasets for more diverse training.
- Optimize the Streamlit UI for a better user experience.

## 🤝 Contributing

Contributions are welcome! Feel free to open a pull request or issue.

## 🧑‍💻 Author

- **Jay Narigara** - [LinkedIn]([https://www.linkedin.com/in/jaynarigara/]) | [GitHub](https://github.com/jaynarigara91/)
