# SMS_Spam_Detection
Detect spam messages using Logistic Regression. Includes exploratory data analysis, feature engineering (like detecting links, money mentions, and urgency), TF-IDF vectorization, model training with Scikit-learn, and deployment-ready model serialization using pickle.



- Exploratory Data Analysis (EDA)
- Data balancing
- Feature engineering
- Text vectorization using TF-IDF
- Training a **Logistic Regression** model
- Testing with custom messages
- Model serialization with `pickle`

***

### Dataset
- Source: [spam.tsv](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/spam.tsv)
- Contains 5,572 SMS messages labeled as either `ham` (non-spam) or `spam`.

***

### Exploratory Data Analysis (EDA)
- Distribution of message length and punctuation for `ham` and `spam`.
- Data balancing using undersampling to address class imbalance.
- Visualization of feature distribution using `matplotlib`.

***

### Feature Engineering
Custom features extracted from messages:
- `has_link`: Contains URLs (http, .com, etc.)
- `has_money`: Contains money mentions (`$100`, etc.)
- `has_urgent_words`: Detects urgent language like `urgent`, `win`, `immediately`

***

### Model Building
- **TF-IDF Vectorizer** for text features
- **StandardScaler** for engineered features
- Combined using `ColumnTransformer`
- **Logistic Regression** trained using scikit-learn pipeline

***

### Performance
Evaluated using:
- Precision, Recall, F1-Score
- Train/Test split with stratification
- Sample prediction: ✅ detects spam from user-provided message

***

### Example Test Message
```python
test_msg = "Congratulations! You've won $1000. Click the link to claim now!"
