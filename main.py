import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
import os

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')

# App title and description
st.title("📧 Spam Detection App")
st.write("""
This app detects whether an email/message is spam or not using Machine Learning.
Upload your dataset or try it with example messages!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Try Demo", "Train Model", "Batch Predict"])

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    try:
        words = nltk.word_tokenize(text)
    except:
        words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

# Try Demo Mode
if app_mode == "Try Demo":
    st.header("Try with Your Own Message")
    
    # Load model if exists
    if os.path.exists('spam_model.pkl'):
        with open('spam_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        vectorizer = model_data['vectorizer']
        classifier = model_data['classifier']
        
        # Input text
        user_input = st.text_area("Enter a message to check if it's spam:")
        
        if st.button("Check"):
            if user_input:
                # Preprocess and predict
                processed_text = preprocess_text(user_input)
                text_vec = vectorizer.transform([processed_text])
                prediction = classifier.predict(text_vec)[0]
                
                # Display result
                if prediction == 1:
                    st.error("🚨 This is SPAM!")
                else:
                    st.success("✅ This is NOT spam (ham)")
                
                # Show confidence
                proba = classifier.predict_proba(text_vec)[0]
                st.write(f"Confidence: {max(proba)*100:.2f}%")
            else:
                st.warning("Please enter a message")
    else:
        st.warning("No trained model found. Please train a model first.")

# Train Model Mode
elif app_mode == "Train Model":
    st.header("Train Your Spam Detection Model")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            df = df[['v1', 'v2']]  # Select relevant columns
            df.columns = ['label', 'text']
            
            st.success("Dataset loaded successfully!")
            st.write(f"Total messages: {len(df)}")
            st.write("Sample data:")
            st.dataframe(df.head())
            
            # Preprocess
            st.write("Preprocessing texts...")
            df['processed_text'] = df['text'].apply(preprocess_text)
            
            # Prepare data
            X = df['processed_text']
            y = df['label'].map({'ham': 0, 'spam': 1})
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Vectorize
            vectorizer = TfidfVectorizer(max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            # Train
            classifier = MultinomialNB()
            classifier.fit(X_train_vec, y_train)
            
            # Evaluate
            y_pred = classifier.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success("Model trained successfully!")
            st.write(f"Accuracy: {accuracy:.2%}")
            
            # Save model
            model_data = {
                'vectorizer': vectorizer,
                'classifier': classifier
            }
            with open('spam_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
                
            st.success("Model saved as 'spam_model.pkl'")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Batch Predict Mode
elif app_mode == "Batch Predict":
    st.header("Batch Prediction")
    
    if os.path.exists('spam_model.pkl'):
        with open('spam_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        vectorizer = model_data['vectorizer']
        classifier = model_data['classifier']
        
        uploaded_file = st.file_uploader("Upload messages to classify (CSV with 'text' column)", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                else:
                    # Preprocess and predict
                    st.write("Processing messages...")
                    df['processed_text'] = df['text'].apply(preprocess_text)
                    text_vec = vectorizer.transform(df['processed_text'])
                    predictions = classifier.predict(text_vec)
                    
                    # Add predictions to dataframe
                    df['prediction'] = predictions
                    df['prediction'] = df['prediction'].map({0: 'ham', 1: 'spam'})
                    
                    st.success("Predictions complete!")
                    st.dataframe(df)
                    
                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "spam_predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("No trained model found. Please train a model first.")

# Example messages
if app_mode == "Try Demo":
    st.subheader("Example Messages")
    examples = [
        ("Hey, are we still meeting for lunch tomorrow?", "ham"),
        ("Congratulations! You've won a $1000 gift card. Click here to claim!", "spam"),
        ("Your account has been compromised. Secure it now!", "spam"),
        ("Hi Mom, I'll be home late tonight", "ham")
    ]
    
    for msg, label in examples:
        if st.button(f"{msg[:30]}... ({label})"):
            processed_text = preprocess_text(msg)
            text_vec = vectorizer.transform([processed_text])
            prediction = classifier.predict(text_vec)[0]
            
            if prediction == 1:
                st.error(f"Prediction: SPAM (actual: {label})")
            else:
                st.success(f"Prediction: HAM (actual: {label})")