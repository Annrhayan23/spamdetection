import nltk

# Download all required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # Sometimes needed
nltk.download('omw-1.4')  # Open Multilingual WordNet
nltk.download('averaged_perceptron_tagger')

# Try to download punkt_tab if available
try:
    nltk.download('punkt_tab')
except:
    print("punkt_tab not available - using fallback methods")

print("All required NLTK data downloaded successfully!")