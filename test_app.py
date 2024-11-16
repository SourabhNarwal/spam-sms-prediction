import pytest
import pickle
from app import transform_text

# Load the vectorizer and model for testing
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def test_transform_text():
    """Test the text transformation function."""
    # Sample input
    input_text = "Hello!! How are you doing today??"
    
    # Expected behavior: lowercasing, punctuation removal, stopwords removal, stemming
    expected_output = "hello today"
    assert transform_text(input_text) == expected_output, "Text transformation failed!"

def test_transform_empty_text():
    """Test transformation with an empty input."""
    input_text = ""
    expected_output = ""
    assert transform_text(input_text) == expected_output, "Empty input transformation failed!"

def test_spam_prediction():
    """Test the prediction function with a spam message."""
    input_sms = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim."
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    
    # Since it's a typical spam message, the model should predict 1 (spam)
    result = model.predict(vector_input)[0]
    assert result == 1, "The model failed to classify a spam message."

def test_ham_prediction():
    """Test the prediction function with a non-spam message."""
    input_sms = "Hey, are we still meeting up tomorrow?"
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    
    # Since it's a normal message, the model should predict 0 (not spam)
    result = model.predict(vector_input)[0]
    assert result == 0, "The model failed to classify a non-spam message."

def test_prediction_with_empty_input():
    """Test the prediction function with an empty input."""
    input_sms = ""
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    
    # Should ideally classify as not spam (0)
    result = model.predict(vector_input)[0]
    assert result == 0, "The model misclassified an empty input."

