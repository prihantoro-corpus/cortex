import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.preprocessing.tagging import split_sentences_custom

def test_splitter():
    text = "Hello world. How are you? I'm fine! This has no punctuation But this one does."
    expected = [
        "Hello world.",
        "How are you?",
        "I'm fine!",
        "This has no punctuation But this one does."
    ]
    
    result = split_sentences_custom(text)
    print(f"Text: {text}")
    print(f"Result: {result}")
    
    assert len(result) == 4
    assert result[0] == "Hello world."
    assert result[1] == "How are you?"
    assert result[2] == "I'm fine!"
    assert result[3] == "This has no punctuation But this one does."
    
    # Test with newlines
    text2 = "Line one.\nLine two! Line three?"
    result2 = split_sentences_custom(text2)
    print(f"Text2: {text2}")
    print(f"Result2: {result2}")
    assert len(result2) == 3
    
    print("Sentence Splitter Test PASSED!")

if __name__ == "__main__":
    test_splitter()
