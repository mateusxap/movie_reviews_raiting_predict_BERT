import torch
from transformers import BertConfig
from .models import BertForSentimentAndRating
import re
import string
from textblob import TextBlob
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def load_model(model_path, device):
    config = BertConfig.from_pretrained(model_path)
    model = BertForSentimentAndRating.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()  # Переключаем модель в режим оценки
    return model

# 1. Lowercasing
def lowercase_text(text):
    return text.lower()

# 2. Remove HTML Tags
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

# 3. Remove URLs
def remove_urls(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

# 4. Remove Punctuations
def remove_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# 5. Handling ChatWords
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "B4": "Before",
    "LOL": "Laughing Out Loud",
    "BRB": "Be Right Back",
    "FYI": "For Your Information",
    "IMO": "In My Opinion",
    "IMHO": "In My Humble Opinion",
    "LMAO": "Laughing My Ass Off",
    "GR8": "Great",
    "IRL": "In Real Life",
    "ILY": "I Love You",
    "BTW": "By The Way",
    "THX": "Thanks",
    "PLS": "Please",
}

def chatword_conversion(text):
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

# 6. Spelling Correction
def correct_spelling(text):
    return str(TextBlob(text).correct())

# 8. Remove Emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"      # symbols & pictographs
        u"\U0001F680-\U0001F6FF"      # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"      # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# 9. Tokenization
def tokenize_text(text):
    return word_tokenize(text)

# 10. Stemming
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

# 11. Lemmatization
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    return " ".join(tokens)

# Полная функция очистки текста
def clean_text(text):
    text = lowercase_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = chatword_conversion(text)
    text = remove_punctuations(text)
    # долгие вычисления
    # text = correct_spelling(text)
    text = stem_words(text)
    text = lemmatize_words(text)
    return text

def predict_review(text, tokenizer, model, device, max_len=256):
    """
    Функция для предсказания сентимента и рейтинга отзыва.
    """
    # Очистка текста
    cleaned_text = clean_text(text)

    # Токенизация
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits, output = model(input_ids, attention_mask)
        sentiment = torch.argmax(logits, dim=1).item()
        rating = output.squeeze().item()

    print(sentiment)
    # Интерпретация сентимента
    sentiment_label = 'Positive' if sentiment == 1 else 'Negative'

    # Округление рейтинга и ограничение его диапазона от 1 до 10
    rating = round(rating)
    rating = max(1, min(10, rating))

    return sentiment_label, rating