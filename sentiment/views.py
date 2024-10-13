from django.shortcuts import render
from .forms import ReviewForm
import torch
from transformers import BertTokenizer
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')


from .model_utils import load_model, clean_text, predict_review

# Загрузка модели и токенизатора при запуске сервера
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment/bert_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = load_model(MODEL_PATH, device)

def index(request):
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['review']
            sentiment, rating = predict_review(review_text, tokenizer, model, device)

            # Передаем список с диапазоном (для генерации 10 звезд)
            star_range = range(1, 11)

            context = {
                'form': form,
                'sentiment': sentiment,
                'rating': rating,
                'review': review_text,
                'star_range': star_range  # Добавляем список с диапазоном для звезд
            }
            return render(request, 'sentiment/result.html', context)
    else:
        form = ReviewForm()
    return render(request, 'sentiment/index.html', {'form': form})