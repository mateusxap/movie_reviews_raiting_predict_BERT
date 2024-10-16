# Movie Reviews Rating Prediction with BERT

Этот проект демонстрирует, как использовать модель BERT для предсказания рейтинга отзывов о фильмах. В основе проекта лежит веб-приложение, созданное на фреймворке Django, которое использует BERT для анализа тональности отзывов и прогнозирования их рейтинга.

## Установка

**Шаг 1: Клонирование репозитория и получение больших файлов**

```bash
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/mateusxap/movie_reviews_raiting_predict_BERT.git
cd movie_reviews_raiting_predict_BERT
git lfs fetch
git lfs pull
cd ..
```

**Шаг 2: Создание виртуального окружения**

```bash
sudo apt install -y python3-venv
python3 -m venv django_env
source django_env/bin/activate
```
**Шаг 3: Установка зависимостей**
```bash
pip install torch transformers beautifulsoup4 textblob nltk django
```
**Шаг 4: Запуск сервера разработки**
```bash
cd movie_reviews_raiting_predict_BERT
python3 manage.py runserver
```
Теперь вы можете открыть браузер и перейти по адресу http://127.0.0.1:8000/, чтобы увидеть работающее приложение.
