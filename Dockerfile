FROM python:3.12.10

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the vectorizer file into the container
COPY app.py .
COPY ./model_repository/movie_review_sentiment_model/1/tfidf_vectorizer.pkl .

EXPOSE 9898

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9898"]