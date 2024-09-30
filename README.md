# Sentiment-Analysis
This project is a Sentiment Analysis Web Application built using Python's Flask framework and a machine learning model that classifies text as positive, negative, or neutral. The application allows users to input text and receive an instant analysis of its sentiment. The underlying model can be either a simple pre-trained model like `TextBlob`, or a more advanced custom-trained model using libraries like `Scikit-learn` or `TensorFlow`.
## Features:
* User-friendly web interface: Enter text via a simple web form and get sentiment predictions.
* Real-time sentiment analysis: Classifies input text into Positive, Negative, or Neutral sentiment.
* Customizable model: Easily switch between different models (e.g., Naive Bayes, LSTM, etc.).
* Model training: Includes code to train a custom sentiment analysis model using a dataset like IMDb reviews or Twitter data.
## Technology Stack:
* Backend: Python (Flask)
* Frontend: HTML/CSS (Bootstrap)
* Model: Pre-trained model (`TextBlob`) or custom machine learning model (`Scikit-learn`, `TensorFlow`)
* Deployment: Deployed via Heroku (or any cloud platform).
## Requirements:
* Python 3.x
* Flask
* Scikit-learn or TensorFlow/Keras (for custom model)
* TextBlob or NLTK (for pre-trained model)
