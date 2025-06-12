# 📺 YouBot — YouTube Comment Analysis & Chatbot


YouBot is a Streamlit web application that empowers users to analyze YouTube video comments with advanced natural language processing (NLP) techniques and interact with a chatbot powered by Google's Gemini-1.5-Flash model. Gain insights into public sentiment, identify key discussion topics, and ask specific questions about the comments.

---
🌐 Live Demo
🖥️ Try the application here:
👉 https://data-mentor.streamlit.app/

## ✨ Features

- **YouTube Comment Fetching:**  
  Easily retrieve comments from any public YouTube video by simply pasting its URL.

- **Dual Sentiment Analysis:**  
  - *TextBlob:* Provides a quick and intuitive sentiment breakdown (positive, neutral, negative).  
  - *VADER Sentiment:* A lexicon and rule-based sentiment analysis tool, especially effective for social media text.  
  Visualized with clear bar charts and interactive pie charts.

- **Topic Modeling:**  
  - *LDA (Latent Dirichlet Allocation):* Uncover prominent themes and topics within the comments.  
  - *BERTopic:* Leverage BERT embeddings to create more coherent and interpretable topics.  
  Visualize dominant words for each model using dynamic Word Clouds.

- **Comment Trend Analysis:**  
  See how comment activity evolves over time with a historical comment count chart, including moving averages and peak indicators.

- **AI Chatbot (Clario):**  
  - Powered by Google's Gemini-1.5-Flash model.  
  - Ask specific questions about the fetched comments and receive intelligent, context-aware answers.  
  - Ideal for quickly extracting information or summarizing common sentiments.

---

## 🛠️ Technologies Used

- Streamlit — For creating interactive web applications.  
- Python — The core programming language.  
- Pandas — For data manipulation and analysis.  
- Requests — For making HTTP requests to the YouTube Data API.  
- TextBlob & VADER Sentiment — For sentiment analysis.  
- SpaCy & NLTK — For natural language processing (tokenization, stop word removal).  
- Gensim (LDA) — For Latent Dirichlet Allocation topic modeling.  
- BERTopic — For advanced topic modeling using BERT embeddings.  
- Matplotlib, Seaborn, Plotly — For data visualization.  
- LangChain — For building the AI chatbot.  
- Google Gemini API — Powering the intelligent chatbot responses.  
- Google YouTube Data API v3 — For fetching YouTube comments.
