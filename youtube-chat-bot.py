import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from urllib.parse import urlparse, parse_qs
import spacy
import nltk
from gensim import corpora
from gensim.models import LdaModel
from wordcloud import WordCloud
from bertopic import BERTopic
import plotly.graph_objects as go
from scipy.signal import find_peaks
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Streamlit Interface Start ---
st.title("📺 YouBot — YouTube Comment Analysis & Chatbot")

API_KEY = st.secrets["API_KEY"] # YOUTUBE API KEY 
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Function to extract Video ID
def extract_video_id(url):
    query = urlparse(url)
    # Check for extended URL formats
    if query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        elif query.path.startswith('/embed/'):
            return query.path.split('/embed/')[1].split('?')[0]
    elif query.hostname == 'youtu.be':
        return query.path[1:]
    return None

# Function to fetch comments
def get_youtube_comments(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Please enter a valid YouTube video URL.")

    comments_data = []
    next_page_token = None
    while True:
        url = f'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={API_KEY}&maxResults=100'  # maxResults eklendi
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        response = requests.get(url)

        if response.status_code == 403:
            st.error("API quota exceeded or the API key is invalid. Please try again later or use a valid API key.")
            return pd.DataFrame(), None  # Return an empty DataFrame in case of an error
        elif response.status_code != 200:
            st.error(f"API request failed: {response.status_code} - {response.text}")
            return pd.DataFrame(), None  # Return an empty DataFrame in case of an error

        data = response.json()

        if 'items' not in data or not data['items']:
            st.warning("Comments on this video may be turned off or no comments were found.")
            break

        for item in data['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            published_at = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments_data.append({'Comment': comment, 'Published At': published_at})

        next_page_token = data.get('nextPageToken', None)
        if not next_page_token:
            break

    return pd.DataFrame(comments_data), video_id

# Create a chatbot with LangChain
@st.cache_resource(show_spinner=False)
def setup_langchain(df):
    st.info("Preparing comments for the chatbot... This may take a while.")
    all_comments = "\n".join(df['Comment'].dropna().astype(str).tolist())
    documents = [Document(page_content=all_comments)]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Add a custom prompt
    template = """You are a helpful AI assistant tasked with answering questions about YouTube comments.
    Use the following comments to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {question}
    """
    qa_chain_prompt = PromptTemplate.from_template(template)

    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_chain_prompt)

    qa_chain = RetrievalQA(combine_documents_chain=doc_chain, retriever=retriever)

    return qa_chain

# YouTube - URL input field
video_url = st.text_input("🎥 Paste the YouTube video link here:")

if video_url:
    try:
        if "comments_df" not in st.session_state or st.session_state.get('video_url_for_analysis') != video_url:
            st.session_state.video_url_for_analysis = video_url
            comments_df, video_id = get_youtube_comments(video_url)

            if comments_df.empty:
                st.warning("Could not fetch comments for this video or no comments were found. Please try another link.")
                st.session_state.comments_df = pd.DataFrame()
                if 'qa_chain' in st.session_state:
                    del st.session_state.qa_chain  # Reset the chatbot
                st.stop()  # Stop the rest of the app

            st.session_state.comments_df = comments_df
            st.session_state.chat_history = []  # Reset chat history for a new video
            st.session_state.qa_chain = setup_langchain(comments_df)  # Rebuild the chatbot
        else:
            comments_df = st.session_state.comments_df  # Get comments_df from session state

        # If the comment DataFrame is empty or contains only null values, skip the next steps
        if comments_df.empty or comments_df['Comment'].dropna().empty:
            st.warning("Not enough comments found for sentiment analysis and chatbot. Please try another video link with comments.")
            st.stop()  # Stop the rest of the app

        # --- Chart Section Start (In Sidebar) ---
        st.sidebar.header("📈 Comment Analysis Charts")
        df = comments_df.copy()

        # TextBlob sentiment analysis
        def get_sentiment(text):
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            if polarity > 0:
                return 'positive'
            elif polarity < 0:
                return 'negative'
            else:
                return 'neutral'

        df['TextBlob_Sentiment'] = df['Comment'].apply(get_sentiment)

        # Chart style and color palette
        sns.set_style("whitegrid")
        custom_palette = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}

        # TextBlob chart drawing
        st.sidebar.subheader("📊 TextBlob Sentiment Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        sentiment_plot = sns.countplot(
            data=df,
            x='TextBlob_Sentiment',
            order=['positive', 'neutral', 'negative'],
            palette=custom_palette,
            ax=ax
        )
        ax.set_title("TextBlob Sentiment Distribution", fontsize=16, fontweight='bold')
        ax.set_xlabel("Sentiment Type", fontsize=12)
        ax.set_ylabel("Number of Comments", fontsize=12)
        for p in sentiment_plot.patches:
            height = p.get_height()
            sentiment_plot.text(p.get_x() + p.get_width() / 2., height, f'{int(height)}', ha="center", fontsize=11, fontweight='bold')
        st.sidebar.pyplot(fig)
        plt.close(fig)

        # VADER sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        def vader_sentiment(text):
            score = analyzer.polarity_scores(str(text))['compound']
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'

        df['VADER_Sentiment'] = df['Comment'].apply(vader_sentiment)

        # VADER chart drawing
        st.sidebar.subheader("📊 VADER Sentiment Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        vader_palette = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}
        vader_plot = sns.countplot(
            data=df,
            x='VADER_Sentiment',
            order=['positive', 'neutral', 'negative'],
            palette=vader_palette,
            ax=ax2
        )
        ax2.set_title("VADER Sentiment Distribution", fontsize=16, fontweight='bold')
        ax2.set_xlabel("Sentiment Type", fontsize=12)
        ax2.set_ylabel("Number of Comments", fontsize=12)
        for p in vader_plot.patches:
            height = p.get_height()
            vader_plot.text(p.get_x() + p.get_width() / 2., height, f'{int(height)}', ha="center", fontsize=11, fontweight='bold')
        st.sidebar.pyplot(fig2)
        plt.close(fig2)

        # --- Sentiment Analysis Pie Charts (Side by Side) ---
        st.sidebar.subheader("📊 Sentiment Distribution Percentages (Pie Charts)")

        # Create two columns for pie charts
        col1, col2 = st.sidebar.columns(2)

        with col1:
            # TextBlob Pie Chart
            st.markdown("### TextBlob")
            sentiment_counts_textblob = df['TextBlob_Sentiment'].value_counts()

            # Check to avoid errors if the DataFrame is empty or contains only one category
            if not sentiment_counts_textblob.empty:
                fig_pie_textblob, ax_pie_textblob = plt.subplots(figsize=(6, 6))
                colors_textblob = [custom_palette[label] for label in sentiment_counts_textblob.index]
                ax_pie_textblob.pie(
                    sentiment_counts_textblob,
                    labels=sentiment_counts_textblob.index,
                    autopct='%1.1f%%',
                    colors=colors_textblob,
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'},
                    textprops={'fontsize': 10}
                )
                ax_pie_textblob.set_title("TextBlob Sentiment Percentages", fontsize=14, fontweight='bold')
                ax_pie_textblob.axis('equal')
                st.pyplot(fig_pie_textblob)
                plt.close(fig_pie_textblob)
            else:
                st.info("Insufficient sentiment data found for TextBlob.")

        with col2:
            # VADER Pie Chart
            st.markdown("### VADER")
            sentiment_counts_vader = df['VADER_Sentiment'].value_counts()

            # Check to avoid errors if the DataFrame is empty or contains only one category
            if not sentiment_counts_vader.empty:
                fig_pie_vader, ax_pie_vader = plt.subplots(figsize=(6, 6))
                colors_vader = [vader_palette[label] for label in sentiment_counts_vader.index]
                ax_pie_vader.pie(
                    sentiment_counts_vader,
                    labels=sentiment_counts_vader.index,
                    autopct='%1.1f%%',
                    colors=colors_vader,
                    startangle=90,
                    wedgeprops={'edgecolor': 'black'},
                    textprops={'fontsize': 10}
                )
                ax_pie_vader.set_title("TextBlob Sentiment Percentages", fontsize=14, fontweight='bold')
                ax_pie_vader.axis('equal')
                st.pyplot(fig_pie_vader)
                plt.close(fig_pie_vader)
            else:
                st.info("Insufficient sentiment data found for VADER")

        # Preprocessing for LDA and BERTopic analyses
        st.sidebar.subheader("📝 Comment Topic Analysis (LDA & BERTopic)")
        nltk.download('punkt', quiet=True)
        nlp = spacy.load("en_core_web_sm")

        def preprocess(text):
            if isinstance(text, str):
                doc = nlp(text.lower())
                # Keep only alphabetic characters and non-stop words
                return [token.text for token in doc if not token.is_stop and token.is_alpha]
            else:
                return []

        # Process comments for topic modeling and filter out empty ones
        df_for_topic_analysis = df.dropna(subset=['Comment']).copy()
        df_for_topic_analysis['Processed_Comments'] = df_for_topic_analysis['Comment'].apply(preprocess)

        # Keep only comments with processed text
        df_valid_for_topics = df_for_topic_analysis[df_for_topic_analysis['Processed_Comments'].apply(lambda x: len(x) > 0)]

        if df_valid_for_topics.empty:
            st.sidebar.info("Processed comments are empty. Topic modeling (LDA and BERTopic) could not be performed.")
        else:
            # LDA
            dictionary = corpora.Dictionary(df_valid_for_topics['Processed_Comments'])
            corpus = [dictionary.doc2bow(comment) for comment in df_valid_for_topics['Processed_Comments']]

            if corpus:
                lda_model = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=15, random_state=42)
                lda_topic_assignments = [max(lda_model[doc_bow], key=lambda x: x[1])[0] for doc_bow in corpus]  # Use doc_bow from corpus
                df_valid_for_topics['LDA_Topic'] = lda_topic_assignments  # Assign topics to the correct DataFrame

                st.sidebar.subheader("☁️ LDA Results - WordCloud")
                all_words_lda = ' '.join([word for comment in df_valid_for_topics['Processed_Comments'] for word in comment])
                if all_words_lda:
                    wordcloud_lda = WordCloud(width=800, height=400, background_color='white').generate(all_words_lda)
                    fig_lda = plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_lda, interpolation='bilinear')
                    plt.axis('off')
                    st.sidebar.pyplot(fig_lda)
                    plt.close(fig_lda)
                else:
                    st.sidebar.info("Not enough words found for LDA.")
            else:
                st.sidebar.info("Insufficient data to build LDA model.")

            # BERTopic
            bertopic_docs = df_valid_for_topics['Processed_Comments'].apply(lambda x: ' '.join(x)).tolist()
            if any(bertopic_docs):
                topic_model = BERTopic()
                topics, _ = topic_model.fit_transform(bertopic_docs)
                df_valid_for_topics['BerTopic_Topic'] = topics  # Assign topics to the correct DataFrame

                st.sidebar.subheader("☁️ BERTopic Results - WordCloud")
                all_words_bertopic = ' '.join([word for comment in df_valid_for_topics['Processed_Comments'] for word in comment])
                if all_words_bertopic:
                    wordcloud_bertopic = WordCloud(width=800, height=400, background_color='white').generate(all_words_bertopic)
                    fig_bertopic = plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud_bertopic, interpolation='bilinear')
                    plt.axis('off')
                    st.sidebar.pyplot(fig_bertopic)
                    plt.close(fig_bertopic)
                else:
                    st.sidebar.info("Not enough words found for BERTopic.")
            else:
                st.sidebar.info("Insufficient data to build BERTopic model.")

        # Historical Comment Count Chart
        st.sidebar.subheader("📈 Comment Count Over Time")
        if 'Published At' in df.columns and not df['Published At'].empty:
            df['Published At'] = pd.to_datetime(df['Published At'], errors='coerce')
            df_filtered = df.dropna(subset=['Published At'])  # Drop invalid dates

            if not df_filtered.empty:
                first_comment_date = df_filtered['Published At'].min()
                last_comment_date = df_filtered['Published At'].max()

                # Check the date range, change frequency if it's the same day
                if first_comment_date.date() == last_comment_date.date():
                    st.sidebar.info("Since comments are from a single day, a time series chart would be meaningless.")
                else:
                    date_range = pd.date_range(start=first_comment_date.date(), end=last_comment_date.date(), freq='D')
                    daily_comments = df_filtered.groupby(df_filtered['Published At'].dt.date).size()
                    daily_comments = daily_comments.reindex(date_range.date, fill_value=0)
                    z = daily_comments.rolling(window=3, min_periods=1).mean()

                    peaks, _ = find_peaks(daily_comments.values, distance=2)

                    fig_time_series = go.Figure()
                    fig_time_series.add_trace(go.Scatter(x=daily_comments.index, y=daily_comments.values,
                                                         mode='lines+markers', name='Number of Comments',
                                                         line=dict(color='blue'), marker=dict(symbol='circle', size=6)))
                    fig_time_series.add_trace(go.Scatter(x=daily_comments.index, y=z,
                                                         mode='lines', name='Trend (3-Day Moving Average)',
                                                         line=dict(color='red', dash='dash')))
                    fig_time_series.add_trace(go.Scatter(x=daily_comments.index[peaks], y=daily_comments.values[peaks],
                                                         mode='markers', name='Peak Points',
                                                         marker=dict(color='green', size=10, symbol='star', line=dict(width=2))))
                    fig_time_series.update_layout(title="Comment Count Over Time and Trend",
                                                  xaxis_title="Date",
                                                  yaxis_title="Number of Comments",
                                                  template="plotly_white",
                                                  xaxis_tickformat="%Y-%m-%d", hovermode="closest")
                    st.sidebar.plotly_chart(fig_time_series)
            else:
                st.sidebar.info("No valid date data found for the time series chart.")
        else:
            st.sidebar.info("The 'Published At' column was not found or is empty for the time series chart.")

        # --- Chatbot Section Start ---
        st.success("Chatbot is ready!")
        st.markdown("---")
        st.markdown("## 🤖 YouBot & Clario")
        st.markdown("You can ask questions about the content of the comments to get detailed information.")
        st.markdown("⚠️**Please click the 'Send' button after typing your message. Do not press Enter. To send a new message, delete the previous one and type again.**")
        user_query = st.text_input("Type your question here:", key="chatbot_input")

        if st.button("Send", key="send_button") and user_query:
            if "qa_chain" in st.session_state and st.session_state.qa_chain:
                with st.spinner("Searching for an answer..."):
                    try:
                        response = st.session_state.qa_chain.run(user_query)
                        st.session_state.chat_history.append(("You", user_query))
                        st.session_state.chat_history.append(("Clario", response))
                    except Exception as llm_e:
                        st.error(f"An error occurred while retrieving the chatbot response: {llm_e}")
            else:
                st.warning("Chatbot is not ready yet or comments could not be fetched. Please enter a valid YouTube link.")

        # Show chat history
        chat_history_container = st.container(height=300)
        with chat_history_container:
            for sender, message in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"**👤 {sender}:** {message}")
                else:
                    st.markdown(f"**🤖 {sender}:** {message}")

    except Exception as e:
        st.error(f"A general error occurred: {e}")
        st.info("Please check the YouTube video link or ensure your API keys are valid and try again.")
