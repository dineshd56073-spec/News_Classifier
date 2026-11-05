
import os
import re
import joblib
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_ERROR = None

except Exception as e:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_ERROR = str(e)

st.set_page_config(
    page_title="BBC News Classifier & Summariser",
    page_icon="📰"
)

st.title("BBC News Classifier & Summariser")

st.write(
    "Paste a BBC news article to predict its category "
    "and generate a short summary."
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "news_classifier.pkl"
)


@st.cache_resource
def load_preprocessor():

    try:
        stop_words = set(stopwords.words("english"))

    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    # Preserve negative words
    stop_words -= {"not", "no", "nor"}

    stemmer = PorterStemmer()

    return stop_words, stemmer


@st.cache_resource
def load_models():

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    saved_model = joblib.load(MODEL_PATH)

    if (
        not isinstance(saved_model, dict)
        or "tfidf" not in saved_model
        or "lr_model" not in saved_model
    ):
        raise ValueError(
            "news_classifier.pkl must contain "
            "'tfidf' and 'lr_model' keys."
        )

    tfidf = saved_model["tfidf"]
    lr_model = saved_model["lr_model"]

    return tfidf, lr_model


@st.cache_resource
def load_summarizer():

    if not TRANSFORMERS_AVAILABLE:
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        return tokenizer, model

    except Exception as e:
        print("Summarizer loading error:", e)
        return None, None

def preprocess_text(text):

    stop_words, stemmer = load_preprocessor()

    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


try:

    tfidf, lr_model = load_models()

    st.success("Logistic Regression classifier loaded successfully.")

except Exception as exc:

    st.error(f"Could not load classifier: {exc}")
    st.stop()


article = st.text_area(
    "News article",
    height=250,
    placeholder="Paste the full news article here..."
)


if st.button("Classify and summarise"):

    if not article.strip():

        st.warning("Please paste an article first.")

    else:

        st.subheader("News Classification")

        with st.spinner("Predicting news category..."):

            processed_article = preprocess_text(article)

            article_tfidf = tfidf.transform(
                [processed_article]
            )

            probabilities = lr_model.predict_proba(
                article_tfidf
            )[0]

            best_index = probabilities.argmax()

            predicted_category = lr_model.classes_[best_index]

            confidence = probabilities[best_index] * 100

        st.success(
            f"Predicted category: {predicted_category}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

       
        st.subheader("Category Probabilities")

        sorted_probabilities = sorted(
            zip(lr_model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True
        )

        for category, probability in sorted_probabilities:

            st.write(
                f"{category}: {probability * 100:.2f}%"
            )

            st.progress(float(probability))


        st.subheader("Generated Summary")

        if not TRANSFORMERS_AVAILABLE:

            st.warning(
                "Transformers could not be imported. "
                "Classification works, but summarization is unavailable."
            )

            st.caption(
                f"Import error: {TRANSFORMERS_ERROR}"
            )

        else:

            tokenizer, summarizer_model = load_summarizer()

            if tokenizer is None or summarizer_model is None:

                st.warning(
                    "T5 summarization model could not be loaded."
                )

            else:

                try:

                    with st.spinner("Generating summary..."):

                        inputs = tokenizer(
                            "summarize: " + article.strip(),
                            return_tensors="pt",
                            max_length=512,
                            truncation=True
                        )

                        outputs = summarizer_model.generate(
                            **inputs,
                            max_length=60,
                            min_length=8,
                            do_sample=False
                        )

                        summary = tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )

                    st.write(summary)

                except Exception as e:

                    st.error(
                        f"Summary generation failed: {e}"
                    )
