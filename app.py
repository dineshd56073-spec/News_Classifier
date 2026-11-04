import re, html, joblib, streamlit as st
from transformers import pipeline

st.set_page_config(page_title='News Article Classifier & Summariser', page_icon='📰')
st.title('News Article Classifier & Summariser')
st.write('Paste a wire story to suggest a news desk and generate a short summary.')

@st.cache_resource
def load_models():
    classifier = joblib.load('models/tfidf_logistic_regression.joblib')
    summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small')
    return classifier, summarizer

def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

classifier, summarizer = load_models()
article = st.text_area('News article', height=250)
if st.button('Classify and summarise'):
    if not article.strip():
        st.warning('Please paste an article.')
    else:
        cleaned = clean_text(article)
        probabilities = classifier.predict_proba([cleaned])[0]
        index = probabilities.argmax()
        st.subheader('Suggested desk: ' + classifier.classes_[index])
        st.metric('Confidence', f'{probabilities[index]*100:.2f}%')
        st.write('Desk probabilities')
        for desk, probability in sorted(zip(classifier.classes_, probabilities), key=lambda x:x[1], reverse=True):
            st.write(f'{desk}: {probability*100:.2f}%')
        with st.spinner('Generating summary...'):
            summary = summarizer('summarize: ' + cleaned, max_length=60, min_length=8,
                                 do_sample=False, truncation=True)[0]['summary_text']
        st.subheader('Generated summary')
        st.write(summary)
