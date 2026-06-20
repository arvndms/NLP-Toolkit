import streamlit as st
from models.sentiment import SentimentAnalyzer
from models.summarizer import Summarizer
from models.ner import NERExtractor
from models.question_answering import QuestionAnswering

st.title("NLP App")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Sentiment",
    "Summarizer",
    "NER",
    "Question Answering",
    "POS Tagging",
    "Spell Checker",
])

with tab1:
    sentiment_text = st.text_area("Enter text",key="sentiment_text")

    sentiment_btn = st.button("Analyze Sentiment",key="sentiment_btn")

with tab2:
    summary_text = st.text_area("Enter text",key="summaryt_text")

    summary_btn = st.button("Summarize",key="summary_btn")

with tab3:
    ner_text = st.text_area("Enter text",key="nert_text")

    ner_btn = st.button("Get NER",key="ner_btn")
with tab4:
    question = st.text_input("Enter Question")
    context = st.text_area("Enter Context")

    qa_btn = st.button("Answer",key="qa")

with tab5: 
    from utils.pos_tagger import POSTagger

with tab5:

    pos_text = st.text_area("Enter text")

    if st.button("POS Tagging"):

        pos_model = POSTagger()

        result = pos_model.predict(pos_text)

        st.subheader("Part of Speech")

        for item in result:
            st.write(
                f"{item['word']} → {item['pos']}"
            )

if sentiment_btn and sentiment_text:
    sentiment_model = SentimentAnalyzer()
    result = sentiment_model.predict(sentiment_text)

    # Overall sentiment
    st.subheader("Overall Sentiment")

    overall = result["overall"]

    st.success(
        f"{overall['label']} ({overall['score']:.3f})"
    )

    # Sentence-wise analysis
    st.subheader("Sentence-wise Analysis")

    for i, sentence_result in enumerate(result["sentence_wise"], 1):

        sentiment = sentence_result["Sentiment"]

        if sentiment == "POSITIVE":
            st.success(
                f"{i}. {sentence_result['Sentence']}\n\n"
                f"Sentiment: {sentiment} ({sentence_result['Score']:.3f})"
            )

        elif sentiment == "NEGATIVE":
            st.error(
                f"{i}. {sentence_result['Sentence']}\n\n"
                f"Sentiment: {sentiment} ({sentence_result['Score']:.3f})"
            )

        else:
            st.info(
                f"{i}. {sentence_result['Sentence']}\n\n"
                f"Sentiment: {sentiment} ({sentence_result['Score']:.3f})"
            )

if summary_btn and summary_text:
    summary_model = Summarizer()
    result = summary_model.summarize(summary_text)
    st.write(result)

if ner_btn and ner_text:
    ner_model = NERExtractor()
    result = ner_model.predict(ner_text)
    st.write(result)

if qa_btn and question and context:
    qa_model = QuestionAnswering()
    result = qa_model.predict(question,context)
    st.success(result['answer'])
    st.write(f"Confidence: {result['score']:.3f}")
