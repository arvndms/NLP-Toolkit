import streamlit as st
from backend.models.sentiment import SentimentAnalyzer
from backend.models.summarizer import Summarizer
from backend.models.ner import NERExtractor
from backend.models.question_answering import QuestionAnswering
import requests

st.title("NLP App")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment",
    "Summarizer",
    "NER",
    "Question Answering",
    "POS Tagging",
])

with st.sidebar:
    st.header("NLP Toolkit")

    st.write("""
    Features:
    - Sentiment Analysis
    - Summarization
    - NER
    - Question Answering
    - POS Tagging
    """)

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
    context = st.text_area("Enter Context")
    question = st.text_input("Enter Question")

    qa_btn = st.button("Answer",key="qa")

with tab5: 
    from backend.utils.pos_tagger import POSTagger

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
    with st.spinner("Analyzing..."):
            try:
                    sentiment_model = SentimentAnalyzer() 
                    response = requests.post(
                        "http://127.0.0.1:8000/sentiment",
                        json={"text": sentiment_text}
                                        )
                    result = response.json()

                    # Overall sentiment
                    st.subheader("Overall Sentiment")

                    overall = result["overall"]

                    st.success(
                        f"{overall['label']} ({overall['score']:.3f})"
                    )

                    # Sentence-wise analysis
                    st.subheader("Sentence-wise Analysis")
                    with st.expander("Sentence-wise Analysis"):
                        for i, sentence_result in enumerate(result["sentence_wise"], 1):
                            
                            st.markdown("---")
                            st.markdown(f"### {sentence_result['Sentiment']}")
                            st.write(sentence_result['Sentence'])
                            st.caption(f"Confidence: {sentence_result['Score']:.3f}")
                                
            except RuntimeError :
                        st.write("Sentence Too Long!")
    st.success("Done!")

if summary_btn and summary_text:
    with st.spinner("Summarizing..."):
            summary_model = Summarizer()
            summary_text = "summarize the following : " + summary_text
            response = requests.post(
                        "http://127.0.0.1:8000/summary",
                        json={"text": summary_text}
                                        )
            result = response.json()
            st.text_area(
            "Summary",
            result[0]["summary_text"],
            height=200
            )
    st.success("Done!")

if ner_btn and ner_text:
    ner_model = NERExtractor()
    response = requests.post(
                        "http://127.0.0.1:8000/NER",
                        json={"text": ner_text}
                                        )
    result = response.json()
    for item in result:
      st.info(
        f"Entity: {item['Entity']}\n\n"
        f"Type: {item['Type']}\n\n"
        f"Confidence: {item['Confidence']:.3f}"
      )

if qa_btn and question and context:
    with st.spinner("Studying..."):
        qa_model = QuestionAnswering()
        response = requests.post(
                        "http://127.0.0.1:8000/QuestionAnswer",
                        json={"question": question,
                              "context":context}
                                        )
        result = response.json()
        st.subheader("Answer")

        st.success(result["answer"])

        st.write(
        f"Confidence: {result['score']:.3f}"
        )
    st.success("Done!")