import streamlit as st
import requests

st.set_page_config(
    page_title="NLP Toolkit",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 NLP Toolkit")
st.markdown(
    """
    Welcome to the **Natural Language Processing Toolkit**.

    Perform various NLP tasks powered by Transformer models:
    - 😊 Sentiment Analysis
    - 📝 Text Summarization
    - 🏷 Named Entity Recognition
    - ❓ Question Answering
    - 📖 POS Tagging
    """
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sentiment",
    "Summarizer",
    "NER",
    "Question Answering",
    "POS Tagging",
])


with tab1:
    st.subheader("😊 Sentiment Analysis")
    st.caption(
        "Analyze the emotional tone of a piece of text and classify it as "
        "**Positive**, **Negative**, or **Neutral**."
    )

    with st.expander("ℹ️ About Sentiment Analysis"):
        st.write("""
        Sentiment Analysis is an NLP task used to determine the sentiment
        expressed in text. It is widely used for product reviews, social media
        monitoring, customer feedback, and opinion mining.
        """)
    sentiment_text = st.text_area("Enter text",key="sentiment_text")

    sentiment_btn = st.button("Analyze Sentiment",key="sentiment_btn")

with tab2:
    st.subheader("📝 Text Summarization")
    st.caption(
        "Generate a concise summary while preserving the key information."
    )

    with st.expander("ℹ️ About Text Summarization"):
        st.write("""
        Text Summarization condenses long documents into shorter versions
        containing the most important information. It is useful for articles,
        reports, research papers, and news.
        """)
    summary_text = st.text_area("Enter text",key="summaryt_text")

    summary_btn = st.button("Summarize",key="summary_btn")

with tab3:
    st.subheader("🏷 Named Entity Recognition")
    st.caption(
        "Identify entities such as people, organizations, locations, and dates."
    )

    with st.expander("ℹ️ About NER"):
        st.write("""
        Named Entity Recognition (NER) detects and classifies important entities
        in text, including Person, Organization, Location, Date, Time, and more.
        """)
    ner_text = st.text_area("Enter text",key="nert_text")

    ner_btn = st.button("Get NER",key="ner_btn")
with tab4:
    st.subheader("❓ Question Answering")
    st.caption(
        "Ask questions based on the provided context."
    )

    with st.expander("ℹ️ About Question Answering"):
        st.write("""
        The Question Answering model reads the given context and extracts the
        most relevant answer to your question without generating unrelated text.
        """)
    context = st.text_area("Enter Context")
    question = st.text_input("Enter Question")

    qa_btn = st.button("Answer",key="qa")

with tab5: 
    st.subheader("📖 Part-of-Speech Tagging")
    st.caption(
        "Label each word with its grammatical role."
    )

    with st.expander("ℹ️ About POS Tagging"):
        st.write("""
        Part-of-Speech (POS) Tagging assigns grammatical labels such as Noun,
        Verb, Adjective, Adverb, Pronoun, and Preposition to every word in a
        sentence.
        """)
    pos_text = st.text_area("Enter text")

    if st.button("POS Tagging"):
        response = requests.post("http://127.0.0.1:8000/POS",
                        json={"text": pos_text})
        result = response.json()

        st.subheader("Part of Speech")

        for item in result:
            st.write(
                f"{item['word']} → {item['pos']}"
            )

if sentiment_btn and sentiment_text:
    with st.spinner("Analyzing..."):
            try:
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