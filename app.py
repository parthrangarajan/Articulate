import streamlit as st
import transformers
# import streamlit.components.v1 as components
import tensorflow as tf
from transformers import pipeline
import pandas as pd


favicon= "articulate_favicon.ico"
st.set_page_config(
    page_title="Articulate by ParthRangarajan",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# loading the model
@st.cache(allow_output_mutation=True)#keeps up performance under hight performance
def load_model():
    model= pipeline("question-answering")
    return model

qa_model= load_model()

st.header("Welcome to Articulate!")
st.title("Have questions based on an article?")
# text area
article= st.text_area("Enter the article here!")
question_asked= st.text_input("Ask your questions here.")
button= st.button("Search Answer")

with st.spinner("Searching for your answers..."):
    if button and article:
        answers= qa_model(question= question_asked, context= article)
        st.success(answers['answer'])
        st.balloons()
# st.title("Examples")
# st.text("What is this article about?")
example_df=pd.DataFrame(["What is this article about ?", "What are the types of ___ ?", "What is the application of machine learning ?"], columns=["Examples"])
st.dataframe(example_df)

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;color: "red"' href="https://github.com/parthrangarajan" target="_blank">Parth Rangarajan</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)