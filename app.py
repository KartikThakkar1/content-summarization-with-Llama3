import validators
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain



# Setting up the streamlit app
st.set_page_config(page_title="Content Summarizer",page_icon=":material/description:")
st.title("Summarize THAT with us 🖖🏼")
st.subheader("Summarize a youtube video or any website with us. Just provide the URL")


# on the sidebar, ask the user to provide their groq api key

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value="",type="password")

# get the url from the user
url = st.text_input("URL",label_visibility="collapsed")


# define the llm to be used. Using llama 3.3 70b from Groq
llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)


# defining a prompt template
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""

# creating the prompt from the prompt template using PromptTemplate function
prompt = PromptTemplate(template=prompt_template,input_variables=["text"])



# writing the logic for what happens when user presses the button
if st.button("SUMMARIZE THIS"):

    # 1. Validate if the user entered the inputs needed i.e. groq api key and the url to be summarized
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide all the inputs to get started.")

    # this elif block will check for the URLs validity
    elif not validators.url(url):
        st.error("Please provide a valid URL")


    # 2. Once inputs are validated, load the website data based on the URL type, create a chain, run the chain, process the output from chain
    else:
        try:
            with st.spinner("Summarizing it for you..."):

                # if it is a youtube link, use the youtube transcript from the yt transcript api else use the unstructuredurl loader from langchain
                if "youtube.com" in url:

                    video_id = url.split("v=")[-1]
                    transcript = YouTubeTranscriptApi.get_transcript(video_id=video_id,languages=["en","hi"])
                    text = " ".join([entry['text'] for entry in transcript])
                    docs = [Document(page_content=text)]
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False)
                    # headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    # 
                    ## load the documents back from the loader
                    docs = loader.load()

                # define the chain for summarization
                chain = load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)


        except Exception as e:
            st.exception(f"Exception:{e}")