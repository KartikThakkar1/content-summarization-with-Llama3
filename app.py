import validators
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer

from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains.summarize import load_summarize_chain


# Helper function to count the tokens in the input, using transformers to access the tokenizer for Meta-Llama-3-70B-Instruct
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
def count_tokens(text):
    return len(tokenizer.encode(text))

# Setting up the streamlit app
st.set_page_config(page_title="Content Summarizer",page_icon=":material/description:")
st.title("Summarize THAT with us 🖖🏼")
st.subheader("Summarize a youtube video or any website with us. Just provide the URL")


# on the sidebar, asking the user to provide their groq api key

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



                # combining all the documents into a single string to check for number of tokens 
                full_text = " ".join([doc.page_content for doc in docs])
                num_tokens = count_tokens(full_text)
                
                # dynamically chosing the chain type based on number of tokens
                if num_tokens < 8000:
                    chain_type = "stuff"
                elif num_tokens < 15000:
                    chain_type = "map_reduce"
                else:
                    chain_type = "refine"

                
                ##st.info(f"Selected chain type: {chain_type} ({num_tokens} tokens)")

                # llama-3.3-70b-versatile has a TPM Limit of 12k tokens, even when using map_reduce and refine chain methods, number of tokens can still be
                # greater than 12k. 
                # so for map_reduce/refine, spliting docs first and tracking the number of tokens
                if chain_type in ["map_reduce", "refine"]:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)

                    all_chunks = splitter.split_documents(docs)

                    total_tokens = 0
                    max_tokens = 11300  # target under 12k limit
                    selected_chunks = []

                    for doc in all_chunks:
                        chunk_tokens = count_tokens(doc.page_content)
                        if total_tokens + chunk_tokens > max_tokens:
                            break
                        selected_chunks.append(doc)
                        total_tokens += chunk_tokens

                    docs = selected_chunks





                # loading the summarization chain based on the chain type (refine chain type requires 2 prompts.)
                if chain_type == "refine":
                    initial_prompt = PromptTemplate(
                                template= """
                                            Write an initial summary of the following content in 300 words:
                                            Content: {text}
                                        """,
                                  input_variables=["text"]
                                )
                    
                    refine_prompt = PromptTemplate(
                        template= """
                                    Your job is to refine the existing summary using new context below.
                                    Existing summary: {existing_answer}
                                    Additional context: {text}
                                    Refine the summary in 300 words.
                                    """,
                                    input_variables=["existing_answer", "text"]
                                )
                    

                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="refine",
                        question_prompt=initial_prompt,
                        refine_prompt=refine_prompt
                        )
                    

                else:
                    chain = load_summarize_chain(llm=llm, chain_type=chain_type, prompt=prompt)

                # running the chain
                output_summary = chain.run(docs)
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")


