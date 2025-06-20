# Content Summarizer with Llama3 | LangChain & GroqAPI 🗒️

Summarize content from any website or a YouTube video - with a click

## Features

> ### 🥇 **Multi-source content ingestion with smart loaders**
> Automatically detects whether the input URL is a YouTube video or a regular web page, then pulls data with `youtube-transcript-api` or LangChain’s `UnstructuredURLLoader` to build a clean Document object pipeline.

> ### 🥈 **Dynamic summarization strategy based on real-time token counts**
> Uses a custom `count_tokens` helper (Hugging Face tokenizer) to measure prompt size and switch between ***stuff, map-reduce, or refine chains*** on the fly, keeping every request under Groq’s 12 k TPM ceiling while squeezing maximum context into Llama-3.3-70B.

> ### 🥉 **Chunk‐aware splitting and selective context pruning**
> Implements `RecursiveCharacterTextSplitter` with adjustable chunk/overlap sizes, then iteratively trims excess segments to stay within a configurable token budget, showcasing granular control over memory, latency, and cost for large-scale summarization workflows.

## How does it work?
- The Python script leverages functionalities from LangChain and accesses Meta's llama-3.3-70b-versatile through Groq API
- Validates URLs and handles them based on their source (youtube, generic websites, etc)
- Utilizes chain summarization methods from LangChain to chain prompts and inputs
- Provides a Streamlit page for web based interaction

## How to run?

- Install the requirements in your python environment using `pip install -r requirements.txt`
- Run `app.py` with `streamlit run app.py`
- Provide a `Groq API Key` (for more : [Groq API](https://groq.com/))
- Provide a URL and click the summarize button

## Utility Examples

1. A thoughtful Medium article by [Debbie Levitt](https://deltacxdebbie.medium.com/) on [thinking critically about perspectives and information presented in a satirical fashion](https://rbefored.com/the-leaning-tower-of-pisa-is-perfectly-straight-0ba7b7ec0b1f).

   ![Image](https://github.com/user-attachments/assets/dd37aa14-73dd-4b86-8374-5b412e96b366)

2. One of the most popular API Documentation - [Stripe API Docs](https://docs.stripe.com/api)

   ![Image](https://github.com/user-attachments/assets/509185d9-652e-4aca-8bb4-b43a15da1bee)

3. An insightful YouTube video lecture : [Stanford CS229 I Machine Learning I Building Large Language Models (LLMs)](https://www.youtube.com/watch?v=9vM4p9NN0Ts) by [Yann Dubois(PhD Student at Stanford)](https://yanndubs.github.io/)

     ![Image](https://github.com/user-attachments/assets/6ad98195-8dd7-431c-a9a6-62cf0e82ce43)
