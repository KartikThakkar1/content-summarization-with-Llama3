# Content Summarizer with Llama3 | LangChain & GroqAPI 🗒️

Summarize content from any website or a YouTube video - with a click

## How does it work?
- The Python script leverages functionalities from LangChain and accesses Meta's _llama-3.3-70b-versatile_ through Groq API
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
2. One of the most popular API Documentation - [Stripe API Docs](https://docs.stripe.com/api)
3. An informative YouTube video lecture : [Stanford CS229 I Machine Learning I Building Large Language Models (LLMs)](https://www.youtube.com/watch?v=9vM4p9NN0Ts) by [Yann Dubois(PhD Student at Stanford)](https://yanndubs.github.io/)
