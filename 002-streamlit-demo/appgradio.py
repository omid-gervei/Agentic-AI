import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def detect_nsfw(comment):
    llm = ChatOpenAI(
        model="qwen/qwen3.6-plus",
        openai_api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )
    
    system_prompt = """You are a content moderation assistant. Your task is to analyze the given comment and determine if it contains NSFW (Not Safe For Work) content.
    
NSFW content includes but is not limited to:
- Explicit sexual content or nudity
- Hate speech or discriminatory language
- Excessive profanity or vulgar language
- Violent or graphic content
- Illegal activities
    
Respond with only one word: "NSFW" if the content is inappropriate, or "SAFE" if the content is appropriate for work environments."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{comment}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"comment": comment})
    return result.strip()

iface = gr.Interface(
    fn=detect_nsfw,
    inputs=gr.Textbox(label="Enter Comment", placeholder="Type your comment here..."),
    outputs=gr.Textbox(label="Classification"),
    title="NSFW Content Detector",
    description="Analyze comments to detect NSFW content using Qwen 3.6 Plus"
)

if __name__ == "__main__":
    iface.launch()