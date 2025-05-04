from datetime import datetime
import os
import asyncio
from google.colab import userdata
from browser_use import Agent
import nest_asyncio
nest_asyncio.apply()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama

# Set dummy OPENAI_API_KEY to bypass LangChain internal checks
import os
os.environ["OPENAI_API_KEY"] = "key"



# 5 Hindi domain-specific tasks
tasks = [
    ("health", "babychakra.com के अनुसार, नवजात शिशुओं के लिए सर्दी-खांसी से बचाव के घरेलू उपाय क्या हैं?"),
    ("finance", "mymoneykarma.com के अनुसार, भारत में SIP और FD में क्या अंतर है?"),
    ("transport", "irctc.co.in या nget.irctc.in के अनुसार, तत्काल टिकट बुकिंग की प्रक्रिया क्या है?"),
    ("science", "vikaspedia.in के अनुसार, पानी की सतह पर मच्छरों के अंडे कैसे फैलते हैं?"),
    ("policy", "pmkisan.gov.in के अनुसार, पीएम किसान योजना के लिए पात्रता क्या है?")
]

# List of models to evaluate
all_models = [
    "gemini",
    "groq",
    "openrouter",
    "cohere",
    #"vllm",
    #"ollama"
]

planner_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

async def run_model(model_type, domain, task):
    print(f"\n🔍 Running model: {model_type} on task: {domain}")

    use_vision = True
    if model_type == "gemini":
        #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key="key"
            )

    elif model_type == "gpt":
        llm = ChatOpenAI(model="gpt-4o", api_key = "key" )
    elif model_type == "groq":
        use_vision = False
        llm = ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key="key"
        )
    elif model_type == "openrouter":
        llm = ChatOpenAI(
            model="qwen/qwen2.5-vl-72b-instruct:free",
            base_url="https://openrouter.ai/api/v1",
            api_key="key"
        )
    elif model_type == "cohere":
        use_vision = False
        llm = ChatOpenAI(
            model="command-a-03-2025",
            base_url="https://api.cohere.ai/compatibility/v1",
            api_key="key"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"./logs/{model_type}_{domain}_{timestamp}/conversation"

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=use_vision,
        save_conversation_path=save_path,
        planner_llm=planner_llm,
        planner_interval=1,
        max_failures=3,
    )
    await agent.run()

async def main():
    for model_type in all_models:
        for domain, task in tasks:
            try:
                await run_model(model_type, domain, task)
            except Exception as e:
                print(f"Failed: {model_type} on {domain} → {e}")

await main() 
