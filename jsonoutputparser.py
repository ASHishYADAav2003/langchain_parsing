from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# NVIDIA OpenAI-compatible endpoint
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="""
Give me 5 facts about {topic}.

{format_instructions}
""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | llm | parser

result = chain.invoke({
    "topic": "male reproductive part"
})

print(result)