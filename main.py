from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os

load_dotenv("./sample.env")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# print env vars
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))    

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})
# raw_response = {'query': 'What is a embedding model which is used in Ai agents ? write output in a file. ', 'output': '{"topic":"Embedding Models in AI Agents","summary":"An embedding model in the context of AI agents is a type of model that transforms input data into a vector space. This transformation allows the AI agent to represent various types of data, such as words, sentences, images, or other features, in a way that captures the semantic meaning or relationships between them. Common examples of embedding models in natural language processing include Word2Vec, GloVe (Global Vectors for Word Representation), and Sentence Transformers. These models are particularly useful for tasks like text classification, clustering, and information retrieval, as they convert complex data into numerical representations that machine learning algorithms can analyze efficiently.","sources":["Word2Vec Overview - Google Research","GloVe: Global Vectors for Word Representation","Sentence Transformers - NLP Models"],"tools_used":["save_text_to_file"]}'}
try:
    output_text = raw_response.get("output")
    if isinstance(output_text, str):
        structured_response = parser.parse(output_text)
    else:
        raise ValueError("Unexpected output format")
    print(raw_response)
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)