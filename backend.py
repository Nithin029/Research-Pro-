from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_together import Together
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing import List, Dict, Annotated, TypedDict, Sequence, Optional
import operator
from duckduckgo_search import DDGS
import requests
from unstructured.partition.html import partition_html
import random
import regex
import json
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

contents = ""
preferences = []
# Load environment variables from .env file
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

model1 = Together(temperature=0, model="mistralai/Mixtral-8x22B-Instruct-v0.1", together_api_key=TOGETHER_API_KEY,
                  max_tokens=4096)
model2 = Together(temperature=0, model="mistralai/Mixtral-8x22B-Instruct-v0.1", together_api_key=TOGETHER_API_KEY,
                  max_tokens=8000)
model3 = Together(temperature=0, model="mistralai/Mixtral-8x22B-Instruct-v0.1", together_api_key=TOGETHER_API_KEY,
                  max_tokens=4000)

prompt_qa = """You're an helpful assistant.extract following entities from the query:{query} for each requested service./n
{{  "task" : "requested services",
    "niche" : " "
}}
If you find no specific task requested respond with None, Market Research is a valid task./n
You must respond ONLY with valid JSON file.
 Do not add any additional comments./n
"""
prompt = """
You are now expert Market Analyst who can extract structured information from the user provided query:{query}./n
prepare 6 important and unique tasks to do a market research for the provided business {query}.The tasks should provide a comprehensive framework for conducting market research on a {query} market solution offered by a startup before making an investment decision./n
return only the tasks without any description./n
You must RESPND ONLY with valid Json file./n
Do NOT ADD ANY ADDITIONAL COMMENTS and numbers.
ANSWER:
"""

prompt_subtasks = """ YYou are now in the role of an expert AI who can extract structured information from service and user provided query./n
{query},{task}
prepare atleast 10 sub topics to research each task individually {task}./n
Minimum 10 unique sub topics should be mentioned./n
prepare description/prompts for each sub topics./n
Both key and value pairs must be in double quotes./n
You must respond ONLY with valid JSON file./n
Do NOT ADD ANY ADDITIONAL COMMENTS and numbers.
"""

prompt1 = ChatPromptTemplate.from_template(prompt_qa)
prompt2 = ChatPromptTemplate.from_template(prompt)
prompt3 = ChatPromptTemplate.from_template(prompt_subtasks)



prompt_web = """
You are now in the role of an expert AI who can extract structured information from user request using the context./n
{context}
Question: {question}
Ignore everything you know, extract all the factual information  regarding the Question./n
Output should be plain text full of information and factual data, ignore links./n
Output should be in BULLET POINTS.\n
MAKE ANY IMPORTANT SIDE HEADINGS or COMPANY NAME ,FACTUAL DATA BOLD./n
Both key and value pairs must be in double quotes.\n
You must respond ONLY with valid JSON file.\n
Do not add any additional comments
"""


chain1 = {"query": RunnablePassthrough()} | prompt1 | model1 | JsonOutputParser()

chain_for_tasks = {"query": RunnablePassthrough()} | prompt2 | model1 | JsonOutputParser()

chain_default = {"task": chain_for_tasks, "query": RunnablePassthrough()} | prompt3 | model1 | JsonOutputParser()

chain_customize = (
        {"task": chain1, "query": RunnablePassthrough()}
        | prompt3
        | model1
        | JsonOutputParser()
)
def route(info):
    if "None" in info["task"]:
        return chain_default
    elif "Market Research" in info["task"]:
        return chain_default
    else:
        return chain_customize


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

ddgs = DDGS()

with open("http_proxies.txt", "r") as file:
    data = file.readlines()

http_proxies = []
for proxy in data:
    if proxy.split(":")[0] == "http":
        http_proxies.append(proxy[:-1])  # Till -1 to remove \n

header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

processed_urls = set()

class WebTool:
    def __init__(self) -> None:
        pass

    def get_urls(self, text):
        search_results = ddgs.text(text, max_results=2)
        links = [i["href"] for i in search_results]
        #print(links, "\n\n")
        return links

    def get_content(self, url):

        content = ""

        proxy = {"http": random.choice(http_proxies)}
        html_reponse = requests.get(
            url=url, headers=header, proxies=proxy
        ).content.decode()
        html_text_elements = partition_html(text=html_reponse)

        for i in html_text_elements:
            if "unstructured.documents.html.HTMLTitle" in str(type(i)):
                clean = regex.sub("\n", lambda x: "", i.text)
                content = content + "\n\n" + clean

            elif "unstructured.documents.html.HTMLNarrativeText" in str(type(i)):
                clean = regex.sub("\n", lambda x: "", i.text)
                content = content + "\n" + clean

            elif "unstructured.documents.html.HTMLListItem" in str(type(i)):
                clean = regex.sub("\n", lambda x: "", i.text)
                content = content + clean + "|"

        return content

    def fetch_content(self, text):
        content = ""
        urls = self.get_urls(text)
        for e, url in enumerate(urls):
            if url not in processed_urls:  # Check if the URL is processed already
                content += f"Source {e+1}\n\n"
                content += self.get_content(url)
                content += "\n\n\n"
                processed_urls.add(url)  # Add the URL to processed_urls set
            else:
                print(f"URL {url} is already processed, skipping...")

        return content.strip()

def run_web_tool(topic, sub_topic, niche):
    query_text = f"{niche}: {topic}: {sub_topic['description']}"
    try:
        content = remove_stopwords(WebTool().fetch_content(text=query_text))[:8193 * 4]
        global contents  # Use the global variable
        contents += content
        time.sleep(1)
    except UnicodeDecodeError:
        pass


def write_to_file(file_path, content_list):
    with open(file_path, "w") as file:
        for content in content_list:
            file.write(content + "\n")


def select_topic(data):
    print("Available topics:")
    for i, topic in enumerate(data.keys()):
        print(f"{i + 1}. {topic}")
    choice = input("Select topic(s) by index (comma-separated): ")
    choices = [int(choice.strip()) for choice in choice.split(',')]
    selected_topics = [list(data.keys())[choice - 1] for choice in choices]
    return selected_topics


def select_sub_topics(topic_data):
    sub_topics = []
    print(f"\n{topic_data['topic']}:")
    for i, sub_topic in enumerate(topic_data['sub_topics'], start=1):
        print(f"{i}. {sub_topic['sub_topic']}: {sub_topic['description']}")
        sub_topics.append(sub_topic)
    choices = input("\nSelect sub-topic(s) by index (comma-separated): ")
    choices = [int(choice.strip()) for choice in choices.split(',')]
    selected_sub_topics = [sub_topics[choice - 1] for choice in choices]
    return selected_sub_topics


def process_preferences(preferences,qa_chain):
    outputs = {}

    # Loop through each topic and its sub_topics
    for topic_data in preferences:
        topic = topic_data['topic']
        sub_topics = topic_data['sub_topics']

        # Iterate over each sub_topic
        for sub_topic in sub_topics:
            # Get the description of the sub_topic
            question = sub_topic['description']

            # Run the qa_chain function for the description
            output = qa_chain.invoke(question)
            outputs[sub_topic['sub_topic']] = output

    return outputs


def main():
    query = "Do a Market Research for connected car business"
    info = chain1.invoke(query)
    print(info)
    results = route(info).invoke(query)
    print(results)
    with open('results.json', 'w') as f:
        json.dump(results, f)
    file_path = 'results.json'
    data = load_data(file_path)
    niche = info["niche"]

    for topic, sub_topics in data.items():
        for sub_topic in sub_topics:
            run_web_tool(topic, sub_topic, niche)

    for url in processed_urls:
        print(url)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=0
    )
    prompt_web_1 = """
    You are an assistant tasked with summarizing  text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. text: {element}
    """
    prompt_web_1 = ChatPromptTemplate.from_template(prompt_web_1)
    summarize_chain = {"element": lambda x: x} | prompt_web_1 | model2 | StrOutputParser()
    texts = text_splitter.create_documents([contents])
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    file_path_text = "contents.txt"
    write_to_file(file_path_text, text_summaries)


    COHERE_API = COHERE_API_KEY
    cohere_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API)
    cohere_rerank = CohereRerank(cohere_api_key=COHERE_API, top_n=4)
    raw_documents = TextLoader('contents.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(raw_documents)
    embeddings = cohere_embeddings
    new_client = chromadb.EphemeralClient()
    vectorstore = Chroma.from_documents(
        split_documents, embeddings, client=new_client, collection_name="content_research_"
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=cohere_rerank,
        base_retriever=vectorstore.as_retriever()
    )

    prompt = ChatPromptTemplate.from_template(prompt_web)
    qa_chain = {"context": compression_retriever,
                "question": RunnablePassthrough()} | prompt | model2 | JsonOutputParser()

    file_path = 'results.json'
    data = load_data(file_path)
    selected_topics = select_topic(data)
    for topic in selected_topics:
        selected_topic_data = {"topic": topic, "sub_topics": data[topic]}
        selected_sub_topics = select_sub_topics(selected_topic_data)
        preferences.append({"topic": topic, "sub_topics": selected_sub_topics})

    with open('preferences.json', 'w') as file:
        json.dump(preferences, file, indent=4)

    with open('preferences.json', 'r') as file:
        preferences_output = json.load(file)

    outputs = process_preferences(preferences_output,qa_chain=qa_chain)
    json_output = json.dumps(outputs, indent=4)
    print(json_output)


if __name__ == "__main__":
    main()
