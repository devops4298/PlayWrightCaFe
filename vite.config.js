# 0) INSTALLS (uncomment in a notebook/terminal)
# %pip install -qU langchain langchain-core langchain-community langgraph \
#                 langchain-openai beautifulsoup4

# 1) IMPORTS
import os
import getpass
import bs4
from typing_extensions import TypedDict, List

# LLM selection via LangChain's model-init helper
from langchain.chat_models import init_chat_model  # Doc: model init helper for chat models
# Vector store + docs
from langchain_core.vectorstores import InMemoryVectorStore  # Doc: in-memory vector store
from langchain_openai import OpenAIEmbeddings               # Doc: embeddings
from langchain_community.document_loaders import WebBaseLoader  # Doc: load webpage
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Doc: chunking
from langchain_core.documents import Document  # Doc: structured document type

# Prompt + graph
from langchain import hub  # Doc: pull pre-built prompts
from langgraph.graph import START, StateGraph                 # Doc: LangGraph state graphs

# 2) KEYS (set your own; keep secrets safe!)
# Gemini (LLM). Swap for OpenAI/Azure OpenAI if you prefer.
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter GOOGLE_API_KEY: ")

# OpenAI (embeddings)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OPENAI_API_KEY: ")

# 3) CHOOSE MODELS (LLM + embeddings)
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")  # conversational LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")             # high-quality embeddings

# 4) LOAD & CHUNK CONTENT (we’ll index Lilian Weng’s agents blog from the LangChain tutorials)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 5) BUILD VECTOR STORE
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(splits)

# 6) PROMPT (pre-built “RAG prompt” from LangChain Hub)
prompt = hub.pull("rlm/rag-prompt")

# 7) STATE SHAPE FOR OUR GRAPH
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# 8) RETRIEVAL NODE
def retrieve(state: State):
    retrieved = vector_store.similarity_search(state["question"])
    return {"context": retrieved}

# 9) GENERATION NODE
def generate(state: State):
    docs_text = "\n\n".join(d.page_content for d in state["context"])
    # Prompt expects {question, context}
    messages = prompt.invoke({"question": state["question"], "context": docs_text})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 10) COMPOSE GRAPH
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# 11) RUN A QUERY
result = graph.invoke({"question": "What is Task Decomposition?"})

# 12) PRETTY PRINT
print("ANSWER:\n", result["answer"])
print("\nSOURCES:")
for i, d in enumerate(result["context"], 1):
    print(f"{i}. {d.metadata.get('source', 'n/a')}")
