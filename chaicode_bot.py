import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set user agent
os.environ["USER_AGENT"] = "YourUserAgentHere"

# URLs (unchanged)
urls = [
    "https://chaidocs.vercel.app/youtube/getting-started/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/emmit-crash-course/",
    "https://chaidocs.vercel.app/youtube/chai-aur-html/html-tags/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/terminology/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/behind-the-scenes/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/branches/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/diff-stash-tags/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/managing-history/",
    "https://chaidocs.vercel.app/youtube/chai-aur-git/github/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/hello-world/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/variables-and-constants/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/data-types/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/operators/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/control-flow/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/loops/",
    "https://chaidocs.vercel.app/youtube/chai-aur-c/functions/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/getting-started/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/jinja-templates/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/tailwind/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/models/",
    "https://chaidocs.vercel.app/youtube/chai-aur-django/relationships-and-forms/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/introduction/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/postgres/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/normalization/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/database-design-exercise/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-and-keys/",
    "https://chaidocs.vercel.app/youtube/chai-aur-sql/joins-exercise/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/welcome/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-vpc/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/setup-nginx/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://chaidocs.vercel.app/youtube/chai-aur-devops/node-nginx-vps/"
]

# Load and process documents
loader_multiple_pages = WebBaseLoader(urls)
docs = loader_multiple_pages.load()

text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
split_docs = text_spliter.split_documents(documents=docs)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=openai_api_key
)

# Initialize Qdrant retriever
retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="chaicode",
    embedding=embeddings
)

def handle_user_query(query):
    # Retrieve relevant chunks
    try:
        relevant_chunks = retriver.similarity_search(query=query)
    except Exception as e:
        return f"Error retrieving chunks from Qdrant: {str(e)}"

    # Build context (mimicking Streamlit example)
    context = ''.join([
        f"[Source: {chunk.metadata.get('source', 'No source available')}]\n"
        f"{chunk.page_content}\n\n" for chunk in relevant_chunks
    ])

    # Define the system prompt (moved outside loop)
    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who responds based on the available context.
    You have some context and analyze the user query to understand what the user is asking. 
    Check the available context below and give the best result.
    Now, respond to the user query with the best possible answer based on this context and include the source.

    Context:
    {context}
    """

    print("🧠 AI is thinking...")

    # Initialize ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    # Generate response
    try:
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ])
        answer = response.content.strip()
        print("\n🧠 Done thinking! Here is the answer:")
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Interactive loop
while True:
    user_query = input("Please enter your query (or type 'exit' to quit): ")
    
    if user_query.lower() == "exit":
        print("Exiting the program.")
        break
    
    answer = handle_user_query(user_query)
    print(f"Final Answer: {answer}")