import wikipediaapi, os
from langchain.tools import BaseTool, StructuredTool, tool
from serpapi import GoogleSearch
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
SERP_API_KEY= os.getenv("SERP_API_KEY")

@tool
def search_wikipedia(query: str):
    """Search wikipedia for factual information."""
    wiki_api=wikipediaapi.Wikipedia("Info Agent","en")
    page = wiki_api.page(query)
    if page.exists():
        return  page.summary[:500]  
    return 'No relevant information found.'

@tool
def search_google(query: str):
    """Google Search: Search google for latest updates and news."""
    search = GoogleSearch({"q": query, "api_key": SERP_API_KEY})
    results = search.get_dict()
    if 'organic_results' in results:
        top_result = results['organic_results'][0]
        title = top_result.get("title")
        link = top_result.get("link")
        snippet = top_result.get("snippet")
        return  snippet
    return 'No relevant information found.'

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
