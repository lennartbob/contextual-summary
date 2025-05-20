


from core.providers.async_gemini import AsyncGeminiProvider
from benchmarks.utils import process_template

async def query_expansion(query:str) -> list[str]|None:

    gemini = AsyncGeminiProvider("gemini-2.0-flash")

    prompt = process_template(
        "query_expansion.jinja",
        {"query":query}
    )
    response = await gemini.get_response(prompt, format=True)

    if "expanded_queries" not in response:
        print("expanded_query not in response", response)
        return None
    
    if isinstance(response["expanded_queries"],str) and response["expanded_queries"].lower() == "none":
        return None
    else:
        assert isinstance(response["expanded_queries"],list)
        return response["expanded_queries"]
    



    