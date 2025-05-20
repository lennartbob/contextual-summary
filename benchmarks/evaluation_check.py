from benchmarks.dataset import Row
from core.providers.async_gpt import AsyncAzureOpenAIProvider
from benchmarks.utils import process_template


async def self_check(row:Row, answer:str, llm:AsyncAzureOpenAIProvider) -> bool:

    prompt = process_template(
        "self_check.jinja",
        {"question": row.prompt, "groundtruth": row.answer, "answer":answer}
    )
    r = await llm.get_response(prompt)

    if r.lower() == 'true': 
        return True
    elif r.lower() == 'false':
        return False
    else:
        print("not returning either true or false", r)
        return False
