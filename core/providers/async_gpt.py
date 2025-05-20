from __future__ import annotations

import base64
import json
import logging
import os
import time  # Import the time module
from io import BytesIO
from typing import Any
from typing import Optional
from typing import Union

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from openai import RateLimitError  # Import RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

load_dotenv()

# Configure logging
logging.basicConfig(
  level=logging.ERROR,
  filename="error.log",
  filemode="a",
  format="%(asctime)s - %(levelname)s - %(message)s",
)

VALID_RESOURCE_MODELS = {
  "gpt-4o",
  "gpt-4o-mini",
  "o1",
  "o3-mini",
  "gpt-4.1",
  "gpt-4.1-mini",
}


class AsyncAzureOpenAIProvider:
  """The async Azure OpenAI provider."""

  def __init__(self, resource_model_name: str):
    if resource_model_name not in VALID_RESOURCE_MODELS:
      raise ValueError(
        f"Invalid resource model name: {resource_model_name}. Must be one of: {', '.join(VALID_RESOURCE_MODELS)}"
      )
    self.resource_model_name = resource_model_name
    self._client = None  # Initialize client as None
    self.input_tokens: int = 0
    self.output_tokens: int = 0

  @property
  def client(self) -> AsyncAzureOpenAI:
    """
    Returns the AsyncAzureOpenAI client. Initializes it only if it hasn't been initialized yet.
    """
    if self._client is None:
      endpoint_map = {
        "gpt-4o": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "gpt-4o-mini": os.getenv("AZURE_OPENAI_ENDPOINT_MINI"),
        "o1": os.getenv("AZURE_OPENAI_SMART"),
        "o3-mini": os.getenv("AZURE_OPENAI_O3"),
        "gpt-4.1": os.getenv("AZURE_OPENAI_GPT4.1"),
        "gpt-4.1-mini": os.getenv("AZURE_OPENAI_ENDPOINT_4.1_MINI"),
      }
      endpoint = endpoint_map[self.resource_model_name]

      self._client = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
      )
    return self._client

  async def get_response(
    self,
    prompt: str,
    temperature: float = 0.5,
    full_response: bool = False,
    image_paths: list[str] | None = None,
    format: bool = False,
    return_tokens: bool = False,  # Added return_tokens parameter
  ) -> Union[
    str,
    dict[str, Any],
    ChatCompletion,
    tuple[str, int, int],
    tuple[dict[str, Any], int, int],
  ]:  # Updated return type to include tuples
    messages = self._get_messages(prompt=prompt, image_paths=image_paths)
    request_args = {
      "messages": messages,
      "model": self.resource_model_name,
    }
    if self.resource_model_name in {"o1", "o3-mini"}:
      temperature = 1
    request_args["temperature"] = temperature
    if format:
      request_args["response_format"] = {"type": "json_object"}

    try:  # Wrap the API call in a try-except block
      response = await self.client.chat.completions.create(**request_args)
    except RateLimitError as e:  # Catch RateLimitError
      print(
        f"Rate limit error (429) encountered: {e}"
      )  # Optional: Log the error to console
      logging.error(f"Rate limit error (429): {e}")  # Log the error to error.log
      print(
        "Waiting for 20 seconds before retrying..."
      )  # Optional: Inform user about retry
      time.sleep(20)  # Wait for 20 seconds
      return await self.get_response(  # Retry the request
        prompt=prompt,
        temperature=temperature,
        full_response=full_response,
        image_paths=image_paths,
        format=format,
        return_tokens=return_tokens,  # Pass return_tokens in retry
      )
    except Exception as e:  # Catch other potential exceptions
      print(f"An unexpected error occurred: {e}")  # Optional: Log unexpected errors
      logging.error(f"Unexpected error during API call: {e}")  # Log unexpected errors
      raise  # Re-raise the exception to handle it upstream

    self._log_token_usage(response)

    if full_response:
      return response

    content = response.choices[0].message.content

    if format:
      try:
        if return_tokens:
          return (
            json.loads(content),
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
          )
        else:
          return json.loads(content)
      except json.decoder.JSONDecodeError:
        return None  # or raise error if you prefer
    else:
      if return_tokens:
        return content, response.usage.prompt_tokens, response.usage.completion_tokens
      else:
        return content

  def _log_token_usage(self, response: ChatCompletion):
    """Logs token usage information from the API response."""
    # print("Total tokens used:", response.usage.total_tokens)
    # print("Prompt tokens:", response.usage.prompt_tokens)
    # print("Completion tokens:", response.usage.completion_tokens)
    self.input_tokens += response.usage.prompt_tokens
    self.output_tokens += response.usage.completion_tokens
    pass

  @staticmethod
  def _get_messages(
    prompt: str, image_paths: Optional[list[str]] = None
  ) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    user_message_content: list[dict[str, Any]] = [
      {"type": "text", "text": prompt.strip()}
    ]

    if image_paths:
      for image_path in image_paths:
        encoded_image = AsyncAzureOpenAIProvider._encode_image(image_path)
        if encoded_image:
          user_message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
          })

    messages.append(
      ChatCompletionUserMessageParam(role="user", content=user_message_content)
    )
    return messages

  @staticmethod
  def _encode_image(image_path: str) -> Optional[str]:
    max_size_bytes = 20 * 1024 * 1024  # 20 MB limit
    try:
      image_size = os.path.getsize(image_path)
      if image_size > max_size_bytes:
        with Image.open(image_path) as img:
          scale_factor = (max_size_bytes / image_size) ** 0.5
          new_dimensions = (
            int(img.width * scale_factor),
            int(img.height * scale_factor),
          )
          img = img.resize(new_dimensions, Image.LANCZOS)
          buffer = BytesIO()
          img.save(buffer, format="JPEG", quality=85)
          buffer_size = buffer.tell()
          while buffer_size > max_size_bytes:
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            buffer_size = buffer.tell()
          return base64.b64encode(buffer.getvalue()).decode("utf-8")
      else:
        with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
      print(f"Error encoding image: {e}")
      return None

  @staticmethod
  def parse_json_string(input_string: str) -> dict:
    try:
      if input_string.startswith("`json") and input_string.endswith("`"):
        json_part = input_string[len("`json") : -len("`")].strip()
        return json.loads(json_part)
      else:
        raise ValueError("Input string does not follow the '`json ... `' format.")
    except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON: {e}")

  def __repr__(self) -> str:
    return self.resource_model_name
