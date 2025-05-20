from __future__ import annotations

import base64
import json
import logging
import os
import re
from functools import cached_property
from io import BytesIO
from typing import Any
from typing import Optional
from typing import Union

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

load_dotenv(override=True)

# Configure logging
logging.basicConfig(
  level=logging.ERROR,
  filename="error.log",
  filemode="a",
  format="%(asctime)s - %(levelname)s - %(message)s",
)


class AsyncGeminiProvider:
  """The async Gemini provider using OpenAI library for compatibility."""

  def __init__(self, model_name: str = "gemini-2.0-flash"):
    self.model_name = model_name
    self._base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    self.api_key = os.getenv("GEMINI_API_KEY")
    self.input_tokens = 0
    self.output_tokens = 0

  @cached_property
  def client(self) -> AsyncOpenAI:
    client = AsyncOpenAI(
      api_key=self.api_key,
      base_url=self._base_url,
      timeout=120,
      default_headers={
        "Content-Type": "application/json",
      },
      # Add other necessary headers if required
    )
    return client

  async def get_response(
    self,
    prompt: str,
    temperature: float = 0.5,
    full_response: bool = False,
    image_paths: list[str] | list[bytes] | None = None,
    format: bool = False,
    return_tokens: bool = False,
    stream: bool = False,
  ) -> Union[str, dict[str, Any], ChatCompletion]:
    messages = self._get_messages(prompt=prompt, image_paths=image_paths)
    request_args = {
      "messages": messages,
      "model": self.model_name,
    }
    request_args["temperature"] = temperature
    request_args["stream"] = stream
    if format:
      request_args["response_format"] = {"type": "json_object"}

    response: ChatCompletion = await self.client.chat.completions.create(**request_args)
    self._log_token_usage(response)

    if full_response:
      return response

    content = response.choices[0].message.content.strip()
    content = content.replace(
      "\u00a0", " "
    )  # replace non breaking space with regular space.
    content = re.sub(r"[^\x20-\x7E]+", "", content)
    # time.sleep(0.5)
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
      except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}, content: '{repr(content)}'")
        raise e
      except Exception as e:
        print(f"General Error: {e}, content: '{repr(content)}'")
        raise e
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
    prompt: str, image_paths: Optional[list[str] | list[bytes]] = None
  ) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    user_message_content: list[dict[str, Any]] = [
      {"type": "text", "text": prompt.strip()}
    ]

    if image_paths:
      for image_path in image_paths:
        encoded_image = AsyncGeminiProvider._encode_image(image_path)
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
  def _encode_image(image_data: Union[str, bytes]) -> Optional[str]:
    max_size_bytes = 20 * 1024 * 1024  # 20 MB limit

    try:
      if isinstance(image_data, str):  # Path provided
        image_size = os.path.getsize(image_data)
        with open(image_data, "rb") as image_file:
          image_bytes = image_file.read()
      elif isinstance(image_data, bytes):  # Bytes provided
        image_bytes = image_data
        image_size = len(image_bytes)
      else:
        raise ValueError("Invalid image data type. Must be str or bytes.")

      if image_size > max_size_bytes:
        with Image.open(BytesIO(image_bytes)) as img:  # Use BytesIO for bytes
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
        return base64.b64encode(image_bytes).decode("utf-8")  # Encode original bytes

    except Exception as e:
      print(f"Error encoding image: {e}")
      return None

  @staticmethod
  def parse_json_string(input_string: str) -> dict:
    try:
      if input_string.startswith("```json") and input_string.endswith("```"):
        json_part = input_string[len("```json") : -len("```")].strip()
        return json.loads(json_part)
      else:
        raise ValueError("Input string does not follow the '```json ... ```' format.")
    except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON: {e}")

  def __repr__(self) -> str:
    return self.model_name
