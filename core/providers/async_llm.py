from __future__ import annotations

import ast
import base64
import json
import os
import re
from functools import cached_property
from io import BytesIO
from json import JSONDecodeError
from typing import Any
from typing import Optional
from typing import Union

import json5
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionUserMessageParam
from PIL import Image

load_dotenv()

available_models = {
  "deepseek-chat",
  "deepseek-reasoner",
}


class AsyncLocalLLM:
  """The async LocalLLM provider using OpenAI library for compatibility."""

  def __init__(
    self, model_name: str = "Qwen/QwQ-32B", url: str = "http://51.159.177.64:8000/v1"
  ):
    self.model_name = model_name
    self.base_url = url
    self.api_key = "EMPTY"
    self.input_tokens = 0
    self.output_tokens = 0

  @cached_property
  def client(self) -> AsyncOpenAI:
    client = AsyncOpenAI(
      api_key=self.api_key,
      base_url=self.base_url,
      default_headers={
        "Content-Type": "application/json",
      },
    )
    return client

  async def get_response(
    self,
    prompt: str,
    temperature: float = 0.5,
    format: bool = False,
    full_response: bool = True,
    image_paths: Optional[list[str | bytes]] = None,
  ) -> Union[str, dict[str, Any], ChatCompletion]:
    messages = self._get_messages(prompt=prompt, image_paths=image_paths)
    request_args = {
      "messages": messages,
      "model": self.model_name,
      "temperature": temperature,
      "max_tokens": 5000,
    }
    retry_count = 3
    for i in range(retry_count):
      try:
        response = await self.client.chat.completions.create(**request_args)
        break  # Break out of the retry loop if successful
      except JSONDecodeError as e:
        if i < retry_count - 1:
          print(
            f"JSONDecodeError encountered, retrying {i + 1}/{retry_count}... Error: {e}"
          )
          continue  # Retry if not the last attempt
        else:
          raise  # Re-raise the exception if all retries fail

    self._log_token_usage(response)

    content = response.choices[0].message.content

    if "</think" in content:
      content = self.parse_out_thinking(content)

    if format is True:
      try:
        # Strategy 1: Extract JSON from code blocks if present
        code_blocks = re.findall(r"`(?:json)?\n(.*?)\n`", content, re.DOTALL)
        for block in code_blocks:
          parsed = self.parse_json_like_string(block.strip())
          if parsed is not None:
            return parsed

        # Strategy 2: Try parsing the entire response
        parsed = self.parse_json_like_string(content.strip())
        if parsed is not None:
          return parsed

        # Strategy 3: Fallback to raw response if all parsing fails
        return content

      except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return content
    else:
      return content

  def _log_token_usage(self, response: ChatCompletion):
    """Logs token usage information from the API response."""
    self.input_tokens += response.usage.prompt_tokens
    self.output_tokens += response.usage.completion_tokens

  @staticmethod
  def _get_messages(
    prompt: str, image_paths: Optional[list[str | bytes]] = None
  ) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
    # Truncate prompt if needed
    user_message_content: list[dict[str, Any]] = [
      {"type": "text", "text": prompt.strip()}
    ]

    if image_paths:
      for image_input in image_paths:
        encoded_image = AsyncLocalLLM._encode_image(image_input)
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
        with Image.open(BytesIO(image_bytes)) as img:
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
        return base64.b64encode(image_bytes).decode("utf-8")

    except Exception as e:
      print(f"Error encoding image: {e}")
      return None

  @staticmethod
  def parse_json_like_string(input_string: str) -> dict | list | None:
    """
    Robust JSON parsing with multiple fallback strategies.
    Handles:
    - Standard JSON
    - JSON5 (lenient JSON with comments, trailing commas)
    - Python-like dictionaries with single quotes
    - JSON embedded in text/code blocks
    """
    try:
      return json.loads(input_string)
    except json.JSONDecodeError:
      pass

    start_idx = input_string.find("{")
    end_idx = input_string.rfind("}")

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
      json_candidate = input_string
    else:
      json_candidate = input_string[start_idx : end_idx + 1]

    parsing_attempts = [
      lambda: json.loads(json_candidate),
      lambda: json5.loads(json_candidate),
      lambda: ast.literal_eval(json_candidate),
      lambda: json5.loads(re.sub(r",(\s*[}\]])", r"\1", json_candidate)),
      lambda: json5.loads(json_candidate.replace("'", '"')),
    ]

    for attempt in parsing_attempts:
      try:
        result = attempt()
        if isinstance(result, (dict, list)):
          return result
      except Exception:
        continue

    try:
      return json5.loads(input_string)
    except json5.JSONDecodeError:
      return None

  @staticmethod
  def parse_out_thinking(response: str) -> str:
    """
    Parses out the content enclosed in <think>...</think> tags robustly.

    It removes everything within <think> and </think> tags.
    If both tags are not found, it removes everything above the first </think> it encounters.
    """
    start_tag = "<think>"
    end_tag = "</think>"
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)

    if start_index != -1 and end_index != -1:
      # Both tags found, remove the content within
      thinking_pattern = r"<think>.*?</think>"
      return re.sub(thinking_pattern, "", response, flags=re.DOTALL).strip()
    elif end_index != -1:
      # Only the closing tag is found, remove everything before it
      return response[end_index + len(end_tag) :].strip()
    else:
      # Neither tag found, return the original response
      return response.strip()

  def __repr__(self) -> str:
    return self.model_name
