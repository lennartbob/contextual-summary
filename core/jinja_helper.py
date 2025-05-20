from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from typing import Optional

from jinja2 import BaseLoader
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import meta
from jinja2 import select_autoescape


def process_template(
  template_file: str, data: dict[str, Any], parent_path: Optional[str] = None
) -> str:
  """Process the jinja template into a string.

  Function has been inspired by: https://github.com/ArjanCodes/examples/blob/main/2024/tuesday_tips/jinja2/jinja_helper.py

  Args:
    template_file (str): The name of the jinja prompt template.
    data (Dict[str, Any]): The parameters and their values to insert into the prompt.
    parent_path (Optional[str]): The parent path for the template, used for handling relative paths within templates (e.g. using include or extends).

  Returns:
    The formatted prompt as a string.
  """
  if not parent_path:
    parent_path = Path(__file__).parent.absolute().as_posix()
  jinja_env: Environment = Environment(
    loader=FileSystemLoader(searchpath=parent_path + "/prompts"),
    autoescape=select_autoescape(),
  )

  # Check if the template exists
  try:
    template = jinja_env.get_template(template_file)
  except Exception as e:
    raise FileNotFoundError(f"Template file '{template_file}' not found.") from e

  template_source = jinja_env.loader.get_source(jinja_env, template_file)[0]
  parsed_content = jinja_env.parse(template_source)
  template_variables = meta.find_undeclared_variables(parsed_content)

  # Check if all variables in template have been provided as data
  missing_keys = set(template_variables) - set(data.keys())
  extra_keys = set(data.keys()) - set(template_variables)

  if missing_keys:
    raise ValueError(f"Missing data for variables in template: {missing_keys}")

  if extra_keys:
    print(f"Warning: The following keys are not used in the template: {extra_keys}")

  return template.render(**data)


def extract_variables(template_file: str, jinja_env: Environment) -> list[Any]:
  """Extract all variables in a Jinja template in string format.

  Args:
    template_file (str): the name of the jinja prompt template.
    jinja_env (Environment): the jinja Environment.

  Returns:
    A list of all the identified variables in the string template.
  """
  # Check if the baseloader is None
  if not jinja_env.loader:
    raise Exception("Something went wrong formatting the prompt template.")
  else:
    loader: BaseLoader = jinja_env.loader

  # Get the template as plain text
  plain_template: str = loader.get_source(jinja_env, template_file)[0]

  variable_pattern: str = r"\{\{ *([\w_]+) *\}\}"
  return re.findall(variable_pattern, plain_template)
