import logging
logger = logging.getLogger(__name__)
import json
import re
from json_repair import repair_json

def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        logger.warning("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input, re.DOTALL)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json") :]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]
    if input.startswith('"'):
        input = input[1:]
    if input.endswith('"'):
        input = input[: len(input) - 1]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            logger.warning("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                logger.warning("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result


if __name__ == "__main__":

    aa = '"{ ""named_entities"": [""redshift 1.0"", ""redshift 2.0""] }"'
    print(aa)
    print('='*20)
    bb = try_parse_json_object(aa)
    print(bb)
    print('='*20)
    if isinstance(bb, dict):
        entities = bb.get('named_entities', [])

        print(entities)
        print('='*20)

