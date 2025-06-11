import pytest
import json
from fedotllm.agents.utils import jinja_render, render, extract_code, parse_json
from jinja2.exceptions import UndefinedError

# Tests for jinja_render
def test_jinja_render_basic():
    """Test basic template rendering."""
    template = "Hello, {{ name }}!"
    rendered_string = jinja_render(template, name="World")
    assert rendered_string == "Hello, World!"

def test_jinja_render_with_multiple_args():
    """Test template rendering with multiple arguments."""
    template = "The sum of {{ num1 }} and {{ num2 }} is {{ num1 + num2 }}."
    rendered_string = jinja_render(template, num1=5, num2=10)
    assert rendered_string == "The sum of 5 and 10 is 15."

def test_jinja_render_undefined_variable():
    """Test template rendering with an undefined variable (should raise UndefinedError)."""
    template = "Hello, {{ name }}!"
    with pytest.raises(UndefinedError):
        jinja_render(template) # No 'name' provided

# Tests for render
def test_render_user_only():
    """Test rendering with only a user prompt."""
    prompt_config = {"user": "User message: {{ content }}"}
    user_msg, system_msg, temp, freq_penalty = render(prompt_config, content="test content")
    assert user_msg == "User message: test content"
    assert system_msg is None
    assert temp == 0.2  # Default
    assert freq_penalty == 0.0  # Default

def test_render_user_and_system():
    """Test rendering with both user and system prompts."""
    prompt_config = {
        "system": "System message: {{ role }}",
        "user": "User message: {{ content }}",
        "temperature": 0.5,
        "frequency_penalty": 0.1
    }
    user_msg, system_msg, temp, freq_penalty = render(prompt_config, role="AI", content="test")
    assert user_msg == "User message: test"
    assert system_msg == "System message: AI"
    assert temp == 0.5
    assert freq_penalty == 0.1

def test_render_custom_params():
    """Test rendering with custom temperature and frequency_penalty."""
    prompt_config = {
        "user": "User message",
        "temperature": 0.7,
        "frequency_penalty": 0.5
    }
    _, _, temp, freq_penalty = render(prompt_config)
    assert temp == 0.7
    assert freq_penalty == 0.5

def test_render_no_system_default_params():
    """Test rendering with no system prompt and default parameters."""
    prompt_config = {"user": "User message"}
    user_msg, system_msg, temp, freq_penalty = render(prompt_config)
    assert user_msg == "User message"
    assert system_msg is None
    assert temp == 0.2
    assert freq_penalty == 0.0

# Tests for extract_code
def test_extract_code_with_python_block():
    """Test extracting code from a markdown python block."""
    response = "Some text\n```python\nprint('Hello')\n```\nMore text"
    assert extract_code(response) == "print('Hello')"

def test_extract_code_with_generic_block():
    """Test extracting code from a generic markdown block."""
    response = "```\ncode here\n```"
    assert extract_code(response) == "code here"

def test_extract_code_no_block():
    """Test extracting code when no markdown block is present."""
    response = "Just plain text code print('Hello')"
    assert extract_code(response) == "Just plain text code print('Hello')"

def test_extract_code_entire_response_is_code_in_block():
    """Test when the entire response is a code block."""
    response = "```python\nimport os\nos.system('echo hi')\n```"
    assert extract_code(response) == "import os\nos.system('echo hi')"

def test_extract_code_empty_block():
    """Test extracting code from an empty markdown block."""
    response = "```\n\n```"
    assert extract_code(response) == ""

def test_extract_code_block_with_leading_trailing_whitespace():
    """Test code block with leading/trailing whitespace inside the block."""
    response = "```python\n  print('Padded')  \n```"
    assert extract_code(response) == "print('Padded')"

# Tests for parse_json
def test_parse_json_valid():
    """Test parsing a valid JSON string."""
    raw_reply = '{"key": "value", "number": 123}'
    expected_dict = {"key": "value", "number": 123}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_with_backticks():
    """Test parsing JSON enclosed in markdown-style backticks."""
    raw_reply = '```json\n{"name": "test", "valid": true}\n```'
    expected_dict = {"name": "test", "valid": True}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_with_backticks_no_lang_specifier():
    """Test parsing JSON in backticks without 'json' specifier."""
    raw_reply = '```\n{"name": "test", "valid": true}\n```'
    expected_dict = {"name": "test", "valid": True}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_just_object_no_backticks():
    """Test parsing a JSON string that is just the object, no backticks."""
    raw_reply = '{"key": "value"}'
    assert parse_json(raw_reply) == {"key": "value"}

def test_parse_json_repairable_trailing_comma_object():
    """Test parsing JSON with a trailing comma in an object (handled by json_repair)."""
    raw_reply = '{"key": "value",}'
    expected_dict = {"key": "value"}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_repairable_trailing_comma_array():
    """Test parsing JSON with a trailing comma in an array (handled by json_repair)."""
    raw_reply = '{"key": ["item1", "item2",]}'
    expected_dict = {"key": ["item1", "item2"]}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_repairable_comments():
    """Test parsing JSON with comments (handled by json_repair)."""
    raw_reply = '''
    {
        // This is a comment
        "key": "value", // Another comment
        "number": 123
    }
    '''
    expected_dict = {"key": "value", "number": 123}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_repairable_single_quotes():
    """Test parsing JSON with single quotes (handled by json_repair)."""
    raw_reply = "{'key': 'value'}"
    expected_dict = {"key": "value"}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_invalid_not_json():
    """Test parsing a string that is not JSON."""
    raw_reply = "This is not JSON at all."
    assert parse_json(raw_reply) is None

def test_parse_json_empty_string():
    """Test parsing an empty string."""
    raw_reply = ""
    assert parse_json(raw_reply) is None

def test_parse_json_incomplete_json_in_backticks():
    """Test parsing incomplete JSON within backticks, expecting None if not repairable."""
    raw_reply = '```json\n{"key": "value" \n```' # Missing closing brace
    assert parse_json(raw_reply) is None

def test_parse_json_mixed_content_with_json_block():
    """Test parsing when JSON block is mixed with other text."""
    raw_reply = 'Some introductory text.\n```json\n{"data": "content"}\n```\nSome concluding text.'
    expected_dict = {"data": "content"}
    assert parse_json(raw_reply) == expected_dict

def test_parse_json_complex_nested_structure():
    """Test parsing a more complex, nested JSON structure."""
    raw_reply = '{"level1": {"key1": "value1", "level2": {"numbers": [1, 2, 3], "boolean": false}}}'
    expected_dict = {"level1": {"key1": "value1", "level2": {"numbers": [1, 2, 3], "boolean": False}}}
    assert parse_json(raw_reply) == expected_dict
