import os
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from fedotllm.llm import num_tokens_from_string, AIInference, OpenaiEmbeddings
from fedotllm.prompts.utils import structured_response # Assuming this is the correct import
import litellm

# Tests for num_tokens_from_string
def test_num_tokens_from_string(mocker):
    """Test num_tokens_from_string function."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # Simulate 5 tokens
    mocker.patch('tiktoken.get_encoding', return_value=mock_encoding)

    count = num_tokens_from_string("test string", "cl100k_base")

    mock_encoding.encode.assert_called_once_with("test string")
    assert count == 5

# Fixture for AIInference tests
@pytest.fixture
def mock_settings_env_ai(mocker):
    mock_get_settings = mocker.patch('fedotllm.llm.get_settings')
    mock_os_getenv = mocker.patch('os.getenv')

    # Default mock behavior for get_settings
    settings_mock = MagicMock()
    settings_mock.get.side_effect = lambda key, default=None: {
        "config.llm_api_base": "default_base_url",  # Corrected key
        "config.llm_model": "default_model"     # Corrected key
    }.get(key, default)
    mock_get_settings.return_value = settings_mock

    # Default mock behavior for os.getenv
    mock_os_getenv.side_effect = lambda key, default=None: {
        "FEDOTLLM_LLM_API_KEY": "default_api_key"
    }.get(key, default)

    return mock_get_settings, mock_os_getenv

# Tests for AIInference
def test_ai_inference_init_defaults(mocker, mock_settings_env_ai):
    """Test AIInference initialization with default settings."""
    mock_get_settings, mock_os_getenv = mock_settings_env_ai

    inference = AIInference()

    assert inference.base_url == "default_base_url"
    assert inference.model == "default_model"
    assert inference.api_key == "default_api_key"
    mock_get_settings.assert_called_once()
    mock_os_getenv.assert_any_call("FEDOTLLM_LLM_API_KEY") # Removed None


def test_ai_inference_init_custom_params(mocker, mock_settings_env_ai):
    """Test AIInference initialization with custom parameters."""
    inference = AIInference(api_key="custom_key", base_url="custom_url", model="custom_model")

    assert inference.api_key == "custom_key"
    assert inference.base_url == "custom_url"
    assert inference.model == "custom_model"

def test_ai_inference_init_no_api_key_raises(mocker, mock_settings_env_ai):
    """Test AIInference initialization raises Exception if no API key is found."""
    mock_get_settings, mock_os_getenv = mock_settings_env_ai
    mock_os_getenv.side_effect = lambda key, default=None: None # Simulate no API key in env

    # Also simulate no API key in settings if it were checked there directly
    settings_mock = MagicMock()
    settings_mock.get.side_effect = lambda key, default=None: {
        "config.llm_api_base": "default_base_url", # Corrected key
        "config.llm_model": "default_model"    # Corrected key
        # No LLM_API_KEY here
    }.get(key, default)
    mock_get_settings.return_value = settings_mock

    with pytest.raises(Exception, match="LLM API key is not set. Provide it via argument or FEDOTLLM_LLM_API_KEY env variable."): # Corrected message
        AIInference()

def test_ai_inference_query(mocker, mock_settings_env_ai):
    """Test AIInference query method."""
    mock_get_settings, mock_os_getenv = mock_settings_env_ai

    inference = AIInference() # Uses defaults from mock_settings_env_ai

    mock_litellm_completion = mocker.patch('litellm.completion')
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "mocked response content"
    mock_litellm_completion.return_value = mock_response

    user_message = "test user message"
    system_message = "test system message"

    response_content = inference.query(user_message, system_message=system_message, temperature=0.5, frequency_penalty=0.1)

    expected_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    # Check that litellm.completion was called with merged params
    # The exact call structure depends on how query_kwargs are handled by litellm
    # For this test, we ensure the main components are present.
    args, kwargs = mock_litellm_completion.call_args
    assert kwargs['model'] == inference.model
    assert kwargs['messages'] == expected_messages
    assert kwargs['api_key'] == inference.api_key
    assert kwargs['base_url'] == inference.base_url
    assert kwargs['temperature'] == 0.5
    assert kwargs['frequency_penalty'] == 0.1

    assert response_content == "mocked response content"

class MyModel(BaseModel):
    data: str

def test_ai_inference_create(mocker, mock_settings_env_ai):
    """Test AIInference create method for structured output."""
    inference = AIInference() # Uses defaults

    mock_query = mocker.patch.object(AIInference, 'query', return_value='{"data": "test_data"}')
    mock_parse_json = mocker.patch('fedotllm.llm.parse_json', return_value={"data": "test_data"}) # Corrected mock path
    # structured_response is imported from fedotllm.prompts.utils
    mock_structured_response = mocker.patch('fedotllm.prompts.utils.structured_response', return_value="structured_prompt_suffix")
    mock_model_validate = mocker.patch.object(MyModel, 'model_validate', side_effect=lambda x: MyModel(**x))


    user_message = "create a MyModel instance"
    result = inference.create(user_message, MyModel)

    mock_structured_response.assert_called_once_with(MyModel)
    mock_query.assert_called_once_with(
        user_message=user_message, # Corrected argument name
        system_message="structured_prompt_suffix"
    )
    mock_parse_json.assert_called_once_with('{"data": "test_data"}')
    mock_model_validate.assert_called_once_with({"data": "test_data"})

    assert isinstance(result, MyModel)
    assert result.data == "test_data"


# Fixture for OpenaiEmbeddings tests
@pytest.fixture
def mock_settings_env_embed(mocker):
    mock_get_settings = mocker.patch('fedotllm.llm.get_settings')
    mock_os_getenv = mocker.patch('os.getenv')

    settings_mock = MagicMock()
    settings_mock.get.side_effect = lambda key, default=None: {
        "config.embeddings_api_base": "default_embed_base_url", # Corrected key
        "config.embeddings_model": "default_embed_model",    # Corrected key
        "config.max_input_tokens": 8191                  # Corrected key
    }.get(key, default)
    mock_get_settings.return_value = settings_mock

    mock_os_getenv.side_effect = lambda key, default=None: {
        "FEDOTLLM_EMBEDDINGS_API_KEY": "default_embed_api_key"
    }.get(key, default)

    return mock_get_settings, mock_os_getenv

@pytest.fixture
def mock_openai_client(mocker):
    # This mock will represent the OpenAI client instance
    mock_client_instance = MagicMock()
    # Mock the constructor fedotllm.llm.OpenAI to return this instance
    mock_constructor = mocker.patch('fedotllm.llm.OpenAI', return_value=mock_client_instance)
    return mock_constructor, mock_client_instance


# Tests for OpenaiEmbeddings
def test_openai_embeddings_init_defaults(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings initialization with default settings."""
    mock_get_settings, mock_os_getenv = mock_settings_env_embed
    mock_constructor, _ = mock_openai_client

    embeddings = OpenaiEmbeddings()

    assert embeddings.api_key == "default_embed_api_key"
    assert embeddings.model == "default_embed_model"
    assert embeddings.base_url == "default_embed_base_url"
    assert embeddings.MAX_INPUT == 8191

    mock_constructor.assert_called_once_with(
        api_key="default_embed_api_key",
        base_url="default_embed_base_url"
    )

def test_openai_embeddings_init_custom_params(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings initialization with custom parameters."""
    mock_constructor, _ = mock_openai_client

    # Removed max_input_tokens from call, it's not an __init__ param
    embeddings = OpenaiEmbeddings(api_key="custom_key", base_url="custom_url", model="custom_model")

    assert embeddings.api_key == "custom_key"
    assert embeddings.base_url == "custom_url"
    assert embeddings.model == "custom_model"
    # MAX_INPUT is now set from settings or default, not direct param during init for this test design
    # If we wanted to test MAX_INPUT specifically, we'd adjust the settings mock for this test
    # For now, we assume the default from the fixture (or actual code default if settings mock doesn't provide it)
    # The test for MAX_INPUT is implicitly in test_openai_embeddings_encode_exceeds_max_input

    mock_constructor.assert_called_once_with(
        api_key="custom_key",
        base_url="custom_url"
    )

def test_openai_embeddings_init_no_api_key_raises(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings initialization raises Exception if no API key is found."""
    mock_get_settings, mock_os_getenv = mock_settings_env_embed
    mock_os_getenv.side_effect = lambda key, default=None: None # Simulate no API key in env

    settings_mock = MagicMock()
    settings_mock.get.side_effect = lambda key, default=None: {
        "config.embeddings_api_base": "default_embed_base_url", # Corrected key
        "config.embeddings_model": "default_embed_model"     # Corrected key
    }.get(key, default) # No API key from settings
    mock_get_settings.return_value = settings_mock

    with pytest.raises(Exception, match="Embeddings API key is not set. Provide it via argument or FEDOTLLM_EMBEDDINGS_API_KEY env variable."): # Corrected message
        OpenaiEmbeddings()

def test_openai_embeddings_encode_success(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings encode method successfully returns embeddings."""
    _, mock_client_instance = mock_openai_client

    embeddings = OpenaiEmbeddings() # Uses defaults

    mocker.patch('fedotllm.llm.num_tokens_from_string', return_value=100) # Less than MAX_INPUT

    mock_embedding_data = MagicMock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3]
    mock_api_response = MagicMock()
    mock_api_response.data = [mock_embedding_data] # API returns a list of embedding objects
    mock_client_instance.embeddings.create.return_value = mock_api_response

    test_input = "test input string"
    result = embeddings.encode(test_input)

    mock_client_instance.embeddings.create.assert_called_once_with(
        input=[test_input],
        model=embeddings.model,
        encoding_format="float" # as per llm.py
    )
    assert result == [mock_embedding_data]

def test_openai_embeddings_encode_exceeds_max_input(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings encode method raises Exception if input exceeds MAX_INPUT tokens."""
    mock_get_settings, mock_os_getenv = mock_settings_env_embed
    # Ensure MAX_INPUT is set to a known value for this test via settings mock
    settings_mock = MagicMock()
    test_max_input = 50
    settings_mock.get.side_effect = lambda key, default=None: {
        "config.embeddings_api_base": "default_embed_base_url",
        "config.embeddings_model": "default_embed_model",
        "config.max_input_tokens": test_max_input
    }.get(key, default)
    mock_get_settings.return_value = settings_mock

    _, mock_client_instance = mock_openai_client
    embeddings = OpenaiEmbeddings()
    assert embeddings.MAX_INPUT == test_max_input


    mocker.patch('fedotllm.llm.num_tokens_from_string', return_value=embeddings.MAX_INPUT + 1)

    with pytest.raises(Exception, match=f"Input exceeds the limit of {embeddings.MAX_INPUT} tokens for model {embeddings.model}. Given: {embeddings.MAX_INPUT + 1}"):
        embeddings.encode("a very long input string that exceeds token limit")

    mock_client_instance.embeddings.create.assert_not_called()

def test_openai_embeddings_encode_api_error(mocker, mock_settings_env_embed, mock_openai_client):
    """Test OpenaiEmbeddings encode method raises Exception on API error."""
    _, mock_client_instance = mock_openai_client
    embeddings = OpenaiEmbeddings()

    mocker.patch('fedotllm.llm.num_tokens_from_string', return_value=100)
    mock_client_instance.embeddings.create.side_effect = Exception("Mocked API Error")

    with pytest.raises(Exception, match="Embeddings generation failed!"):
        embeddings.encode("some input")
