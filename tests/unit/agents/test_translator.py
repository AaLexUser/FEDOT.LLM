import pytest
from unittest.mock import patch, MagicMock, ANY

# Assuming AIInference is in fedotllm.llm
from fedotllm.llm import AIInference 
from fedotllm.agents.translator import TranslatorAgent
from langdetect import LangDetectException

# No longer need MOCK_LANGUAGES from googletrans

@pytest.fixture
def mock_inference_fixture(): # Renamed to avoid conflict with parameter names in tests
    return MagicMock(spec=AIInference)

class TestTranslatorAgent:

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text" 
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == 'es'
        mock_detect.assert_called_once_with("Hola mundo")

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_failure_defaults_to_english(self, mock_detect, mock_inference_fixture):
        mock_detect.side_effect = LangDetectException(0, "Detection failed")
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.create.return_value = "Translated text"
        agent.translate_input_to_english("Invalid text for detection")
        assert agent.source_language == 'en'
        mock_detect.assert_called_once_with("Invalid text for detection")

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_english_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        mock_inference_fixture.query.return_value = "Hello world"

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english("Hola mundo")
        
        assert translated_text == "Hello world"
        assert agent.source_language == 'es'
        mock_inference_fixture.query.assert_called_once()
        
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert "Translate the following text from es to en." in prompt
        assert "Hola mundo" in prompt

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_english_inference_error_returns_original(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        mock_inference_fixture.query.side_effect = Exception("LLM API error")

        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Hola mundo"
        translated_text = agent.translate_input_to_english(original_text)
        
        assert translated_text == original_text 
        assert agent.source_language == 'es'

    @patch('fedotllm.agents.translator.detect')
    def test_english_input_not_translated(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'en'
        
        agent = TranslatorAgent(inference=mock_inference_fixture)
        input_text = "Hello world, this is English."
        translated_text = agent.translate_input_to_english(input_text)
        
        assert translated_text == input_text
        assert agent.source_language == 'en'
        mock_inference_fixture.query.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    def test_translation_to_source_language_success(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        
        mock_inference_fixture.query.return_value = "Hello world" 
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == 'es'
        
        mock_inference_fixture.query.reset_mock() 
        mock_inference_fixture.query.return_value = "Hola mundo otra vez"
        
        translated_output = agent.translate_output_to_source_language("Hello world again")
        
        assert translated_output == "Hola mundo otra vez"
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert "Translate the following text from English to es." in prompt
        assert "Hello world again" in prompt


    def test_extract_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text_with_code = """Some text
```python
print('hello')
```
More text
```
x = 1
```"""
        processed_text, code_map = agent._extract_code_blocks(text_with_code)
        
        placeholder_0 = f"{agent.code_block_placeholder_prefix}_0__"
        placeholder_1 = f"{agent.code_block_placeholder_prefix}_1__"

        assert placeholder_0 in processed_text
        assert placeholder_1 in processed_text
        assert "print('hello')" not in processed_text
        assert "x = 1" not in processed_text
        
        assert len(code_map) == 2
        assert code_map[placeholder_0] == "```python\nprint('hello')\n```"
        assert code_map[placeholder_1] == "```\nx = 1\n```"

    def test_reinsert_code_blocks(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        text_with_placeholders = f"""Texto traducido
{agent.code_block_placeholder_prefix}_0__
Más texto traducido
{agent.code_block_placeholder_prefix}_1__"""
        code_map = {
            f"{agent.code_block_placeholder_prefix}_0__": "```python\nprint('hello')\n```",
            f"{agent.code_block_placeholder_prefix}_1__": "```\nx = 1\n```"
        }
        
        final_text = agent._reinsert_code_blocks(text_with_placeholders, code_map)
        
        expected_text = """Texto traducido
```python
print('hello')
```
Más texto traducido
```
x = 1
```"""
        assert final_text == expected_text
        assert agent.code_block_placeholder_prefix not in final_text


    @patch('fedotllm.agents.translator.detect')
    def test_code_block_preservation_e2e(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        original_text = """Texto antes
```python
# Esto es un comentario
print('Hola')
```
Texto después"""
        
        # Use the agent's placeholder prefix to construct the expected placeholder
        agent_for_placeholder = TranslatorAgent(inference=mock_inference_fixture) # Temp agent to get prefix if needed
        placeholder_0 = f"{agent_for_placeholder.code_block_placeholder_prefix}_0__"

        placeholder_text_from_llm = f"""Translated before
{placeholder_0}
Translated after"""
        mock_inference_fixture.query.return_value = placeholder_text_from_llm

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english(original_text)

        expected_final_text = """Translated before
```python
# Esto es un comentario
print('Hola')
```
Translated after"""
        assert translated_text == expected_final_text
        
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert f"placeholders like '{agent.code_block_placeholder_prefix}_NUMBER__'" in prompt
        assert "MUST NOT be translated or altered" in prompt
        assert placeholder_0 in prompt 

    @patch('fedotllm.agents.translator.detect')
    def test_markdown_preservation_prompting(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'fr'
        original_text = "# Titre\nCeci est du **gras** et de l'*italique*."
        
        mock_inference_fixture.query.return_value = "# Title\nThis is **bold** and *italic*."

        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.translate_input_to_english(original_text)

        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')

        assert "Translate the following text from fr to en." in prompt
        assert "crucial to preserve the original formatting exactly" in prompt
        assert "markdown syntax: headers" in prompt
        assert "bold (e.g., **text** or __text__)" in prompt
        assert "italics (e.g., *text* or _text_)" in prompt
        assert "links (e.g., [text](url))" in prompt
        assert "tables (using pipe and hyphen syntax)" in prompt
        # In this case, original_text has no code blocks, so it's passed as is to the prompt
        assert original_text in prompt 

    @patch('fedotllm.agents.translator.detect')
    def test_unsupported_language_detection_still_calls_llm(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'xx'
        original_text = "Texte dans une langue inconnue."
        translated_by_llm = "Text in an unknown language, translated by LLM."
        mock_inference_fixture.query.return_value = translated_by_llm

        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english(original_text)

        assert translated_text == translated_by_llm
        mock_inference_fixture.query.assert_called_once()
        args, kwargs = mock_inference_fixture.query.call_args
        prompt = args[0] if args else kwargs.get('messages', '')
        assert "Translate the following text from xx to en." in prompt
        assert original_text in prompt

    # New tests to improve coverage

    @patch('fedotllm.agents.translator.detect')
    def test_translate_input_empty_text(self, mock_detect, mock_inference_fixture):
        # Now, empty text should return early, not calling detect.
        agent = TranslatorAgent(inference=mock_inference_fixture)
        translated_text = agent.translate_input_to_english("")
        assert translated_text == ""
        assert agent.source_language == 'en' # Default set even when returning early
        mock_detect.assert_not_called()      # detect() should not be called
        mock_inference_fixture.query.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    def test_translate_input_whitespace_text(self, mock_detect, mock_inference_fixture):
        # Whitespace text should also return early.
        agent = TranslatorAgent(inference=mock_inference_fixture)
        input_text = "     "
        translated_text = agent.translate_input_to_english(input_text)
        assert translated_text == input_text
        assert agent.source_language == 'en' # Default set even when returning early
        mock_detect.assert_not_called()      # detect() should not be called
        mock_inference_fixture.query.assert_not_called() # No translation needed

    def test_translate_output_empty_message(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = 'es' # Set a source language
        translated_output = agent.translate_output_to_source_language("")
        assert translated_output == ""
        mock_inference_fixture.query.assert_not_called()

    def test_translate_output_whitespace_message(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = 'es'
        message = "     "
        translated_output = agent.translate_output_to_source_language(message)
        # Expecting the original whitespace message to be returned as no actual translation should occur
        assert translated_output == message
        mock_inference_fixture.query.assert_not_called()


    def test_translate_output_source_language_not_set(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = None # Explicitly set to None
        message = "Hello world"
        # Should default to 'en', and since message is 'en', no translation
        translated_output = agent.translate_output_to_source_language(message)
        assert translated_output == message
        mock_inference_fixture.query.assert_not_called()

    def test_translate_output_source_language_is_english(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = 'en'
        message = "This is already in English."
        translated_output = agent.translate_output_to_source_language(message)
        assert translated_output == message
        mock_inference_fixture.query.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    def test_translate_output_source_language_not_set_but_differs(self, mock_detect, mock_inference_fixture):
        # This test checks if _translate_text is called when source_language is None,
        # implying it defaults to 'en', but we want to translate to something else (hypothetically).
        # However, translate_output_to_source_language defaults source to 'en' if not set,
        # and if the target is also 'en' (implicit for this method if source_language is None),
        # it won't translate.
        # To truly test _translate_text through this, source_language must be set.
        # Let's adjust to test _translate_text more directly or a different scenario.

        # Scenario: source_language is None, so it's 'en'. We want to translate "Back to English" (which is already 'en')
        # This should not call the query.
        agent = TranslatorAgent(inference=mock_inference_fixture)
        agent.source_language = None
        translated_output = agent.translate_output_to_source_language("Back to English")
        assert translated_output == "Back to English"
        assert agent.source_language is None # It's not set by translate_output
        mock_inference_fixture.query.assert_not_called()


    # Tests for _translate_text method (indirectly via public methods or directly if made public for testing)
    # For simplicity, we'll test _translate_text by calling it directly.
    # If _translate_text is strictly private, these tests would need to be adapted
    # to go through translate_input_to_english or translate_output_to_source_language.

    def test_internal_translate_text_llm_returns_string(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.query.return_value = "Translated string"
        # Calling _translate_text(self, text: str, target_language: str, source_language: str = "auto")
        translated = agent._translate_text("Some text", "en", "fr")
        assert translated == "Translated string"
        mock_inference_fixture.query.assert_called_once()
        args, _ = mock_inference_fixture.query.call_args
        assert "Translate the following text from fr to en." in args[0]

    def test_internal_translate_text_llm_returns_non_string_falls_back(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Some original text"
        # Mocking a non-string response (e.g., an object, list)
        mock_response_obj = MagicMock()
        mock_response_obj.content = "This content won't be used"
        mock_inference_fixture.query.return_value = mock_response_obj

        # _translate_text should fallback to original (processed) text if LLM response is not a string
        translated = agent._translate_text(original_text, "en", "de")
        assert translated == original_text # Fallback behavior
        mock_inference_fixture.query.assert_called_once()

    def test_internal_translate_text_llm_returns_empty_string_response(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Text to translate"
        # LLM returns an empty string
        mock_inference_fixture.query.return_value = ""

        translated = agent._translate_text(original_text, "en", "ja")
        # According to current logic, if LLM returns empty string, it's treated as "not translated"
        # and the original processed text is reinserted with code blocks.
        # If original_text has no code blocks, processed_text == original_text.
        assert translated == original_text
        mock_inference_fixture.query.assert_called_once()

    def test_internal_translate_text_llm_returns_none_response(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Another text"
        mock_inference_fixture.query.return_value = None # LLM returns None

        translated = agent._translate_text(original_text, "en", "ko")
        # Similar to empty string, None response should lead to fallback
        assert translated == original_text
        mock_inference_fixture.query.assert_called_once()

    def test_internal_translate_text_llm_api_error_returns_original(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_text = "Text that will fail to translate"
        mock_inference_fixture.query.side_effect = Exception("LLM API Error")

        translated = agent._translate_text(original_text, "en", "es")
        assert translated == original_text # Should return original on error
        mock_inference_fixture.query.assert_called_once()

    def test_internal_translate_text_empty_input_string(self, mock_inference_fixture):
        # Note: _translate_text itself doesn't prevent calling with empty string,
        # the public methods (translate_input_to_english) do the check.
        # If _translate_text is called with empty string, it will proceed.
        agent = TranslatorAgent(inference=mock_inference_fixture)
        mock_inference_fixture.query.return_value = "Translated empty (should not happen if not called)"

        translated = agent._translate_text("", "en", "es")
        # The _extract_code_blocks on "" is "", prompt will be for "", LLM might translate it.
        # If LLM returns "Translated empty", that's what we get.
        # However, the prompt construction might be weird for empty text.
        # The crucial part is that `_translate_text` doesn't have a dedicated "is empty" check,
        # it relies on the caller or processes the empty string.
        # Given the prompt structure, an empty input text would still generate a full prompt.
        # Let's assume LLM translates "" to "" for simplicity here or returns what's mocked.
        assert translated == "Translated empty (should not happen if not called)"
        mock_inference_fixture.query.assert_called_once()


    def test_internal_translate_text_preserves_code_blocks_during_translation_fallback(self, mock_inference_fixture):
        agent = TranslatorAgent(inference=mock_inference_fixture)
        # Original text with code mixed in
        original_text_with_code = "Texto before ```python\nprint('hola')\n``` Texto after"

        # Expected processed text and code map (as _extract_code_blocks would produce)
        # Manually deriving what _extract_code_blocks would do for this specific input:
        placeholder = f"{agent.code_block_placeholder_prefix}_0__"
        processed_text_expected = f"Texto before {placeholder} Texto after"
        # code_map_expected = {placeholder: "```python\nprint('hola')\n```"} # Not needed for call
                                                                            # as _translate_text does extraction.

        # LLM fails or returns non-string, causing fallback
        mock_inference_fixture.query.return_value = None

        # Call _translate_text. It will extract code, try to translate `processed_text_expected`,
        # fail (fallback to `processed_text_expected`), then reinsert code.
        translated_fallback = agent._translate_text(original_text_with_code, "en", "es")

        # The result should be the original text because translation failed and code was reinserted
        assert translated_fallback == original_text_with_code
        mock_inference_fixture.query.assert_called_once()
        args, _ = mock_inference_fixture.query.call_args
        # Prompt should contain the processed text with placeholders
        assert placeholder in args[0]
        assert "print('hola')" not in args[0] # Original code content should not be in prompt to LLM

    @patch('fedotllm.agents.translator.detect')
    def test_translate_input_to_english_handles_llm_non_string_responses(self, mock_detect, mock_inference_fixture):
        mock_detect.return_value = 'es'
        agent = TranslatorAgent(inference=mock_inference_fixture)
        original_input = "Hola obj"

        # Mock LLM to return a non-string (e.g., an object or list)
        mock_response_obj = MagicMock()
        mock_response_obj.content = "Translated from object (but won't be used)"
        mock_inference_fixture.query.return_value = mock_response_obj

        # The agent should fallback to the original text if LLM response is not a string
        assert agent.translate_input_to_english(original_input) == original_input

        # Also test with a list response
        mock_inference_fixture.query.return_value = ["Lista", "de", "strings"]
        assert agent.translate_input_to_english("Hola list str") == "Hola list str"
