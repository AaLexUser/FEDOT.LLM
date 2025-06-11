# tests/unit/agents/test_translator.py
import pytest
from unittest.mock import patch, MagicMock

from fedotllm.agents.translator import TranslatorAgent
from langdetect import LangDetectException

# Mock LANGUAGES from googletrans
MOCK_LANGUAGES = {
    'en': 'english',
    'es': 'spanish',
    'fr': 'french',
    # Add other languages as needed for tests
}

@patch('fedotllm.agents.translator.LANGUAGES', MOCK_LANGUAGES)
class TestTranslatorAgent:

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_success(self, mock_detect):
        mock_detect.return_value = 'es'
        agent = TranslatorAgent()
        agent.translate_input_to_english("Hola mundo")
        assert agent.source_language == 'es'
        mock_detect.assert_called_once_with("Hola mundo")

    @patch('fedotllm.agents.translator.detect')
    def test_language_detection_failure_defaults_to_english(self, mock_detect):
        mock_detect.side_effect = LangDetectException(0, "Detection failed")
        agent = TranslatorAgent()
        agent.translate_input_to_english("Invalid text for detection")
        assert agent.source_language == 'en'
        mock_detect.assert_called_once_with("Invalid text for detection")

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_translation_to_english_success(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'es'
        mock_translator_instance = MockTranslator.return_value
        mock_translator_instance.translate.return_value = MagicMock(text="Hello world")

        agent = TranslatorAgent()
        translated_text = agent.translate_input_to_english("Hola mundo")

        assert translated_text == "Hello world"
        assert agent.source_language == 'es'
        mock_translator_instance.translate.assert_called_once_with("Hola mundo", dest='en')

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_translation_to_english_error_returns_original(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'es'
        mock_translator_instance = MockTranslator.return_value
        mock_translator_instance.translate.side_effect = Exception("Translation API error")

        agent = TranslatorAgent()
        translated_text = agent.translate_input_to_english("Hola mundo")

        assert translated_text == "Hola mundo" # Returns original on error
        assert agent.source_language == 'es'

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_english_input_not_translated(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'en'
        mock_translator_instance = MockTranslator.return_value

        agent = TranslatorAgent()
        input_text = "Hello world, this is English."
        translated_text = agent.translate_input_to_english(input_text)

        assert translated_text == input_text
        assert agent.source_language == 'en'
        mock_translator_instance.translate.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_translation_to_source_language_success(self, MockTranslator, mock_detect):
        # First, simulate input translation to set source_language
        mock_detect.return_value = 'es'
        mock_translator_input = MockTranslator.return_value
        # Clear any previous call counts on the mock for this specific test flow
        mock_translator_input.translate.reset_mock()
        mock_translator_input.translate.return_value = MagicMock(text="Hello world") # Mock input translation

        agent = TranslatorAgent()
        agent.translate_input_to_english("Hola mundo") # This sets source_language to 'es'
        assert agent.source_language == 'es'
        mock_translator_input.translate.assert_called_once_with("Hola mundo", dest='en')


        # Now, test output translation
        # Ensure the mock is configured for the output translation call
        mock_translator_output = MockTranslator.return_value
        mock_translator_output.translate.return_value = MagicMock(text="Hola mundo otra vez") # Mock output translation

        translated_output = agent.translate_output_to_source_language("Hello world again")

        assert translated_output == "Hola mundo otra vez"
        # Ensure translate was called for output with correct destination
        mock_translator_output.translate.assert_called_with("Hello world again", dest='es')


    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_translation_to_source_language_error_returns_original(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'es'
        agent = TranslatorAgent()
        # Reset mock from any potential previous calls in other tests if instance is somehow shared (though it shouldn't be)
        MockTranslator.return_value.translate.reset_mock()
        # Simulate input translation call
        MockTranslator.return_value.translate.return_value = MagicMock(text="Hello world")
        agent.translate_input_to_english("Hola mundo") # Sets source_language to 'es'
        assert agent.source_language == 'es'

        # Configure mock for the output translation call to raise an error
        mock_translator_instance = MockTranslator.return_value
        mock_translator_instance.translate.side_effect = Exception("Translation API error")

        translated_output = agent.translate_output_to_source_language("Hello world again")
        assert translated_output == "Hello world again" # Returns original on error

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_unsupported_language_for_input_translation(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'xx' # Unsupported language
        agent = TranslatorAgent()
        input_text = "Text in unsupported language"
        translated_text = agent.translate_input_to_english(input_text)

        assert translated_text == input_text # Should return original
        assert agent.source_language == 'xx'
        MockTranslator.return_value.translate.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_unsupported_language_for_output_translation(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'xx' # Unsupported language
        agent = TranslatorAgent()
        # Simulate input process
        agent.translate_input_to_english("Text in unsupported language") # Sets source_language to 'xx'
        assert agent.source_language == 'xx'
        MockTranslator.return_value.translate.reset_mock() # Reset mock calls from input processing

        english_text = "This is the English response"
        translated_output = agent.translate_output_to_source_language(english_text)

        assert translated_output == english_text # Should return original English text
        MockTranslator.return_value.translate.assert_not_called()

    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_markdown_preservation_basic(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'es'
        mock_translator_instance = MockTranslator.return_value

        original_md = "# Title\nSome *bold* text."
        translated_md_mock = "# Título\nUn texto *en negrita*."
        # Input translation
        mock_translator_instance.translate.return_value = MagicMock(text=translated_md_mock)

        agent = TranslatorAgent()
        translated_text = agent.translate_input_to_english(original_md)
        assert translated_text == translated_md_mock
        mock_translator_instance.translate.assert_called_once_with(original_md, dest='en')

        # Simulate output translation
        agent.source_language = 'es' # Ensure source language is set for output
        mock_translator_instance.translate.return_value = MagicMock(text=original_md) # Mocking translation back
        output_text = agent.translate_output_to_source_language(translated_md_mock)
        assert output_text == original_md
        mock_translator_instance.translate.assert_called_with(translated_md_mock, dest='es')


    @patch('fedotllm.agents.translator.detect')
    @patch('fedotllm.agents.translator.Translator')
    def test_code_block_preservation_basic(self, MockTranslator, mock_detect):
        mock_detect.return_value = 'es'
        mock_translator_instance = MockTranslator.return_value

        original_code = "```python\nprint('Hello')\n```"
        translated_code_mock = "```python\nprint('Hola')\n```"
        # Input translation
        mock_translator_instance.translate.return_value = MagicMock(text=translated_code_mock)

        agent = TranslatorAgent()
        translated_text = agent.translate_input_to_english(original_code)
        assert translated_text == translated_code_mock
        mock_translator_instance.translate.assert_called_once_with(original_code, dest='en')

        # Simulate output translation
        agent.source_language = 'es'
        mock_translator_instance.translate.return_value = MagicMock(text=original_code) # Mocking translation back
        output_text = agent.translate_output_to_source_language(translated_code_mock)
        assert output_text == original_code
        mock_translator_instance.translate.assert_called_with(translated_code_mock, dest='es')
