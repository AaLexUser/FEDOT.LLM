# fedotllm/agents/translator.py
from langdetect import detect, LangDetectException
from googletrans import Translator, LANGUAGES

class TranslatorAgent:
    def __init__(self):
        self.source_language = None
        self.translator = Translator()

    def translate_input_to_english(self, message: str) -> str:
        try:
            self.source_language = detect(message)
        except LangDetectException:
            self.source_language = 'en' # Default to English if detection fails
            print("Language detection failed, defaulting to English.")

        if self.source_language != 'en':
            print(f"Detected language: {self.source_language}. Translating to English.")
            try:
                # Check if the detected language is supported by Google Translate
                if self.source_language not in LANGUAGES:
                    print(f"Language {self.source_language} is not supported for translation. Returning original message.")
                    return message
                translated_message = self.translator.translate(message, dest='en').text
                return translated_message
            except Exception as e:
                print(f"Error during translation to English: {e}")
                return message # Return original message in case of error
        else:
            print("Detected language: English. No translation needed.")
            return message

    def translate_output_to_source_language(self, message: str) -> str:
        if self.source_language and self.source_language != 'en':
            print(f"Translating back to {self.source_language}.")
            try:
                # Check if the source language is supported by Google Translate
                if self.source_language not in LANGUAGES:
                    print(f"Language {self.source_language} is not supported for translation. Returning original message.")
                    return message
                translated_message = self.translator.translate(message, dest=self.source_language).text
                return translated_message
            except Exception as e:
                print(f"Error during translation to {self.source_language}: {e}")
                return message # Return original message in case of error
        return message
