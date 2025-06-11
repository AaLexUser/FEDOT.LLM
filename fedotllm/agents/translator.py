import re # Import re for regex operations
from langdetect import detect, LangDetectException
from fedotllm.llm import AIInference

class TranslatorAgent:
    def __init__(self, inference: AIInference):
        self.source_language = None
        self.inference = inference
        self.code_block_placeholder_prefix = "__CODE_BLOCK_AGENT_JULES__"

    def _extract_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        code_blocks_map = {}
        # Regex to find code blocks (both fenced and with language specifier)
        # It captures the opening fence, language (optional), content, and closing fence.
        # Using re.DOTALL so '.' matches newlines within the code block.
        # Non-greedy match for content (.*?) is important.
        # Corrected pattern to capture ``` with optional language, content, and closing ```
        pattern = re.compile(r"(```(?:[a-zA-Z0-9_.-]+)?\n)(.*?)(\n```)", re.DOTALL)

        idx = 0
        def replace_match(match):
            nonlocal idx
            placeholder = f"{self.code_block_placeholder_prefix}_{idx}__"
            # Store the full matched block (including fences and language specifier)
            code_blocks_map[placeholder] = match.group(0)
            idx += 1
            return placeholder

        processed_text = pattern.sub(replace_match, text)
        return processed_text, code_blocks_map

    def _reinsert_code_blocks(self, text: str, code_blocks_map: dict[str, str]) -> str:
        for placeholder, original_code_block in code_blocks_map.items():
            # Escape placeholder for regex to avoid issues if it contains special characters
            escaped_placeholder = re.escape(placeholder)
            # Use a lambda with re.sub to replace only the first occurrence of the placeholder
            # This is safer if somehow a placeholder text appears in the translated content by chance
            text = re.sub(escaped_placeholder, lambda m: original_code_block, text, count=1)
        return text

    def _translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        processed_text, code_blocks_map = self._extract_code_blocks(text)

        prompt_source_lang = source_language
        if source_language == "auto" or source_language is None:
            prompt_source_lang = "the auto-detected source language"

        prompt = (
            f"Translate the following text from {prompt_source_lang} to {target_language}. "
            f"It is absolutely crucial to preserve the original formatting exactly. "
            f"This includes all markdown syntax: headers (e.g., #, ##), lists (e.g., -, *, 1.), "
            f"bold (e.g., **text** or __text__), italics (e.g., *text* or _text_), "
            f"strikethrough (e.g., ~~text~~), links (e.g., [text](url)), images (e.g., ![alt](url)), "
            f"tables (using pipe and hyphen syntax), and blockquotes (e.g., > text). "
            f"The text provided may contain placeholders like '{self.code_block_placeholder_prefix}_NUMBER__' "
            f"(e.g., {self.code_block_placeholder_prefix}_0__, {self.code_block_placeholder_prefix}_1__). "
            f"These placeholders represent original code blocks and MUST NOT be translated or altered in any way. "
            f"They must be preserved exactly as they appear in the input text. "
            f"Only translate the surrounding text. "
            f"If the text (excluding placeholders) is already in {target_language} and requires no translation, "
            f"return it as is, ensuring placeholders are also returned as is.\n\n"
            f"Text to translate (placeholders like {self.code_block_placeholder_prefix}_0__ must be kept as is):\n{processed_text}"
        )

        try:
            # Assuming AIInference().create() expects a list of messages
            response = self.inference.create(messages=[{"role": "user", "content": prompt}])

            translated_text_with_placeholders = ""
            if isinstance(response, str):
                translated_text_with_placeholders = response
            elif hasattr(response, 'text') and isinstance(response.text, str):
                translated_text_with_placeholders = response.text
            elif hasattr(response, 'content') and isinstance(response.content, str):
                translated_text_with_placeholders = response.content
            elif isinstance(response, dict) and 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if 'text' in choice and choice['text']: # Older OpenAI completion style
                    translated_text_with_placeholders = choice['text']
                elif 'message' in choice and 'content' in choice['message'] and choice['message']['content']: # Chat completion style
                    translated_text_with_placeholders = choice['message']['content']
                else:
                    print(f"Translation response choice had unexpected format: {choice}. Using processed text.")
                    translated_text_with_placeholders = processed_text
            else:
                print(f"Translation response had unexpected format: {type(response)}. Using processed text.")
                translated_text_with_placeholders = processed_text # Fallback to text with placeholders

        except Exception as e:
            print(f"Error during translation using AIInference: {e}")
            translated_text_with_placeholders = processed_text # Fallback in case of error

        final_translated_text = self._reinsert_code_blocks(translated_text_with_placeholders, code_blocks_map)
        return final_translated_text

    def translate_input_to_english(self, message: str) -> str:
        try:
            self.source_language = detect(message)
        except LangDetectException:
            self.source_language = 'en'
            print("Language detection failed, defaulting to English.")

        if self.source_language != 'en':
            print(f"Detected language: {self.source_language}. Translating to English using AIInference with improved formatting preservation.")
            return self._translate_text(message, target_language='en', source_language=self.source_language)
        else:
            print("Detected language: English. No translation needed.")
            return message

    def translate_output_to_source_language(self, message: str) -> str:
        if self.source_language and self.source_language != 'en':
            print(f"Translating back to {self.source_language} using AIInference with improved formatting preservation.")
            return self._translate_text(message, target_language=self.source_language, source_language='en')
        return message
