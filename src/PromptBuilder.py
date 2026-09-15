from typing import Any


class PromptBuilder:
    """
    Build prompts containing available functions and generation instructions.
    """
    def set_available_functions(self, functions_definition: list[Any]) -> None:
        """
        Set available functions for prompt generation.

        Args:
            functions_definition (list[Any]): List of function definitions.

        Returns:
            None
        """
        self.__available_functions: str = ''
        for function in functions_definition:
            self.__available_functions += function.model_dump_json() + '\n'

    def __call__(self) -> str:
        """
        Generate the formatted prompt string.

        Returns:
            str: Prompt containing available functions and instructions.
        """
        functions_block = self.__available_functions

        clean_prompt = f"""
        You are a strict function_calling AI assistant.

        AVAILABLE FUNCTIONS:
            {functions_block}

        INSTRUCTIONS:
            choose one matching function
        RULES:
            - output ONLY valid JSON. NO EXTRA TEXT.
            - NO EXPLANATION. NO MARKDOWN.
            - never invent functions that are not listed above
            - extract values from the user prompt accurately
        EXAMPLE OUTPUT:
            {{
                "prompt": (
                    "Replace all vowels in 'Programming is fun' "
                    "with asterisks"
                ),
                "name": "fn_substitute_string_with_regex",
                "parameters": {{
                    "source_string": "Programming is fun",
                    "regex": "([aeiouAEIOU]+)",
                    "replacement": "*"
                }}
            }}
        """
        return clean_prompt
