from typing import Any
from src.ExecutingStage import ExecutingStage
from src.Validator import PromptSchema, FunctionDefinitionSchema
from src.Exceptions import ParsingError
from pathlib import Path
from pydantic import ValidationError


class JSONParser(ExecutingStage):
    """Parse user input and extracted JSON arguments."""

    def __init__(self) -> None:
        """
        Initialize the JSON parser with empty input data.

        Returns:
            None
        """
        self.__prompts: list[PromptSchema] = list()
        self.__functions_definition: list[FunctionDefinitionSchema] = list()

    def __create_prompts_models(
        self,
        prompts: list[dict[str, Any]],
    ) -> None:
        """
        Create and validate prompt models from input data.

        Args:
            prompts (list[dict[str, Any]]): Prompt data to validate and convert
                into prompt models.

        Returns:
            None
        """
        if isinstance(prompts, list):
            self.__prompts = [
                PromptSchema(prompt=prompt)
                for prompt in prompts
            ]
        elif isinstance(prompts, dict):
            self.__prompts.append(PromptSchema(prompt=prompts))
        else:
            raise ValueError("Error: prompts data must be list o dict.")

    def __create_functions_definition_models(
        self,
        functions_definition: list[dict[str, Any]],
    ) -> None:
        """
        Create and validate function definition models.

        Args:
            functions_definition (list[dict[str, Any]]):
                Function definition data to convert into validated models.

        Returns:
            None
        """
        if isinstance(functions_definition, list):
            for func_def in functions_definition:
                self.__functions_definition.append(
                    FunctionDefinitionSchema(
                        **func_def
                    )
                )
        elif isinstance(functions_definition, dict):
            self.__functions_definition.append(
                FunctionDefinitionSchema(
                    **functions_definition
                )
            )

        else:
            raise ValueError(
                "Error: FunctionsDefinition schema must list or dictionary"
            )

    def execute(self, data: Any) -> Any:
        """
        Parse input data and prepare it for the next pipeline stage.

        Args:
            data (Any): Input data containing prompts, function definitions,
                and output path.

        Returns:
            Any: Updated data containing validated models and prepared paths.
        """
        try:
            self.__create_prompts_models(data['prompts'])
            self.__create_functions_definition_models(
                data['functions_definition']
            )
            self.__create_folders_path(data['output_path'])
        except (ValidationError) as e:
            raise ParsingError(e.errors()[0].get("msg", ''))
        data.update(
            {
                'functions_definition': self.__functions_definition,
                'prompts': self.__prompts,
                'output_path': self.__path,
            }
        )

        return data

    def __create_folders_path(self, output_path: str) -> None:
        """
        Create the output file path and required parent directories.

        Args:
            output_path (str): Path to the output file.

        Returns:
            None
        """
        self.__path = Path(output_path)
        dirs = self.__path.parent
        dirs.mkdir(parents=True, exist_ok=True)
        self.__path.touch()


# if __name__ == "__main__":
#     parser = JSONParser()
#     parser.execute({
#         'functions_definition': {
#             "name": "fn_get_square_root",
#             "description": "Calculate the square root of a number.",
#             "parameters": {
#                 "a": {
#                     "type": "number"
#                 }
#             },
#             "returns": {
#                   "type": "number"
#             }
#           },
#         'prompts': {
#             'prompt': "hello"
#         }
#     })
