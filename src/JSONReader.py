from typing import Any
import json
from src.ExecutingStage import ExecutingStage
from src.Exceptions import ReadingError


class JSONReader(ExecutingStage):
    """Class JSONReader reads and stores json files."""

    def __init__(self) -> None:
        """
        Initialize the JSON reader with empty data containers.

        Returns:
            None
        """
        self.__prompts: list[dict[str, str]] = list()
        self.__functions_definition: list[dict[str, Any]] = list()

    # getters
    def get_prompts(self) -> list[dict[str, str]]:
        """
        Get all prompts.

        Returns:
            list[dict[str, str]]: List of stored prompts.
        """
        return self.__prompts

    def get_functions_definition(self) -> list[dict[str, Any]]:
        """
        Get function definition schemas.

        Returns:
            list[dict[str, Any]]: List of function definition schemas.
        """
        return self.__functions_definition

    # for pipeline execution
    def execute(self, data: Any) -> Any:
        """
        Read JSON files and prepare data for the next pipeline stage.

        Args:
            data (Any): Input data containing JSON file paths.

        Returns:
            Any: Updated data containing loaded prompts
                and function definitions.
        """
        self.__read_functions_definition(
            data.get('functions_definition_path', '')
        )
        self.__read_prompts(data.get('prompts_path', ''))

        data.update(
            {
                'functions_definition': self.__functions_definition,
                'prompts': self.__prompts,
            }
        )
        return data

    # read input files
    def __read_functions_definition(self, path: str) -> None:
        """
        Read function definitions from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            None
        """
        self.__read_file(path, 'functions_definition')

    def __read_prompts(self, path: str) -> None:
        """
        Read prompts from a JSON file.

        Args:
            path (str): Path to the JSON file.

        Returns:
            None
        """
        self.__read_file(path, 'prompts')

    def __read_file(self, filename: str, key: str) -> None:
        """
        Read a JSON file and store its content by key.

        Args:
            filename (str): Path to the JSON file.
            key (str): Data key indicating the type of content to store.

        Returns:
            None
        """
        try:
            with open(filename, "r") as file:
                data = json.load(file)
            if key == 'prompts':
                self.__prompts = data
            else:
                self.__functions_definition = data
        except IsADirectoryError:
            raise ReadingError(f"Error: Cannot read {key} is a directory!")
        except PermissionError:
            raise ReadingError(f"Error: Cannot read {key} not permitted!")
        except FileNotFoundError:
            raise ReadingError(f"Error: Cannot read {key} file not found!")
        except json.JSONDecodeError:
            raise ReadingError(f"Error: Failed to parse json from '{key}'!")
