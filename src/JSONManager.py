from src.ExecutingStage import ExecutingStage
from src.JSONReader import JSONReader
from src.JSONParser import JSONParser
from src.JSONGenerator import JSONGenerator
from src.JSONWriter import JSONWriter
from typing import Any


class JSONManager:
    """JSONManager class manage pipeline to generate structured json
    """

    def __init__(self) -> None:
        """
        Initialize the JSON manager with pipeline execution stages.

        Returns:
            None
        """
        self.__data: dict[str, Any] = dict()
        self.__stages: list[ExecutingStage] = [
            JSONReader(),
            JSONParser(),
            JSONGenerator(),
            JSONWriter(),
        ]

    def set_data_input(self, data: dict[str, Any]) -> None:
        """
        Set the input data for the JSON pipeline.

        Args:
            data (dict[str, Any]): Input data to process.

        Returns:
            None
        """
        self.__data = data

    def get_stages(self) -> list[ExecutingStage]:
        """
        Get all stages in the pipeline.

        Returns:
            list[ExecutingStage]: List of pipeline execution stages.
        """
        return self.__stages

    def add_stage(self, stage: ExecutingStage) -> None:
        """
        Add a new execution stage to the pipeline.

        Args:
            stage (ExecutingStage): Pipeline stage to add.

        Returns:
            None
        """
        self.__stages.append(stage)

    def generate_json_file(self) -> None:
        """
        Run the full pipeline to generate the JSON file.

        Returns:
            None
        """
        for stage in self.__stages:
            self.__data = stage.execute(self.__data)


# if __name__ == '__main__':
#     manager = JSONManager()
#     print(manager.create_folder_output())
