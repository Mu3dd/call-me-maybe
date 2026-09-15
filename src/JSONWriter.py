from typing import Any
import json
from src.ExecutingStage import ExecutingStage


class JSONWriter(ExecutingStage):
    """ JSONWriter class write a json to a specific file from user"""

    def execute(self, data: Any) -> Any:
        """
        Serialize generated JSON results and write them to the output file.

        Args:
            data (Any): Input data containing JSON results and output path.

        Returns:
            Any: Updated pipeline data.
        """
        list_of_json = data.get('json_results', [])
        with open(data['output_path'], "w") as file:

            if len(list_of_json) > 1:
                file.write("[\n")

            for index, output_json in enumerate(list_of_json):
                try:
                    data_object = json.loads(output_json)
                    json.dump(data_object, file, indent=4)
                    if index < len(list_of_json) - 1:
                        file.write(",\n")
                    else:
                        file.write("\n")
                except Exception:
                    print(f"Warning: {output_json} is Invalid.")
            if len(list_of_json) > 1:
                file.write("]")

# if __name__ == '__main__':
#     writer = JSONWriter("dasa/output/file.txt")
#     writer.execute([])
