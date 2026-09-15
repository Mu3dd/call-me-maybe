from typing import Any
from src.ExecutingStage import ExecutingStage
from src.PromptBuilder import PromptBuilder
from src.Enums import JSONState
from llm_sdk import Small_LLM_Model
import numpy as np
from numpy.typing import NDArray
from src.FiniteStateMachine import FiniteStateMachine
from src.FilterDecoder import FilterDecoder
from src.Enums import JSONStatic, ParameterState
from src.Visualizer import Visualizer
import json
import time


class JSONGenerator(ExecutingStage):
    """JSONGenerator generate json by llm with constrained decoding
    """
    def __init__(self) -> None:
        """
        Initialize the JSON generator and its internal components.

        This constructor initializes the prompt builder, finite state machine,
        decoder filter, and internal state used during JSON generation.
        """
        super().__init__()
        self.__json_results: list[str] = list()

        self.__prompt_builder = PromptBuilder()

        self.__fsm = FiniteStateMachine()
        self.__filter_decoder = FilterDecoder()
        self.__ids_for_strings: list[int] = []
        self.__ids_for_numbers: list[int] = []

        self.__index_key_param: int = 0
        self.__current_function_name: str = ''
        self.__dynamic_generated: str = ''
        self.__dynamic_ids: list[int] = []
        self.__json_result: str = ''

    def __prepare_function_names(self) -> None:
        """
        Prepare function names and their token IDs for constrained decoding.

        Returns:
            None
        """
        self.__function_names: list[str] = list(
            function.name + "\", " for function in self.__functions_definition
        )
        self.__ids_function_names_set: set[int] = set()
        for name in self.__function_names:
            function_ids = self.__model.encode(name).tolist()[0]
            for token_id in function_ids:
                self.__ids_function_names_set.add(token_id)

    def __load_vocabulary(self) -> None:
        """
        Load the model vocabulary required for constrained decoding.

        Returns:
            None
        """
        vocab_path = self.__model.get_path_to_vocab_file()
        with open(vocab_path) as file:
            self.__vocabulary = json.load(file)

    def __init_ids_for_parameters(self) -> None:
        """
        Initialize token IDs used for parameter
            value masking during decoding.

        Args:
            None
        Returns:
            None
        """
        for token_id in self.__vocabulary.values():
            decoded_id = self.__model.decode(token_id)
            if (
                decoded_id.isascii()
                and ',' not in decoded_id
                and '}' not in decoded_id
            ):
                self.__ids_for_strings.append(token_id)
            if (
                decoded_id.isdigit()
                or decoded_id == '-'
                or ((decoded_id in '."') and ',' not in decoded_id)
                and '}' not in decoded_id
            ):
                self.__ids_for_numbers.append(token_id)

    def __prepare_data(self, data: Any) -> None:
        """
        Prepare input data and initialize components
        required for JSON generation.

        Args:
            data (Any): Input data containing function definitions and prompts
                required for JSON generation.
        Returns:
            None
        """
        self.__functions_definition = data['functions_definition']
        self.__prompts = data['prompts']
        self.__prepare_function_names()

        self.__load_vocabulary()
        self.__prompt_builder.set_available_functions(
            self.__functions_definition
        )

        tokens_before_prompt = self.__model.encode(
            JSONStatic.STR_BEFORE_PROMPT.value
        ).tolist()[0]
        tokens_before_name = self.__model.encode(
            JSONStatic.STR_BEFORE_NAME.value
        ).tolist()[0]
        tokens_before_parameters = self.__model.encode(
            JSONStatic.STR_BEFORE_PARAMETERS.value
        ).tolist()[0]

        self.__filter_decoder.set_tokens_before_prompt(
            tokens_before_prompt
        )
        self.__filter_decoder.set_tokens_before_name(tokens_before_name)
        self.__filter_decoder.set_tokens_before_parameters(
            tokens_before_parameters
        )

        self.__init_ids_for_parameters()

        self.__prefix_ids: list[int] = self.__model.encode(
            self.__prompt_builder()
        ).tolist()[0]

    # generating
    def __generate_tokens_before_prompt(self) -> None:
        """
        Generate tokens for the before-prompt state.

        Returns:
            None
        """
        self.__ids_current_prompt += (
            self.__filter_decoder.get_tokens_before_prompt()
        )
        self.__json_result += self.__fsm.get_static_json()
        self.__fsm.set_static_json(JSONStatic.STR_BEFORE_NAME)
        self.__fsm.set_state(JSONState.IN_PROMPT)

    def __generate_tokens_in_prompt(self) -> None:
        """
        Generate tokens for the in-prompt state.

        Returns:
            None
        """
        self.__ids_current_prompt += self.__model.encode(
            json.dumps(self.__current_prompt) + ', '
        ).tolist()[0]
        self.__json_result += (
            json.dumps(self.__current_prompt) + ', '
        )
        self.__fsm.set_state(JSONState.BEFORE_NAME)

    def __generate_tokens_before_name(self) -> None:
        """
        Generate tokens for the before-name state.

        Returns:
            None
        """
        self.__ids_current_prompt += (
            self.__filter_decoder.get_tokens_before_name()
        )
        self.__json_result += self.__fsm.get_static_json()
        self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PARAMETERS)
        self.__fsm.set_state(JSONState.IN_NAME)

    def __is_at_least_one_function(self, generated: str) -> bool:
        """
        Check whether a generated string matches a function name prefix.

        Args:
            generated (str): Generated text to compare with function names.

        Returns:
            bool: True if at least one function name starts with the generated
                string, otherwise False.
        """
        for function_name in self.__function_names:
            if function_name.startswith(generated):
                return True
        return False

    def __add_tokens_of_function_name(self) -> None:
        """
        Add generated tokens for the selected function name.

        Returns:
            None
        """
        self.__dynamic_generated = self.__get_only_available_function(
            self.__dynamic_generated
        )
        self.__dynamic_generated = self.__dynamic_generated
        self.__json_result += self.__dynamic_generated
        self.__current_function_name = self.__dynamic_generated
        self.__dynamic_ids = self.__model.encode(
            self.__dynamic_generated
        ).tolist()[0]
        self.__ids_current_prompt += self.__dynamic_ids
        self.__dynamic_ids = []
        self.__dynamic_generated = ''

    def __generate_tokens_in_name(self) -> None:
        """
        Generate tokens for the in-name state.

        Returns:
            None
        """
        full_ids = self.__ids_current_prompt + self.__dynamic_ids
        logits = self.__model.get_logits_from_input_ids(full_ids)
        masked_logits = self.__mask_logits_by_type(
            logits,
            'function_name',
        )

        index_max_logit = np.argmax(masked_logits)
        self.__dynamic_ids.append(int(index_max_logit))
        self.__dynamic_generated += self.__model.decode(
            [int(index_max_logit)]
        )

        if self.__filter_decoder.is_found_only_one_function(
            self.__dynamic_generated,
            self.__function_names,
        ):
            self.__add_tokens_of_function_name()
            self.__fsm.set_state(JSONState.BEFORE_PARAMETERS)

    def __generate_tokens_before_parameters(self) -> None:
        """
        Generate tokens for the before-parameters state.

        Returns:
            None
        """
        self.__ids_current_prompt += (
            self.__filter_decoder.get_tokens_before_parameters()
        )
        self.__json_result += self.__fsm.get_static_json()
        self.__fsm.set_state(JSONState.IN_PARAMETERS)
        self.__dynamic_generated = ""
        self.__dynamic_ids = []

    def __generate_tokens_in_key_parameters(
        self,
        parameters: dict[str, dict[str, str]],
    ) -> None:
        """
        Generate tokens for a parameter key.

        Args:
            parameters (dict[str, dict[str, str]]): Function parameters
                definitions.

        Returns:
            None
        """
        for index, key_name in enumerate(parameters.keys()):
            if index == self.__index_key_param:
                break

        self.__json_result += f'"{key_name}": '
        self.__ids_current_prompt += self.__model.encode(
            f'"{key_name}": '
        ).tolist()[0]
        self.__fsm.set_parameters_state(ParameterState.IN_VALUE)

    def __mask_logits_by_type(
        self,
        logits: list[float],
        type_mask: str,
    ) -> NDArray[np.float32]:
        """
        Apply a mask to logits based on the specified token type.

        Args:
            logits (list[float]): Model output logits for each token.
            type_mask (str): Type of token mask to apply.

        Returns:
            NDArray[np.float32]: Masked logits array.
        """
        if type_mask == '':
            return np.array(logits, dtype=np.float32)

        masked_logits = np.full(len(logits), -np.inf)

        if type_mask in {'number', 'float', 'integer'}:
            for allowed_id in self.__ids_for_numbers:
                masked_logits[allowed_id] = logits[allowed_id]
            return masked_logits

        elif type_mask == 'function_name':
            for index_id in self.__ids_function_names_set:
                if self.__is_at_least_one_function(
                    self.__dynamic_generated + self.__model.decode([index_id])
                ):
                    masked_logits[index_id] = logits[index_id]
            return masked_logits

        return np.array(logits, dtype=np.float32)

    def __add_dynamic_value_by_type(
        self,
        values_param: list[dict[str, str]],
    ) -> None:
        """
        Add a generated parameter value according to its type.

        Args:
            values_param (list[dict[str, str]]): List of parameter definitions.

        Returns:
            None
        """
        dict_schema = values_param[self.__index_key_param - 1]
        type_param = dict_schema.get('type', '')

        stripped_number = self.__dynamic_generated.rstrip().strip('"')
        if type_param == 'number' and stripped_number.find('.') == -1:
            stripped_number += ".0"

        self.__json_result += stripped_number
        self.__ids_current_prompt += self.__model.encode(
            stripped_number
        ).tolist()[0]

    def __append_escaped_string_value(self, raw_value: str) -> None:
        """
        Append an escaped string value to the generated JSON result.

        Args:
            raw_value (str): Raw string value to escape and append.
        Returns:
            None
        """
        escaped_value = json.dumps(raw_value)
        self.__json_result += escaped_value
        self.__ids_current_prompt += self.__model.encode(
            escaped_value
        ).tolist()[0]

    def __generate_tokens_in_value_parameters(
        self,
        parameters: dict[str, dict[str, str]],
    ) -> None:
        """
        Generate tokens for a parameter value.

        Args:
            parameters (dict[str, dict[str, str]]): Function parameter
                definitions.

        Returns:
            None
        """
        dict_schema = list(parameters.values())[self.__index_key_param - 1]
        type_param = dict_schema.get('type', '')

        self.__dynamic_ids = self.__model.encode("\"").tolist()[0]
        self.__dynamic_generated = "\""
        generated_value = ''
        max_generated_chars = self.__len_current_prompt
        while True:
            logits = self.__model.get_logits_from_input_ids(
                self.__ids_current_prompt + self.__dynamic_ids
            )
            masked_logits = self.__mask_logits_by_type(
                logits,
                type_param,
            )
            index_max_logit = np.argmax(masked_logits)
            decoded_token = self.__model.decode([int(index_max_logit)])

            self.__dynamic_generated += decoded_token
            self.__dynamic_ids.append(int(index_max_logit))
            for char in decoded_token:
                if char == '"':
                    break
                generated_value += char
            else:
                if len(generated_value) < max_generated_chars:
                    continue

            break

        self.__append_new_parameter(parameters, generated_value, type_param)
        self.__change_state_in_parameters(parameters)

    def __append_new_parameter(
        self,
        parameters: dict[str, dict[str, str]],
        generated_value: str, type_param: str
    ) -> None:
        """
        Append a generated parameter value based on its type.

        Args:
            parameters (dict[str, dict[str, str]]): Function parameter
                definitions.
            generated_value (str): Generated parameter value.
            type_param (str): Parameter data type.

        Returns:
            None
        """
        if type_param in {'number', 'integer', 'float'}:
            self.__add_dynamic_value_by_type(list(parameters.values()))
        else:
            self.__append_escaped_string_value(generated_value)

    def __change_state_in_parameters(
        self,
        parameters: dict[str, dict[str, str]]
    ) -> None:
        """
        Update the state after generating a parameter value.

        Args:
            parameters (dict[str, dict[str, str]]): Function parameter
                definitions.

        Returns:
            None
        """
        if self.__filter_decoder.is_closed_brackets(self.__json_result):
            self.__fsm.set_state(JSONState.IN_END)

        elif self.__index_key_param == len(parameters):
            self.__fsm.set_parameters_state(ParameterState.IN_CLOSE)

        elif self.__index_key_param < len(parameters):
            self.__json_result += ","
            self.__ids_current_prompt += self.__model.encode(',').tolist()[0]
            self.__fsm.set_parameters_state(ParameterState.IN_KEY)

    def __generate_tokens_in_close_parameters(self) -> None:
        """
        Generate closing tokens for the JSON object.

        Returns:
            None
        """
        bracket_id = self.__model.encode("}}")
        self.__json_result += "}}"
        self.__ids_current_prompt.append(int(bracket_id))
        self.__fsm.set_state(JSONState.IN_END)

    def __generate_tokens_in_parameters(self) -> None:
        """
        Generate closing tokens for the JSON object.

        Returns:
            None
        """
        function_parameters = self.__get_parameters_function(
            self.__current_function_name[:-3]
        )

        if self.__fsm.get_parameters_state() == ParameterState.IN_KEY:
            self.__generate_tokens_in_key_parameters(function_parameters)
            self.__index_key_param += 1

        elif self.__fsm.get_parameters_state() == ParameterState.IN_VALUE:
            self.__generate_tokens_in_value_parameters(function_parameters)

        elif self.__fsm.get_parameters_state() == ParameterState.IN_CLOSE:
            self.__generate_tokens_in_close_parameters()
        return None

    def __generate(self) -> None:
        """
        Generate a JSON output according to the current state machine state.

        Returns:
            None
        """
        while not self.__fsm.is_in_end_state():
            if self.__fsm.get_state() == JSONState.BEFORE_PROMPT:
                self.__generate_tokens_before_prompt()

            elif self.__fsm.get_state() == JSONState.IN_PROMPT:
                self.__generate_tokens_in_prompt()

            elif self.__fsm.get_state() == JSONState.BEFORE_NAME:
                self.__generate_tokens_before_name()

            elif self.__fsm.get_state() == JSONState.IN_NAME:
                self.__generate_tokens_in_name()

            elif self.__fsm.get_state() == JSONState.BEFORE_PARAMETERS:
                self.__generate_tokens_before_parameters()

            elif self.__fsm.get_state() == JSONState.IN_PARAMETERS:
                self.__generate_tokens_in_parameters()
        Visualizer.print_next(self.__json_result.rstrip().rstrip('\n'))

    def __reinitial_data_for_each_prompt(self, user_prompt: str) -> None:
        """
        Reinitialize generation data for a new user prompt.

        Args:
            user_prompt (str): User prompt to process.

        Returns:
            None
        """
        self.__fsm.set_state(JSONState.BEFORE_PROMPT)
        self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
        self.__fsm.set_parameters_state(ParameterState.IN_KEY)

        self.__current_prompt = user_prompt
        self.__len_current_prompt = len(self.__current_prompt)
        self.__ids_current_prompt = self.__prefix_ids[:]

        self.__index_key_param = 0
        self.__current_function_name = ''
        self.__dynamic_generated = ''
        self.__dynamic_ids = []
        self.__json_result = ''

    def execute(self, data: Any) -> Any:
        """
        Execute JSON generation for the provided input data.

        Args:
            data (Any): Input data containing prompts
                and generation settings.
        Returns:
            Any: Dictionary containing generated JSON results
                and output path.
        """
        self.__model = Small_LLM_Model()
        self.__prepare_data(data)
        start = time.time()
        for prompt_schema in self.__prompts:
            self.__reinitial_data_for_each_prompt(
                prompt_schema.prompt['prompt']
            )

            self.__generate()
            self.__json_results.append(self.__json_result)
        print(f"{(time.time() - start) / 60}")
        return {
            'json_results': self.__json_results,
            'output_path': data.get('output_path', '')
        }

    def __get_parameters_function(
        self,
        function_name: str,
    ) -> dict[str, dict[str, str]]:
        """
        Get function parameters by function name.

        Args:
            function_name (str): Name of the function.
        Returns:
            dict[str, dict[str, str]]:
                Dictionary containing function parameters.
        """
        for function in self.__functions_definition:
            if function.name == function_name:
                return {
                    key: value
                    for key, value in function.parameters.items()
                }
        return dict()

    def __get_only_available_function(
        self,
        dynamic_generated: str,
    ) -> str:
        """
        Return the matching function name prefix.

        Args:
            dynamic_generated (str): Generated function name prefix.

        Returns:
            str: Matching function name if found, otherwise the original value.
        """
        for function in self.__function_names:
            if function.startswith(dynamic_generated):
                return function
        return dynamic_generated
