from typing import Any
from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
from src.Enums import FunctionDefinitionKeys


class PromptSchema(BaseModel):
    """PromptSchema model for each prompt."""
    prompt: dict[str, str] = Field(max_length=1, alias="prompt")

    @model_validator(mode='after')
    def validate_prompt_schema(self) -> Self:
        """
        Validate that the prompt dictionary contains only the required key.

        Returns:
            Self: Validated prompt schema instance.

        Raises:
            ValueError: If the prompt dictionary does not contain exactly
                one valid prompt key.
        """
        keys = self.prompt.keys()
        if len(keys) != 1:
            raise ValueError("dictionary must contain one pair.")
        if list(self.prompt)[0] != 'prompt':
            raise ValueError("key prompt must always prompt.")
        return self


class FunctionDefinitionSchema(BaseModel):
    """Model function definition schema for every function schema."""
    name: str = Field(
        max_length=40,
        min_length=6,
        pattern=r"^[A-Za-z_]+$",
        alias="name",
    )
    description: str = Field(
        max_length=300,
        min_length=10,
        alias="description",
    )
    parameters: dict[str, dict[str, str]] = Field(
        max_length=10,
        min_length=0,
        alias="parameters",
    )
    returns: dict[str, str] = Field(max_length=1, alias="returns")

    @model_validator(mode='before')
    def validate_keys_functions_definition(cls, data: Any) -> Any:
        """
        Validate function definition keys before model creation.

        Args:
            data (Any): Raw function definition data.

        Returns:
            Any: Validated function definition data.

        Raises:
            ValueError: If the input is not a valid dictionary or contains
                unsupported keys.
        """
        if not isinstance(data, dict):
            raise ValueError('functions definition must be dictionary')
        keys = list(data.keys())
        for key in keys:
            if key not in FunctionDefinitionKeys:
                raise ValueError(f'{key} must be in {keys}')
        return data

    @model_validator(mode='after')
    def validate_function_definition_schema(self) -> Self:
        """
        Validate the function definition after model creation.

        Returns:
            Self: Validated function definition schema.
        """

        self.__validate_parameters()
        self.__validate_returns()
        return self

    def __validate_keys_parameters(self) -> None:
        """
        Validate function parameter keys.

        Returns:
            None
        """
        keys_parameters = list(self.parameters.keys())
        for key_param in keys_parameters:
            if not key_param:
                raise ValueError('key parameters cannot be empty')

    def __validate_values_parameters(self) -> None:
        """
        Validate function parameter values.

        Returns:
            None
        """
        values_parameters = list(self.parameters.values())
        for key_val in values_parameters:
            if not isinstance(key_val, dict):
                raise ValueError('values in parameters must be dictionary')

            if len(key_val) != 1:
                raise ValueError('values in parameters must one pair')

            keys_of_values = list(key_val.keys())
            if keys_of_values[0] != 'type':
                raise ValueError('key of values in parameters must be "type"')

    def __validate_parameters(self) -> None:
        """
        Validate function parameters structure.

        Returns:
            None
        """
        self.__validate_keys_parameters()
        self.__validate_values_parameters()

    def __validate_returns(self) -> None:
        """
        Validate function return structure.

        Returns:
            None

        Raises:
            ValueError: If the return definition does not contain a single
                type key.
        """
        if len(self.returns) > 1:
            raise ValueError(
                "returns dictionary must contain 1 item {key: value}"
            )

        for key in self.returns.keys():
            if key != "type":
                raise ValueError(
                    f'in returns key "{key}" must be "type".'
                )
