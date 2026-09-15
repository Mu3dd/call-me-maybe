from enum import Enum, auto


class JSONStatic(str, Enum):
    """small set of structural JSON tokens controlled by the FSM."""
    STR_BEFORE_PROMPT = '{"prompt": '
    STR_BEFORE_NAME = '"name": "'
    STR_BEFORE_PARAMETERS = '"parameters": {'


class JSONState(Enum):
    """states for this exact JSON shape.

    Target object:
    {
        "prompt": "<prompt user>",
        "name": "<function name>",
        "parameters": { ... }
    }
    """
    BEFORE_PROMPT = auto()
    IN_PROMPT = auto()

    BEFORE_NAME = auto()
    IN_NAME = auto()

    BEFORE_PARAMETERS = auto()
    IN_PARAMETERS = auto()

    IN_END = auto()


class ParameterState(Enum):
    """ states for parameters JSON shape"""
    IN_KEY = auto()
    IN_VALUE = auto()
    IN_CLOSE = auto()


class FunctionDefinitionKeys(Enum):
    """enum for functions definition schema"""
    name = 'name'
    description = 'description'
    parameters = 'parameters'
    returns = 'returns'
