from src.Enums import JSONState, JSONStatic, ParameterState


class FiniteStateMachine:
    """ finite state machine for remain model what a state in

    Attributes:
        self.__state: JSONState is global state of FSM
        self.__static_json: static ids decode by state
        self.__parameters_state: Parameters State if in Parameters state
    """

    def __init__(self) -> None:
        """constructor of finite state machine  initial stats

        Args:
            None:
        returns:
            None
        """
        self.__state = JSONState.BEFORE_PROMPT
        self.__static_json = JSONStatic.STR_BEFORE_PROMPT
        self.__parameters_state = ParameterState.IN_KEY

    def get_static_json(self) -> str:
        """ get static json static data json as string

        Args:
            None
        Returns:
            str: static json value for encoding by state
        """
        return self.__static_json.value

    def get_state(self) -> JSONState:
        """ get state of finite state machine

        Args:
            None
        Returns:
            JSONState: return global current state
        """
        return self.__state

    def get_parameters_state(self) -> ParameterState:
        """ get parameters state

        Args:
            None
        Returns:
            ParameterState: current Parameter state"""
        return self.__parameters_state

    def set_parameters_state(self, state: ParameterState) -> None:
        """
        Set the current parameters state.

        Args:
            state (ParameterState): The new parameters state.
        """
        self.__parameters_state = state

    def set_state(self, new_state: JSONState) -> None:
        """
        Set the current state of the finite state machine.

        Args:
            new_state (JSONState): The new state to assign.
        """
        self.__state = new_state

    def set_static_json(self, state: JSONStatic) -> None:
        """
        Set the static JSON state.

        Args:
            state (JSONStatic): The static JSON state to assign.
        """
        self.__static_json = state

    def is_in_end_state(self) -> bool:
        """
        Check whether the finite state machine is in the end state.

        Returns:
            bool: True if the current state is JSONState.IN_END,
                otherwise False``.
        """
        if self.__state == JSONState.IN_END:
            return True
        return False

    def reinitial_stats(self) -> None:
        """
        Reinitialize the state machine to its initial state
            for JSON generation.

        Returns:
            None
        """
        self.set_state(JSONState.BEFORE_PROMPT)
        self.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
        self.set_parameters_state(ParameterState.IN_KEY)
