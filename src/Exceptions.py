class ParsingError(Exception):
    """ParsingError class throw exception in specific pipeline"""
    def __init__(self, msg: str) -> None:
        """ Raising parsingError when error threw in parser

        Args:
            msg: display msg what raises ParsingError
        Returns:
            None
        """
        super().__init__(msg)


class ReadingError(Exception):
    """ ReadingError class throw Exception in specific pipeline"""
    def __init__(self, msg: str) -> None:
        """ Raising ReadingError when error threw in Reader

        Args:
            msg: display msg what raises ReadingError
        Returns:
            None
        """
        super().__init__(msg)
