"""Custom exceptions."""


class MessageHandlerError(Exception):
    """Exception raised when an error occur while handling a message."""


class TritonServerError(Exception):
    """Exception raised when TritonServer inference request fails."""



