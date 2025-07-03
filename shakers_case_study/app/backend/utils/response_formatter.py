class ResponseFormatter:
    """
    A utility class to standardize API responses.

    Attributes:
        locale (str): Language/locale code for formatting responses (default is 'en').
    """

    def __init__(self, locale: str = "en"):
        """
        Initialize the ResponseFormatter with an optional locale.

        Args:
            locale (str, optional): Locale code for response formatting. Defaults to "en".
        """
        self.locale = locale

    def format(self, payload=None, status: str = None, message: str = None, **metadata):
        """
        Format the response dictionary consistently with a status, optional payload,
        message, and any additional metadata.

        Args:
            payload (Any, optional): The main content/data of the response.
            status (str): Status of the response, e.g., "success" or "error". Required.
            message (str, optional): Additional message or description.
            **metadata: Arbitrary keyword arguments to include as extra fields in the response.

        Returns:
            dict: A dictionary containing the formatted response.

        Raises:
            ValueError: If the 'status' argument is not provided.
        """
        if status is None:
            raise ValueError("Status must be provided (e.g., 'success' or 'error')")

        response = {
            "status": status,
            "payload": payload,
        }

        if message:
            response["message"] = message

        # Add any extra metadata fields to the response
        response.update(metadata)

        return response
