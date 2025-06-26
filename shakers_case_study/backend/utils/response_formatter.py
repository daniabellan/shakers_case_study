class ResponseFormatter:
    def __init__(self, locale="en"):
        self.locale = locale
    
    def format(self, payload=None, status: str = None, message=None, **metadata):
        if status is None:
            raise ValueError("Status must be provided (e.g., 'success' or 'error')")
        
        response = {
            "status": status,
            "payload": payload,
        }
        if message:
            response["message"] = message
        response.update(metadata)
        return response