"""
Custom exception hierarchy for the service layer.

These exceptions are caught by the global exception handler in main.py
and converted to appropriate HTTP responses.
"""


class ServiceError(Exception):
    """Base exception for all service-layer errors."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class NotFoundError(ServiceError):
    """Resource not found (404)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ValidationError(ServiceError):
    """Invalid input or business rule violation (400)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class PricingError(ServiceError):
    """Pricing computation failure (500)."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)
