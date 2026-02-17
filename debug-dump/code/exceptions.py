class TriadError(Exception):
    """Base exception for all application-specific errors."""
    pass

class StatusCheckError(TriadError):
    """Raised when a status check fails, e.g., due to a network error."""
    pass