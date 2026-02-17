from enum import Enum
from pydantic import BaseModel, Field

class StatusLevel(str, Enum):
    """Enumeration for status severity levels."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class Status(BaseModel):
    """
    Represents the status of a checked service.
    """
    source: str = Field(..., description="The name of the service being checked, e.g., 'Google'.")
    details: str = Field(..., description="Specifics of what was checked, e.g., 'quota'.")
    level: StatusLevel = Field(..., description="The severity level of the status.")

    @property
    def icon(self) -> str:
        """A unicode icon representing the status level."""
        mapping = {
            StatusLevel.OK: "✅",
            StatusLevel.WARNING: "⚠",
            StatusLevel.CRITICAL: "🔥",
            StatusLevel.UNKNOWN: "❓",
        }
        return mapping.get(self.level, "❓")