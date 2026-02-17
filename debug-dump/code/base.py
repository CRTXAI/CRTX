from abc import ABC, abstractmethod
from triad.models.status import Status
from triad.exceptions import StatusCheckError

class IStatusChecker(ABC):
    """
    Interface for a service that can check and report its status.
    """

    @abstractmethod
    def get_status(self) -> Status:
        """
        Checks the service and returns its current status.

        :raises StatusCheckError: If the status check fails for any reason,
                                  such as a network error or API failure.
        :return: A Status object representing the service's current state.
        """
        raise NotImplementedError