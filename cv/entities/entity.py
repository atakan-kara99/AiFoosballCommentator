from typing import Any, Dict
from abc import ABC, abstractmethod

class Entity(ABC):
    """
    Base interface for all entities in the system.

    Every entity must provide a JSON-serializable representation of its state and
    a string representation.
    """
    
    # Global constant for pretty-printing JSON in the terminal.
    INDENT: int = 4

    @abstractmethod
    def get_json(self) -> Dict[str, Any]:
        """
        Returns:
            Dict[str, Any]: A JSON-serializable dictionary representing the entity's state.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns:
            str: A human-readable string representation of the entity.
        """
        pass
