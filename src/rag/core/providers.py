"""A typed registry mapping a configuration choice to a factory (ADR-011).

Adding a provider becomes one registration rather than an edit to a growing
``if/elif`` chain. That is the mechanism behind every "add a provider" extension
point in the architecture: the container registers the factories it knows, and
resolution is a lookup.

Kept deliberately small. A dependency-injection framework was considered and
rejected: for a graph this size it adds a vocabulary to learn, and its wiring
mistakes surface at runtime rather than under a type checker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag.domain.errors import ConfigurationError

__all__ = ["ProviderRegistry"]


class ProviderRegistry[K, T]:
    """Maps provider keys to the factories that build them.

    Type parameters:
        K: The key type, normally a configuration enum.
        T: The port type every registered factory produces.
    """

    def __init__(self, port_name: str) -> None:
        """Initialise an empty registry.

        Args:
            port_name: What is being registered, e.g. ``"embedder"``. Used in
                error messages so an operator can tell *which* provider setting
                is wrong.
        """
        self._port_name = port_name
        self._factories: dict[K, Callable[..., T]] = {}

    def register(self, key: K, factory: Callable[..., T]) -> None:
        """Register a factory for a provider key.

        Args:
            key: The configuration value that selects this provider.
            factory: Callable building the adapter.

        Raises:
            ConfigurationError: If the key is already registered. Silent
                replacement would make wiring depend on import order, which is
                the kind of bug that reproduces only in one environment.
        """
        if key in self._factories:
            raise ConfigurationError(
                f"{self._port_name} provider {key!r} is already registered",
                context={"port": self._port_name, "provider": str(key)},
            )
        self._factories[key] = factory

    def create(self, key: K, *args: Any, **kwargs: Any) -> T:
        """Build the adapter registered for a key.

        Args:
            key: The configured provider.
            *args: Positional arguments for the factory.
            **kwargs: Keyword arguments for the factory.

        Returns:
            The constructed adapter.

        Raises:
            ConfigurationError: If nothing is registered for the key. The
                message lists what is available, so the operator's next step is
                not reading source code.
        """
        factory = self._factories.get(key)
        if factory is None:
            available = ", ".join(str(known) for known in self.registered()) or "none"
            raise ConfigurationError(
                f"no {self._port_name} provider registered for {key!r}; available: {available}",
                context={"port": self._port_name, "requested": str(key)},
            )
        return factory(*args, **kwargs)

    def registered(self) -> tuple[K, ...]:
        """Return every registered key, in registration order."""
        return tuple(self._factories)

    def __contains__(self, key: object) -> bool:
        """Whether a factory is registered for a key."""
        return key in self._factories
