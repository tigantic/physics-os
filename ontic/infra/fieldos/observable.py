"""
Reactive Observable System
===========================

Event-driven reactive state management.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


T = TypeVar("T")


# =============================================================================
# EVENT TYPES
# =============================================================================


class EventType(Enum):
    """Standard event types."""

    # Value events
    CHANGE = "change"
    UPDATE = "update"
    RESET = "reset"

    # Lifecycle events
    CREATE = "create"
    DESTROY = "destroy"

    # State events
    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"

    # Error events
    ERROR = "error"
    WARNING = "warning"

    # Custom
    CUSTOM = "custom"


@dataclass
class Event:
    """
    An event with type, data, and metadata.
    """

    type: EventType
    data: Any = None
    source: str | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def change(cls, old_value: Any, new_value: Any, source: str = "") -> Event:
        """Create change event."""
        return cls(
            type=EventType.CHANGE,
            data={"old": old_value, "new": new_value},
            source=source,
        )

    @classmethod
    def update(cls, value: Any, source: str = "") -> Event:
        """Create update event."""
        return cls(type=EventType.UPDATE, data=value, source=source)

    @classmethod
    def error(cls, error: str, source: str = "") -> Event:
        """Create error event."""
        return cls(type=EventType.ERROR, data=error, source=source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "data": str(self.data),
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# =============================================================================
# OBSERVER
# =============================================================================


class Observer(ABC):
    """
    Abstract observer interface.
    """

    @abstractmethod
    def on_event(self, event: Event):
        """Handle event."""
        pass

    def on_complete(self):
        """Called when observable completes."""
        pass

    def on_error(self, error: Exception):
        """Called on error."""
        pass


class FunctionObserver(Observer):
    """Observer wrapping a function."""

    def __init__(
        self,
        on_event: Callable[[Event], None],
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self._on_event = on_event
        self._on_complete = on_complete
        self._on_error = on_error

    def on_event(self, event: Event):
        self._on_event(event)

    def on_complete(self):
        if self._on_complete:
            self._on_complete()

    def on_error(self, error: Exception):
        if self._on_error:
            self._on_error(error)


# =============================================================================
# SUBSCRIPTION
# =============================================================================


class Subscription:
    """
    A subscription that can be unsubscribed.

    Holds strong reference to observer to prevent GC.
    """

    def __init__(
        self,
        unsubscribe_fn: Callable[[], None],
        observer: Observer | None = None,
    ):
        self._unsubscribe = unsubscribe_fn
        self._observer = observer  # Strong reference keeps observer alive
        self._active = True

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def observer(self) -> Observer | None:
        return self._observer

    def unsubscribe(self):
        """Unsubscribe from observable."""
        if self._active:
            self._unsubscribe()
            self._active = False
            self._observer = None  # Release observer on unsubscribe


# =============================================================================
# OBSERVABLE
# =============================================================================


class Observable(Generic[T]):
    """
    Reactive observable value.

    Example:
        obs = Observable(0)

        # Subscribe
        sub = obs.subscribe(lambda e: print(f"Value: {e.data}"))

        # Update triggers notification
        obs.set(42)  # Prints: Value: 42

        # Unsubscribe
        sub.unsubscribe()
    """

    def __init__(
        self,
        initial: T | None = None,
        name: str = "",
    ):
        self._value = initial
        self._name = name
        self._observers: list[weakref.ref[Observer]] = []
        self._lock = threading.RLock()
        self._completed = False

    @property
    def value(self) -> T | None:
        """Get current value."""
        return self._value

    @property
    def name(self) -> str:
        return self._name

    def get(self) -> T | None:
        """Get current value."""
        return self._value

    def set(self, value: T, source: str = ""):
        """
        Set new value and notify observers.

        Args:
            value: New value
            source: Source identifier
        """
        if self._completed:
            return

        with self._lock:
            old_value = self._value
            self._value = value

            # Emit change event
            event = Event.change(old_value, value, source or self._name)
            self._notify(event)

    def update(self, fn: Callable[[T], T], source: str = ""):
        """
        Update value using function.

        Args:
            fn: Update function
            source: Source identifier
        """
        with self._lock:
            new_value = fn(self._value)
            self.set(new_value, source)

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    def subscribe(
        self,
        observer_or_fn: Observer | Callable[[Event], None],
    ) -> Subscription:
        """
        Subscribe to events.

        Args:
            observer_or_fn: Observer or callback function

        Returns:
            Subscription for unsubscribing
        """
        if callable(observer_or_fn) and not isinstance(observer_or_fn, Observer):
            observer = FunctionObserver(observer_or_fn)
        else:
            observer = observer_or_fn

        with self._lock:
            ref = weakref.ref(observer)
            self._observers.append(ref)

        def unsubscribe():
            with self._lock:
                self._observers = [o for o in self._observers if o() != observer]

        return Subscription(unsubscribe, observer)

    def _notify(self, event: Event):
        """Notify all observers."""
        with self._lock:
            # Clean up dead references
            live_observers = []
            for ref in self._observers:
                observer = ref()
                if observer is not None:
                    live_observers.append(ref)
                    try:
                        observer.on_event(event)
                    except Exception as e:
                        logger.error(f"Observer error: {e}")

            self._observers = live_observers

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def complete(self):
        """Complete the observable."""
        self._completed = True

        with self._lock:
            for ref in self._observers:
                observer = ref()
                if observer:
                    try:
                        observer.on_complete()
                    except Exception as e:
                        logger.error(f"Observer complete error: {e}")

    def error(self, err: Exception):
        """Signal error to observers."""
        with self._lock:
            for ref in self._observers:
                observer = ref()
                if observer:
                    try:
                        observer.on_error(err)
                    except Exception as e:
                        logger.error(f"Observer error handler error: {e}")

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def map(self, fn: Callable[[T], Any]) -> Observable:
        """
        Map values through function.

        Returns new Observable with transformed values.
        """
        mapped = Observable(fn(self._value) if self._value else None)

        def on_event(event: Event):
            if event.type == EventType.CHANGE:
                new_val = fn(event.data["new"])
                mapped.set(new_val, event.source)

        # Store subscription to prevent GC
        mapped._source_subscription = self.subscribe(on_event)
        return mapped

    def filter(self, predicate: Callable[[T], bool]) -> Observable:
        """
        Filter values by predicate.

        Returns new Observable that only updates when predicate is True.
        """
        initial = self._value if self._value and predicate(self._value) else None
        filtered = Observable(initial)

        def on_event(event: Event):
            if event.type == EventType.CHANGE:
                new_val = event.data["new"]
                if predicate(new_val):
                    filtered.set(new_val, event.source)

        # Store subscription to prevent GC
        filtered._source_subscription = self.subscribe(on_event)
        return filtered

    def debounce(self, delay: float) -> Observable:
        """
        Debounce updates.

        Only propagates after delay seconds of no updates.
        """
        debounced = Observable(self._value)
        timer: threading.Timer | None = None
        lock = threading.Lock()

        def on_event(event: Event):
            nonlocal timer
            with lock:
                if timer:
                    timer.cancel()

                def set_value():
                    if event.type == EventType.CHANGE:
                        debounced.set(event.data["new"], event.source)

                timer = threading.Timer(delay, set_value)
                timer.start()

        # Store subscription to prevent GC
        debounced._source_subscription = self.subscribe(on_event)
        return debounced

    def throttle(self, interval: float) -> Observable:
        """
        Throttle updates.

        Only propagates at most once per interval.
        """
        throttled = Observable(self._value)
        last_time = [0.0]

        def on_event(event: Event):
            now = time.time()
            if now - last_time[0] >= interval:
                last_time[0] = now
                if event.type == EventType.CHANGE:
                    throttled.set(event.data["new"], event.source)

        # Store subscription to prevent GC
        throttled._source_subscription = self.subscribe(on_event)
        return throttled

    def combine(self, other: Observable) -> Observable:
        """
        Combine with another observable.

        Returns Observable with tuple of both values.
        """
        combined = Observable((self._value, other._value))

        def on_self(event: Event):
            if event.type == EventType.CHANGE:
                combined.set((event.data["new"], other.value), event.source)

        def on_other(event: Event):
            if event.type == EventType.CHANGE:
                combined.set((self.value, event.data["new"]), event.source)

        # Store subscriptions to prevent GC
        combined._source_subscriptions = [
            self.subscribe(on_self),
            other.subscribe(on_other),
        ]

        return combined


# =============================================================================
# OBSERVABLE FIELD
# =============================================================================


class ObservableField(Observable[np.ndarray]):
    """
    Observable wrapper for numpy arrays.

    Adds field-specific operations and events.
    """

    def __init__(
        self,
        data: np.ndarray,
        name: str = "",
    ):
        super().__init__(data, name)

    def set_region(
        self,
        region: tuple,
        value: float | np.ndarray,
        source: str = "",
    ):
        """
        Set a region of the field.

        Args:
            region: Slice tuple
            value: Value to set
            source: Source identifier
        """
        with self._lock:
            old_value = self._value.copy()
            self._value[region] = value

            event = Event(
                type=EventType.UPDATE,
                data={"region": region, "value": value},
                source=source or self._name,
            )
            self._notify(event)

    def apply(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        source: str = "",
    ):
        """
        Apply function to field data.

        Args:
            fn: Function to apply
            source: Source identifier
        """
        with self._lock:
            old_value = self._value.copy()
            self._value = fn(self._value)

            event = Event.change(old_value, self._value, source or self._name)
            self._notify(event)

    @property
    def shape(self) -> tuple:
        return self._value.shape

    @property
    def dtype(self):
        return self._value.dtype


# =============================================================================
# EVENT BUS
# =============================================================================


class EventBus:
    """
    Central event bus for decoupled communication.

    Example:
        bus = EventBus()

        bus.on("field.updated", lambda e: print(e.data))
        bus.emit("field.updated", {"name": "temperature"})
    """

    def __init__(self):
        self._handlers: dict[str, list[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()

    def on(
        self,
        event_name: str,
        handler: Callable[[Event], None],
    ) -> Subscription:
        """
        Subscribe to event.

        Args:
            event_name: Event name (can use wildcards)
            handler: Event handler

        Returns:
            Subscription for unsubscribing
        """
        with self._lock:
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(handler)

        def unsubscribe():
            with self._lock:
                if event_name in self._handlers:
                    self._handlers[event_name] = [
                        h for h in self._handlers[event_name] if h != handler
                    ]

        return Subscription(unsubscribe)

    def off(self, event_name: str, handler: Callable[[Event], None]):
        """Unsubscribe from event."""
        with self._lock:
            if event_name in self._handlers:
                self._handlers[event_name] = [
                    h for h in self._handlers[event_name] if h != handler
                ]

    def emit(
        self,
        event_name: str,
        data: Any = None,
        source: str = "",
    ):
        """
        Emit event to subscribers.

        Args:
            event_name: Event name
            data: Event data
            source: Source identifier
        """
        event = Event(
            type=EventType.CUSTOM,
            data=data,
            source=source,
            metadata={"name": event_name},
        )

        with self._lock:
            handlers = self._handlers.get(event_name, []).copy()

            # Also check wildcard handlers
            for pattern, pattern_handlers in self._handlers.items():
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    if event_name.startswith(prefix):
                        handlers.extend(pattern_handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event_name}: {e}")

    def clear(self, event_name: str | None = None):
        """Clear handlers for event or all events."""
        with self._lock:
            if event_name:
                self._handlers.pop(event_name, None)
            else:
                self._handlers.clear()


# =============================================================================
# COMPUTED OBSERVABLE
# =============================================================================


class Computed(Observable[T]):
    """
    Computed observable that depends on other observables.

    Example:
        x = Observable(2)
        y = Observable(3)

        sum_xy = Computed(lambda: x.value + y.value, [x, y])
        print(sum_xy.value)  # 5

        x.set(10)
        print(sum_xy.value)  # 13
    """

    def __init__(
        self,
        compute: Callable[[], T],
        dependencies: list[Observable],
        name: str = "",
    ):
        self._compute = compute
        self._dependencies = dependencies
        self._subscriptions: list[Subscription] = []

        # Compute initial value
        initial = compute()
        super().__init__(initial, name)

        # Subscribe to dependencies
        for dep in dependencies:
            sub = dep.subscribe(self._on_dependency_change)
            self._subscriptions.append(sub)

    def _on_dependency_change(self, event: Event):
        """Recompute when dependency changes."""
        try:
            new_value = self._compute()
            self.set(new_value, "computed")
        except Exception as e:
            self.error(e)

    def set(self, value: T, source: str = ""):
        """Override set to update internal value and notify."""
        if self._completed:
            return

        with self._lock:
            old_value = self._value
            self._value = value

            if old_value != value:
                event = Event.change(old_value, value, source or self._name)
                self._notify(event)

    def dispose(self):
        """Dispose of subscriptions."""
        for sub in self._subscriptions:
            sub.unsubscribe()
        self._subscriptions.clear()
        self.complete()
