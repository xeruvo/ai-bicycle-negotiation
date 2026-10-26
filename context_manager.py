"""Bounded conversation context management.

Content limits are measured in UTF-8 bytes, not exact model tokens. Explicit
model context sizing and output reservation will be connected later.
Implemented with AI assistance and reviewed against the context-management
requirements.
"""


class ContextLimitError(ValueError):
    """Raised when the protected context cannot fit within its byte limit."""


class SlidingWindow:
    """Keep a recent message window while protecting the latest three replies."""

    def __init__(self, max_messages=4, max_content_bytes=3000):
        if (
            isinstance(max_messages, bool)
            or not isinstance(max_messages, int)
            or max_messages < 3
        ):
            raise ValueError("max_messages must be an integer of at least 3")
        if (
            isinstance(max_content_bytes, bool)
            or not isinstance(max_content_bytes, int)
            or max_content_bytes <= 0
        ):
            raise ValueError("max_content_bytes must be a positive integer")
        self.max_messages = max_messages
        self.max_content_bytes = max_content_bytes
        self.events = []
        self._call_number = 0

    @staticmethod
    def _content_bytes(messages):
        return sum(len(message["content"].encode("utf-8")) for message in messages)

    def __call__(self, messages):
        self._call_number += 1
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
            raise ValueError("messages must begin with a system message")
        if any(
            not isinstance(message, dict)
            or "role" not in message
            or "content" not in message
            or not isinstance(message["content"], str)
            for message in messages
        ):
            raise ValueError("each message must contain a role and string content")

        original_count = len(messages)
        original_bytes = self._content_bytes(messages)
        system = dict(messages[0])
        non_system = messages[1:][-self.max_messages :]
        retained = [system] + [dict(message) for message in non_system]
        protected_count = min(3, len(non_system))
        protected_start = len(retained) - protected_count

        while self._content_bytes(retained) > self.max_content_bytes:
            removable_index = 1 if protected_start > 1 else None
            if removable_index is None:
                self._record_event(
                    original_count,
                    len(retained),
                    original_bytes,
                    self._content_bytes(retained),
                    status="rejected",
                )
                raise ContextLimitError(
                    "system prompt and protected recent messages exceed the byte limit"
                )
            retained.pop(removable_index)
            protected_start -= 1

        dropped_count = original_count - len(retained)
        if dropped_count:
            self._record_event(
                original_count,
                len(retained),
                original_bytes,
                self._content_bytes(retained),
            )
        return retained

    def _record_event(
        self,
        original_count,
        retained_count,
        original_bytes,
        retained_bytes,
        status="accepted",
    ):
        event = {
            "call_number": self._call_number,
            "original_count": original_count,
            "retained_count": retained_count,
            "dropped_count": original_count - retained_count,
            "original_content_bytes": original_bytes,
            "retained_content_bytes": retained_bytes,
            "strategy": "truncation",
        }
        if status == "rejected":
            event["status"] = status
        self.events.append(event)
