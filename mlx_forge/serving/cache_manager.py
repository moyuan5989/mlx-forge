"""Multi-turn KV cache persistence and system prompt caching."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass


@dataclass
class ConversationState:
    """Cached state for a single conversation."""

    cache: list  # list[KVCache]
    token_ids: list[int]  # all tokens processed so far
    last_access: float
    conversation_id: str


class CacheManager:
    """Manages per-conversation KV caches for multi-turn inference.

    Each conversation's KV state is preserved between turns so that only
    new tokens need prefill. Also caches system prompt KV states, shared
    across conversations with the same system prompt.

    Args:
        max_conversations: Maximum number of conversation caches to keep.
        ttl_seconds: Time-to-live for idle conversations before eviction.
    """

    def __init__(self, max_conversations: int = 16, ttl_seconds: float = 600):
        self._conversations: dict[str, ConversationState] = {}
        self._system_caches: dict[str, list] = {}  # hash -> cache
        self._system_tokens: dict[str, list[int]] = {}  # hash -> token_ids
        self._max_conversations = max_conversations
        self._ttl_seconds = ttl_seconds

    def get_or_create(
        self, conversation_id: str, prompt_tokens: list[int], model
    ) -> tuple[list, list[int]]:
        """Get cached state or create new. Returns (cache, tokens_to_prefill).

        If the conversation exists and the prompt starts with the cached
        token sequence, returns only the delta tokens needing prefill.
        Otherwise creates a fresh cache.

        Args:
            conversation_id: Unique conversation identifier.
            prompt_tokens: Full tokenized prompt for this turn.
            model: Model instance (for creating new caches).

        Returns:
            Tuple of (cache, new_tokens_to_prefill).
        """
        self._evict_expired()

        if conversation_id in self._conversations:
            state = self._conversations[conversation_id]
            state.last_access = time.time()

            cached = state.token_ids
            # Check if prompt starts with cached tokens (prefix match)
            if (
                len(prompt_tokens) >= len(cached)
                and prompt_tokens[: len(cached)] == cached
            ):
                # Return delta tokens
                new_tokens = prompt_tokens[len(cached) :]
                return state.cache, new_tokens
            else:
                # Prompt diverged — reset cache
                del self._conversations[conversation_id]

        # Create new cache
        from mlx_forge.inference.engine import _make_model_cache

        max_size = len(prompt_tokens) + 4096  # generous buffer
        cache = _make_model_cache(model, max_size=max_size)
        return cache, prompt_tokens

    def update(
        self, conversation_id: str, cache: list, all_token_ids: list[int]
    ) -> None:
        """Store/update conversation state after generation.

        Args:
            conversation_id: Unique conversation identifier.
            cache: The KV cache after generation.
            all_token_ids: All token IDs processed (prompt + generated).
        """
        self._conversations[conversation_id] = ConversationState(
            cache=cache,
            token_ids=list(all_token_ids),
            last_access=time.time(),
            conversation_id=conversation_id,
        )
        # Enforce max limit via LRU eviction
        while len(self._conversations) > self._max_conversations:
            self._evict_lru()

    def evict(self, conversation_id: str) -> bool:
        """Explicitly evict a conversation cache.

        Returns:
            True if the conversation was found and evicted.
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "active_conversations": len(self._conversations),
            "max_conversations": self._max_conversations,
            "system_caches": len(self._system_caches),
            "conversation_ids": list(self._conversations.keys()),
        }

    # --- System prompt caching ---

    def _hash_tokens(self, tokens: list[int]) -> str:
        """Hash a token sequence for cache lookup."""
        data = bytes(str(tokens), "utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    def get_system_cache(
        self, system_tokens: list[int], model
    ) -> list | None:
        """Look up pre-computed KV for a system prompt.

        Returns a deep copy of the cache so each conversation can grow independently.

        Args:
            system_tokens: Tokenized system prompt.
            model: Model instance (unused, for interface consistency).

        Returns:
            Deep copy of cached KV state, or None if not cached.
        """
        key = self._hash_tokens(system_tokens)
        if key in self._system_caches:
            return copy.deepcopy(self._system_caches[key])
        return None

    def store_system_cache(self, system_tokens: list[int], cache: list) -> None:
        """Store a system prompt's KV cache for reuse.

        Args:
            system_tokens: Tokenized system prompt.
            cache: KV cache after processing the system prompt.
        """
        key = self._hash_tokens(system_tokens)
        self._system_caches[key] = copy.deepcopy(cache)
        self._system_tokens[key] = list(system_tokens)

    # --- Internal ---

    def _evict_expired(self) -> None:
        """Remove conversations that exceeded TTL."""
        now = time.time()
        expired = [
            cid
            for cid, state in self._conversations.items()
            if (now - state.last_access) > self._ttl_seconds
        ]
        for cid in expired:
            del self._conversations[cid]

    def _evict_lru(self) -> None:
        """Remove the least-recently-used conversation."""
        if not self._conversations:
            return
        lru_id = min(
            self._conversations, key=lambda k: self._conversations[k].last_access
        )
        del self._conversations[lru_id]
