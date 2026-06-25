"""Tree-of-Thoughts — search over reasoning steps (docs/05 §Phase 3).

A deliberately small **beam search** engine: from a frontier of partial solutions,
*expand* each into candidate next states, *evaluate* them with a value function,
*prune* to the best ``beam``, and repeat to ``depth`` — returning the first goal
hit (or the best leaf found).

It's decoupled from where thoughts come from. ``expand`` is any
``Callable[[state], Iterable[state]]``: in a verifiable domain it's the problem's
move generator (deterministic — what the tests use); in production it can wrap a
model via :class:`ModelExpander`. ``value_fn`` is likewise a plain callable (a
Python verifier in tests, an LLM-judge in production). So the same engine tests
the *search machinery* offline and drives *model* reasoning live — only the two
callables change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class _Node:
    state: Any
    path: list  # states from root to here, inclusive
    score: float


@dataclass
class SearchResult:
    found: bool
    state: Any  # the goal state if found, else the best leaf reached
    path: list
    score: float
    frontier_sizes: list[int] = field(default_factory=list)  # |frontier| per depth
    expansions: int = 0  # how many nodes were expanded (cost)


class TreeOfThought:
    def __init__(
        self,
        expand: Callable[[Any], Iterable[Any]],
        value_fn: Callable[[Any], float],
        is_goal: Callable[[Any], bool],
        *,
        branch: int = 3,
        beam: int = 2,
        depth: int = 3,
        key: Callable[[Any], Any] | None = None,  # de-dup key for states
    ) -> None:
        self.expand = expand
        self.value_fn = value_fn
        self.is_goal = is_goal
        self.branch = branch
        self.beam = beam
        self.depth = depth
        self.key = key or (lambda s: s)

    def search(self, root: Any) -> SearchResult:
        if self.is_goal(root):
            return SearchResult(True, root, [root], self.value_fn(root))
        frontier = [_Node(root, [root], self.value_fn(root))]
        sizes: list[int] = []
        expansions = 0

        for _ in range(self.depth):
            pool: list[_Node] = []
            seen: set = set()
            for node in frontier:
                expansions += 1
                children = []
                for child_state in self.expand(node.state):
                    if self.is_goal(child_state):  # goal check on generation
                        return SearchResult(
                            True,
                            child_state,
                            node.path + [child_state],
                            self.value_fn(child_state),
                            sizes,
                            expansions,
                        )
                    children.append(
                        _Node(child_state, node.path + [child_state], self.value_fn(child_state))
                    )
                # keep the best `branch` children proposed from this node
                children.sort(key=lambda n: n.score, reverse=True)
                for child in children[: self.branch]:
                    k = self.key(child.state)
                    if k in seen:
                        continue
                    seen.add(k)
                    pool.append(child)
            if not pool:
                break
            # prune the level to the best `beam` (this is the search's whole point)
            pool.sort(key=lambda n: n.score, reverse=True)
            frontier = pool[: self.beam]
            sizes.append(len(frontier))

        best = max(frontier, key=lambda n: n.score)
        return SearchResult(False, best.state, best.path, best.score, sizes, expansions)

    async def asearch(self, root: Any) -> SearchResult:
        """Like :meth:`search`, but ``expand`` may be async (a real-model expander).

        ``value_fn`` and ``is_goal`` stay synchronous (Python verifiers); only the
        thought-generation step is awaited, so the model drives the search.
        """
        import inspect

        async def expand(state):
            out = self.expand(state)
            return await out if inspect.isawaitable(out) else out

        if self.is_goal(root):
            return SearchResult(True, root, [root], self.value_fn(root))
        frontier = [_Node(root, [root], self.value_fn(root))]
        sizes: list[int] = []
        expansions = 0

        for _ in range(self.depth):
            pool: list[_Node] = []
            seen: set = set()
            for node in frontier:
                expansions += 1
                children = []
                for child_state in await expand(node.state):
                    if self.is_goal(child_state):
                        return SearchResult(
                            True, child_state, node.path + [child_state],
                            self.value_fn(child_state), sizes, expansions,
                        )
                    children.append(
                        _Node(child_state, node.path + [child_state], self.value_fn(child_state))
                    )
                children.sort(key=lambda n: n.score, reverse=True)
                for child in children[: self.branch]:
                    k = self.key(child.state)
                    if k in seen:
                        continue
                    seen.add(k)
                    pool.append(child)
            if not pool:
                break
            pool.sort(key=lambda n: n.score, reverse=True)
            frontier = pool[: self.beam]
            sizes.append(len(frontier))

        best = max(frontier, key=lambda n: n.score)
        return SearchResult(False, best.state, best.path, best.score, sizes, expansions)


class ModelExpander:
    """Adapt a :class:`Model` into an ``expand`` callable (production thought-gen).

    Not used by the deterministic tests — they pass a plain move generator — but it
    shows the engine is model-pluggable: ``render`` turns a state into a prompt,
    ``parse`` turns the model's reply into candidate next states.
    """

    def __init__(self, model, render: Callable[[Any], str], parse: Callable[[str], list]) -> None:
        self.model = model
        self.render = render
        self.parse = parse

    async def __call__(self, state: Any) -> list:
        from openmate.kernel.types import Message, TextPart
        from openmate.ports.model import ModelRequest

        resp = await self.model.generate(
            ModelRequest(messages=[Message("user", [TextPart(self.render(state))])])
        )
        return self.parse(resp.message.text)
