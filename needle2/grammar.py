"""Byte-level constrained decoding for a deliberately bounded tool schema subset.

Supports closed objects, required/optional properties, nested arrays, primitive
types and scalar enums. Object properties follow declaration order; tool call
objects use name then arguments. This is not the official grammar compiler or
full JSON Schema: unsupported validation keywords fail at construction. Optional
properties use a linear-size expression, not enumeration of all subsets.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import regex

_WS = br"[ \t\n\r]*"
_STRING = br'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
_INTEGER = br"-?(?:0|[1-9][0-9]*)"
_NUMBER = _INTEGER + br"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_METADATA = {"title", "description", "default", "examples", "$comment", "deprecated",
             "readOnly", "writeOnly"}


def _literal(value: Any) -> bytes:
    try:
        return regex.escape(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                       separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"Not a valid finite JSON literal: {value!r}") from exc


def _is_type(value, kind):
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return type(value) is bool
    if kind == "null":
        return value is None
    if kind == "integer":
        return type(value) is int
    if kind == "number":
        return type(value) in (int, float) and math.isfinite(value)
    return False


def _compile_schema(schema: dict, *, location: str = "$", depth: int = 0) -> bytes:
    if not isinstance(schema, dict):
        raise TypeError(f"{location}: schemas must be objects; boolean schemas are unsupported")
    if depth > 32:
        raise ValueError(f"{location}: nesting deeper than 32 is unsupported")
    kind = schema.get("type")
    if kind is not None and not isinstance(kind, str):
        raise ValueError(f"{location}: type unions are unsupported")
    specific = ({"properties", "required", "additionalProperties"} if kind == "object" else
                {"items", "minItems", "maxItems"} if kind == "array" else set())
    unsupported = set(schema) - (_METADATA | {"type", "enum"} | specific)
    if unsupported:
        raise ValueError(f"{location}: unsupported schema keywords: {sorted(unsupported)}")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{location}: enum must be a nonempty array")
        if any(isinstance(value, (dict, list)) for value in values):
            raise ValueError(f"{location}: only scalar enum values are supported")
        if kind is not None and not all(_is_type(value, kind) for value in values):
            raise ValueError(f"{location}: enum values do not match type {kind!r}")
        return br"(?:" + b"|".join(_literal(value) for value in values) + br")"
    if kind == "string":
        return _STRING
    if kind == "integer":
        return _INTEGER
    if kind == "number":
        return _NUMBER
    if kind == "boolean":
        return br"(?:true|false)"
    if kind == "null":
        return br"null"
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not all(isinstance(key, str) for key in properties):
            raise ValueError(f"{location}: properties must map string keys to schemas")
        if (not isinstance(required, list) or not all(isinstance(key, str) for key in required)
                or len(set(required)) != len(required) or set(required) - set(properties)):
            raise ValueError(f"{location}: required must contain distinct declared property names")
        if schema.get("additionalProperties", False) is not False:
            raise ValueError(f"{location}: additionalProperties must be false or omitted")
        # The cursor is immediately after '{' until the first property. Later
        # properties require a comma; no capture state leaks into nested arrays.
        separator = br"(?:(?<=\{)|(?<!\{)" + _WS + br",)" + _WS
        pieces = []
        for name, subschema in properties.items():
            value = _compile_schema(subschema, location=f"{location}.{name}", depth=depth + 1)
            piece = separator + _literal(name) + _WS + b":" + _WS + value
            pieces.append(br"(?:" + piece + br")" + (b"" if name in required else b"?"))
        return br"\{" + b"".join(pieces) + _WS + br"\}"
    if kind == "array":
        if "items" not in schema:
            raise ValueError(f"{location}: homogeneous arrays need an items schema")
        item = _compile_schema(schema["items"], location=location + "[]", depth=depth + 1)
        minimum, maximum = schema.get("minItems", 0), schema.get("maxItems")
        if (type(minimum) is not int or minimum < 0 or
                (maximum is not None and (type(maximum) is not int or maximum < minimum))):
            raise ValueError(f"{location}: invalid minItems/maxItems")
        if minimum > 1024 or (maximum is not None and maximum > 1024):
            raise ValueError(f"{location}: array count bounds above 1024 are unsupported")
        if maximum == 0:
            return br"\[" + _WS + br"\]"
        lower = max(0, minimum - 1)
        upper = b"" if maximum is None else str(maximum - 1).encode()
        repetition = b"{" + str(lower).encode() + b"," + upper + b"}"
        sequence = item + br"(?:" + _WS + b"," + _WS + item + br")" + repetition
        if minimum == 0:
            sequence = br"(?:" + sequence + br")?"
        return br"\[" + _WS + sequence + _WS + br"\]"
    raise ValueError(f"{location}: explicit supported type or scalar enum is required (got {kind!r})")


def compile_tools(tools: list[dict]) -> bytes:
    """Return a regex for an array of closed name/arguments tool-call objects."""
    if not isinstance(tools, list):
        raise TypeError("tools must be a list of schemas")
    alternatives, names = [], set()
    for index, original in enumerate(tools):
        if not isinstance(original, dict):
            raise TypeError(f"tool {index}: expected an object")
        if original.get("type") == "function" and "function" in original:
            if set(original) - {"type", "function"}:
                raise ValueError(f"tool {index}: unsupported function wrapper fields")
            tool = original["function"]
        else:
            tool = original
        if not isinstance(tool, dict) or set(tool) - {"name", "description", "parameters", "strict"}:
            raise ValueError(f"tool {index}: unsupported tool fields")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"tool {index}: name must be a distinct nonempty string")
        names.add(name)
        parameters = tool.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise ValueError(f"tool {name}: parameters must be an object schema")
        arguments = _compile_schema(parameters, location=f"{name}.arguments")
        alternative = (br"\{" + _WS + br'"name"' + _WS + b":" + _WS + _literal(name)
                       + _WS + b"," + _WS + br'"arguments"' + _WS + b":" + _WS
                       + arguments + _WS + br"\}")
        alternatives.append(alternative)
    if not alternatives:
        return _WS + br"\[" + _WS + br"\]" + _WS
    call = br"(?:" + b"|".join(alternatives) + br")"
    return (_WS + br"\[" + _WS + br"(?:" + call + br"(?:" + _WS + b"," + _WS
            + call + br")*)?" + _WS + br"\]" + _WS)


def _utf8_prefix_valid(data: bytes, *, complete: bool = False) -> bool:
    try:
        data.decode("utf-8", "strict")
        return True
    except UnicodeDecodeError as exc:
        return not complete and exc.reason == "unexpected end of data" and exc.end == len(data)


class ToolGrammar:
    """Stateful schema gate, inactive until <tool_call> (Needle token 10).

    select does not mutate state. Pass the emitted token to accept. Before the
    opening marker logits are unconstrained, preserving the model's reasoning
    behavior. After a complete JSON array only normal whitespace or token 11
    may follow. Once token 11 is accepted, finished is true.
    """

    def __init__(self, tools, tokenizer):
        self.tokenizer = tokenizer
        self.pattern = compile_tools(tools)
        self._matcher = regex.compile(self.pattern, regex.VERSION1)
        self.start_id = tokenizer.p2id.get("<tool_call>", 10)
        self.end_id = tokenizer.p2id.get("</tool_call>", 11)
        self.eos_id = 1
        self.prefix = b""
        self.active = False
        self.finished = False
        self._pieces = []
        self._by_lead = [[] for _ in range(256)]
        for tok_id, (piece, kind) in enumerate(zip(tokenizer.pieces, tokenizer.types)):
            if kind == 4:
                p = bytes([int(piece[3:5], 16)])
                self._pieces.append(p)
                self._by_lead[p[0]].append(tok_id)
            elif kind == 0:
                p = piece.replace("▁", " ").encode("utf-8")
                self._pieces.append(p)
                if p:
                    self._by_lead[p[0]].append(tok_id)
            else:
                self._pieces.append(None)

    @property
    def json_complete(self):
        if not self.active or not _utf8_prefix_valid(self.prefix, complete=True):
            return False
        match = self._matcher.fullmatch(self.prefix, partial=True)
        return match is not None and not match.partial

    def can_accept(self, token_id: int) -> bool:
        token_id = int(token_id)
        if token_id < 0 or token_id >= len(self._pieces):
            return False
        if self.finished:
            return token_id == self.eos_id
        if not self.active:
            return True
        if token_id == self.end_id:
            return self.json_complete
        piece = self._pieces[token_id]
        if not piece:
            return False
        candidate = self.prefix + piece
        return (_utf8_prefix_valid(candidate) and
                self._matcher.fullmatch(candidate, partial=True) is not None)

    def candidate_tokens(self, max_candidates: int = 1024) -> list[int] | None:
        """Return acceptable token IDs at current state, or None if unconstrained / broad."""
        if self.finished:
            return [self.eos_id]
        if not self.active:
            return None
        matched_bytes = [b for b in range(256) if self._matcher.fullmatch(self.prefix + bytes([b]), partial=True)]
        if len(matched_bytes) > 64:
            return None
        cands = []
        if self.end_id is not None and self.json_complete:
            cands.append(self.end_id)
        for b in matched_bytes:
            for t in self._by_lead[b]:
                if self.can_accept(t):
                    cands.append(t)
                    if len(cands) > max_candidates:
                        return None
        return cands if cands else None

    def select_candidate(self, candidates: list[int], logits) -> int:
        """Select the highest-scoring valid token among candidates."""
        if not candidates:
            raise RuntimeError("No candidate tokens provided")
        if len(candidates) == 1:
            return candidates[0]
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        scores = np.asarray(logits).reshape(-1)
        if len(scores) != len(candidates):
            raise ValueError(f"Expected {len(candidates)} candidate logits, got {len(scores)}")
        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def select(self, logits) -> int:
        if hasattr(logits, "detach"):
            logits = logits.detach().float().cpu().numpy()
        scores = np.asarray(logits).reshape(-1)
        if len(scores) != len(self._pieces):
            raise ValueError(f"Expected {len(self._pieces)} logits, got {len(scores)}")
        if np.isnan(scores).any():
            raise ValueError("NaN logits are not supported")
        if not self.active and not self.finished:
            return int(np.argmax(scores))
        for token in np.argsort(-scores, kind="stable"):
            if scores[token] == -np.inf:
                break
            if self.can_accept(int(token)):
                return int(token)
        raise RuntimeError(f"No finite-logit token extends the tool JSON grammar at {self.prefix[-120:]!r}")

    def accept(self, token_id: int):
        token_id = int(token_id)
        if not self.can_accept(token_id):
            raise ValueError(f"Token {token_id} violates the tool grammar at {self.prefix[-120:]!r}")
        if self.finished:
            return
        if not self.active:
            self.active = token_id == self.start_id
        elif token_id == self.end_id:
            self.finished = True
        else:
            self.prefix += self._pieces[token_id]


class NativeGrammarDFA:
    """Compiled prefix DFA state machine for zero-overhead C++ decode loop."""

    def __init__(
        self,
        num_states: int,
        initial_state: int,
        eos_id: int,
        stop_id: int,
        tool_start_id: int,
        tool_end_id: int,
        state_types: list[int] | np.ndarray,
        fallback_next_states: list[int] | np.ndarray,
        candidate_offsets: list[int] | np.ndarray,
        candidate_tokens: list[int] | np.ndarray,
        next_states: list[int] | np.ndarray,
    ):
        import ctypes as ct
        from .native import _DFAStateDesc

        self.num_states = int(num_states)
        self.initial_state = int(initial_state)
        self.eos_id = int(eos_id)
        self.stop_id = int(stop_id)
        self.tool_start_id = int(tool_start_id)
        self.tool_end_id = int(tool_end_id)

        self.state_types = np.ascontiguousarray(state_types, dtype=np.int32)
        self.fallback_next_states = np.ascontiguousarray(fallback_next_states, dtype=np.int32)
        self.candidate_offsets = np.ascontiguousarray(candidate_offsets, dtype=np.int32)
        self.candidate_tokens = np.ascontiguousarray(candidate_tokens, dtype=np.int32)
        self.next_states = np.ascontiguousarray(next_states, dtype=np.int32)

        self._desc = _DFAStateDesc()
        self._desc.num_states = self.num_states
        self._desc.initial_state = self.initial_state
        self._desc.eos_id = self.eos_id
        self._desc.stop_id = self.stop_id
        self._desc.tool_start_id = self.tool_start_id
        self._desc.tool_end_id = self.tool_end_id
        self._desc.state_types = self.state_types.ctypes.data_as(ct.POINTER(ct.c_int))
        self._desc.fallback_next_states = self.fallback_next_states.ctypes.data_as(ct.POINTER(ct.c_int))
        self._desc.candidate_offsets = self.candidate_offsets.ctypes.data_as(ct.POINTER(ct.c_int))
        self._desc.candidate_tokens = self.candidate_tokens.ctypes.data_as(ct.POINTER(ct.c_int))
        self._desc.next_states = self.next_states.ctypes.data_as(ct.POINTER(ct.c_int))


def compile_tool_dfa(tools: list[dict], tokenizer) -> NativeGrammarDFA:
    """Compile tool schemas into a prefix DFA state transition table for native C++ decoding."""
    states = []

    def add_state(stype=1, fallback=-1):
        idx = len(states)
        states.append({"type": stype, "fallback": fallback, "transitions": {}})
        return idx

    def add_transition(src, tok, dst):
        states[src]["transitions"][tok] = dst

    def add_chain(src, tokens):
        curr = src
        for t in tokens:
            nxt = add_state(1)
            add_transition(curr, t, nxt)
            curr = nxt
        return curr

    eos_id = tokenizer.p2id.get("</s>", 1)
    stop_id = tokenizer.p2id.get("<stop>", 5)
    tool_start_id = tokenizer.p2id.get("<tool_call>", 10)
    tool_end_id = tokenizer.p2id.get("</tool_call>", 11)

    s_root = add_state(stype=0, fallback=0)
    s_term = add_state(stype=4)

    # Closing sequence:
    # s_close expects 1270 ("}}]")
    s_close = add_state(stype=1)
    s_close_1 = add_state(stype=1)  # after 1270
    s_close_2 = add_state(stype=1)  # after tool_end_id (11)
    add_transition(s_close, 1270, s_close_1)
    add_transition(s_close_1, tool_end_id, s_close_2)
    add_transition(s_close_2, eos_id, s_term)

    # Start of tool call
    s_after_start = add_state(stype=1)
    add_transition(s_root, tool_start_id, s_after_start)
    # Prefix: [{" -> name -> ":"
    s_branch = add_chain(s_after_start, [1075, 598, 359])

    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            tool = tool["function"]
        name = tool["name"]
        pref = 'name":"'
        name_tokens = tokenizer.encode(pref + name)[len(tokenizer.encode(pref)):]
        curr = s_branch
        for tok_id in name_tokens:
            if tok_id in states[curr]["transitions"]:
                curr = states[curr]["transitions"][tok_id]
            else:
                nxt = add_state(1)
                add_transition(curr, tok_id, nxt)
                curr = nxt

        # After tool name: "," -> arguments -> ":{"
        curr = add_chain(curr, [362, 1376, 433])

        properties = list(tool.get("parameters", {}).get("properties", {}).items())
        for p_idx, (p_name, p_schema) in enumerate(properties):
            is_last = (p_idx == len(properties) - 1)
            p_type = p_schema.get("type")
            pref_p = 'arguments":{"' if p_idx == 0 else '","'
            p_tokens = tokenizer.encode(pref_p + p_name)[len(tokenizer.encode(pref_p)):]
            curr = add_chain(curr, p_tokens)

            # Delimiter after property name:
            delim = 359 if p_type == "string" else 314
            s_val_entry = add_state(1)
            add_transition(curr, delim, s_val_entry)

            if p_type == "string":
                states[s_val_entry]["type"] = 0
                states[s_val_entry]["fallback"] = s_val_entry
                if not is_last:
                    nxt_prop_entry = add_state(1)
                    add_transition(s_val_entry, 362, nxt_prop_entry)
                    curr = nxt_prop_entry
                else:
                    add_transition(s_val_entry, 1270, s_close_1)
                    curr = s_close
            elif p_type == "boolean":
                tok_true = tokenizer.p2id.get("true", 1111)
                tok_false = tokenizer.p2id.get("false", 1325)
                if not is_last:
                    nxt_prop_entry = add_state(1)
                    s_bool_comma = add_state(1)
                    add_transition(s_val_entry, tok_true, s_bool_comma)
                    add_transition(s_val_entry, tok_false, s_bool_comma)
                    add_transition(s_bool_comma, 362, nxt_prop_entry)
                    curr = nxt_prop_entry
                else:
                    add_transition(s_val_entry, tok_true, s_close)
                    add_transition(s_val_entry, tok_false, s_close)
                    curr = s_close
            elif p_type in ("integer", "number"):
                states[s_val_entry]["type"] = 0
                states[s_val_entry]["fallback"] = s_val_entry
                if not is_last:
                    nxt_prop_entry = add_state(1)
                    add_transition(s_val_entry, 362, nxt_prop_entry)
                    curr = nxt_prop_entry
                else:
                    add_transition(s_val_entry, 1270, s_close_1)
                    curr = s_close

    num_states = len(states)
    state_types = [s["type"] for s in states]
    fallback_next = [s["fallback"] for s in states]
    cand_offsets = [0]
    cand_tokens = []
    next_states = []
    for s in states:
        for t, nxt in s["transitions"].items():
            cand_tokens.append(t)
            next_states.append(nxt)
        cand_offsets.append(len(cand_tokens))

    return NativeGrammarDFA(
        num_states=num_states,
        initial_state=0,
        eos_id=eos_id,
        stop_id=stop_id,
        tool_start_id=tool_start_id,
        tool_end_id=tool_end_id,
        state_types=state_types,
        fallback_next_states=fallback_next,
        candidate_offsets=cand_offsets,
        candidate_tokens=cand_tokens,
        next_states=next_states,
    )
