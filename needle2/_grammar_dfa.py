"""Thompson NFA for the validated ToolGrammar subset, determinized over tokens.

All body states constrain tokens, including string/number contents. UTF-8
validation is composed with the byte automaton before tokenizer transitions.
Compilation has explicit size limits; callers can fall back to ToolGrammar.
"""
from collections import deque
import json


class GrammarTooLarge(ValueError):
    """The exact DFA exceeds the compilation budget; use the regex gate."""


class _NFA:
    def __init__(self):
        self.edges = []
        self.eps = []

    def state(self):
        if len(self.edges) >= 20000:
            raise GrammarTooLarge("tool grammar NFA exceeds 20000 states")
        self.edges.append([])
        self.eps.append([])
        return len(self.edges) - 1

    def chars(self, values):
        a, b = self.state(), self.state()
        self.edges[a].append((sum(1 << v for v in set(values)), b))
        return a, b

    def seq(self, *parts):
        if not parts:
            a = self.state()
            return a, a
        for left, right in zip(parts, parts[1:]):
            self.eps[left[1]].append(right[0])
        return parts[0][0], parts[-1][1]

    def literal(self, data):
        return self.seq(*(self.chars([b]) for b in data))

    def alt(self, *parts):
        a, b = self.state(), self.state()
        for start, end in parts:
            self.eps[a].append(start)
            self.eps[end].append(b)
        return a, b

    def optional(self, part):
        a, b = self.alt(part)
        self.eps[a].append(b)
        return a, b

    def star(self, part):
        a, b = self.optional(part)
        self.eps[part[1]].append(part[0])
        return a, b

    def ws(self):
        return self.star(self.chars(b' \t\n\r'))

    def integer(self):
        return self.seq(self.optional(self.literal(b'-')),
                        self.alt(self.literal(b'0'), self.seq(self.chars(b'123456789'),
                                                             self.star(self.chars(b'0123456789')))))

    def string(self):
        escaped = self.seq(self.literal(b'\\'), self.alt(
            self.chars(b'"\\/bfnrt'),
            self.seq(self.literal(b'u'), *(self.chars(b'0123456789abcdefABCDEF') for _ in range(4)))))
        body = self.alt(self.chars(b for b in range(32, 256) if b not in (34, 92)), escaped)
        return self.seq(self.literal(b'"'), self.star(body), self.literal(b'"'))

    def value(self, value):
        return self.literal(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                       separators=(',', ':')).encode('utf-8'))

    def schema(self, schema):
        if 'enum' in schema:
            return self.alt(*(self.value(v) for v in schema['enum']))
        kind = schema['type']
        if kind == 'string':
            return self.string()
        if kind == 'integer':
            return self.integer()
        if kind == 'number':
            def digits():
                return self.seq(self.chars(b'0123456789'), self.star(self.chars(b'0123456789')))
            return self.seq(self.integer(),
                            self.optional(self.seq(self.literal(b'.'), digits())),
                            self.optional(self.seq(self.chars(b'eE'), self.optional(self.chars(b'+-')), digits())))
        if kind == 'boolean':
            return self.alt(self.literal(b'true'), self.literal(b'false'))
        if kind == 'null':
            return self.literal(b'null')
        if kind == 'object':
            start, empty = self.literal(b'{')
            nonempty = None
            required = schema.get('required', [])
            for name, subschema in schema.get('properties', {}).items():
                field_start, field_end = self.seq(self.ws(), self.value(name), self.ws(),
                                                  self.literal(b':'), self.ws(), self.schema(subschema))
                if empty is not None:
                    self.eps[empty].append(field_start)
                if nonempty is not None:
                    sep = self.seq(self.ws(), self.literal(b','))
                    self.eps[nonempty].append(sep[0])
                    self.eps[sep[1]].append(field_start)
                if name in required:
                    empty = None
                    nonempty = field_end
                else:
                    # Keep skip edges separate from the field's end. Connecting
                    # back to an old state would allow repeated/reordered keys.
                    join = self.state()
                    self.eps[field_end].append(join)
                    if nonempty is not None:
                        self.eps[nonempty].append(join)
                    nonempty = join
            close, end = self.seq(self.ws(), self.literal(b'}'))
            for last in (empty, nonempty):
                if last is not None:
                    self.eps[last].append(close)
            return start, end
        if kind == 'array':
            start, cursor = self.seq(self.literal(b'['), self.ws())
            close, end = self.seq(self.ws(), self.literal(b']'))
            minimum, maximum = schema.get('minItems', 0), schema.get('maxItems')
            if minimum == 0:
                self.eps[cursor].append(close)
            if maximum == 0:
                return start, end
            first = self.schema(schema['items'])
            self.eps[cursor].append(first[0])
            cursor = first[1]
            # At least one item; unbounded repetitions share a loop only after
            # satisfying the minimum. Bounded repetitions have distinct states.
            count = max(1, minimum) if maximum is None else maximum
            for n in range(1, count + 1):
                if n >= minimum:
                    self.eps[cursor].append(close)
                if n < count or maximum is None:
                    more = self.seq(self.ws(), self.literal(b','), self.ws(), self.schema(schema['items']))
                    self.eps[cursor].append(more[0])
                    if n == count:
                        self.eps[more[1]].append(cursor)
                    else:
                        cursor = more[1]
            return start, end
        raise ValueError(f'unsupported schema type {kind!r}')

    def tools(self, tools):
        calls = []
        for tool in tools:
            if tool.get('type') == 'function':
                tool = tool['function']
            calls.append(self.seq(self.literal(b'{'), self.ws(), self.value('name'), self.ws(),
                                  self.literal(b':'), self.ws(), self.value(tool['name']), self.ws(),
                                  self.literal(b','), self.ws(), self.value('arguments'), self.ws(),
                                  self.literal(b':'), self.ws(),
                                  self.schema(tool.get('parameters', {'type': 'object', 'properties': {}})),
                                  self.ws(), self.literal(b'}')))
        start, cursor = self.seq(self.ws(), self.literal(b'['), self.ws())
        close, end = self.seq(self.ws(), self.literal(b']'), self.ws())
        self.eps[cursor].append(close)
        if calls:
            call = self.alt(*calls)
            self.eps[cursor].append(call[0])
            self.eps[call[1]].append(close)
            sep = self.seq(self.ws(), self.literal(b','), self.ws())
            self.eps[call[1]].append(sep[0])
            self.eps[sep[1]].append(call[0])
        return start, end


def _utf8_next(state, byte):
    # States 1..3 count continuation bytes; 4..7 narrow the first continuation
    # to reject overlong sequences, surrogates and code points above U+10FFFF.
    if state == 0:
        if byte < 128: return 0
        if 0xC2 <= byte <= 0xDF: return 1
        if byte == 0xE0: return 4
        if byte == 0xED: return 5
        if 0xE1 <= byte <= 0xEF: return 2
        if byte == 0xF0: return 6
        if byte == 0xF4: return 7
        if 0xF1 <= byte <= 0xF3: return 3
    elif state <= 3:
        if 0x80 <= byte <= 0xBF: return state - 1
    elif state == 4:
        if 0xA0 <= byte <= 0xBF: return 1
    elif state == 5:
        if 0x80 <= byte <= 0x9F: return 1
    elif state == 6:
        if 0x90 <= byte <= 0xBF: return 2
    elif state == 7:
        if 0x80 <= byte <= 0x8F: return 2
    return -1


def token_dfa(tools, pieces, types, start_id, end_id):
    nfa = _NFA()
    start, end = nfa.tools(tools)
    closures = {}

    def closure(nodes):
        key = frozenset(nodes)
        if key not in closures:
            reached = set(key)
            pending = list(key)
            while pending:
                for target in nfa.eps[pending.pop()]:
                    if target not in reached:
                        reached.add(target)
                        pending.append(target)
            closures[key] = frozenset(reached)
        return closures[key]

    # The trie shares common tokenizer prefixes; no assumptions about token IDs
    # or tokenization at property/name boundaries are made.
    trie = [{}]
    leaves = [[]]
    for token, (piece, kind) in enumerate(zip(pieces, types)):
        data = bytes([int(piece[3:5], 16)]) if kind == 4 else piece.replace('▁', ' ').encode('utf-8') if kind == 0 else b''
        if not data:
            continue
        node = 0
        for byte in data:
            if byte not in trie[node]:
                trie[node][byte] = len(trie)
                trie.append({})
                leaves.append([])
            node = trie[node][byte]
        leaves[node].append(token)

    byte_states = [(closure([start]), 0)]
    byte_indices = {byte_states[0]: 0}
    moves = {}

    def move(state, byte):
        key = state, byte
        if key in moves:
            return moves[key]
        active, utf8 = byte_states[state]
        next_utf8 = _utf8_next(utf8, byte)
        targets = set()
        if next_utf8 >= 0:
            for node in active:
                for mask, target in nfa.edges[node]:
                    if (mask >> byte) & 1:
                        targets.add(target)
        result = -1
        if targets:
            dest = closure(targets), next_utf8
            if dest not in byte_indices:
                if len(byte_states) >= 20000:
                    raise GrammarTooLarge('tool grammar byte DFA exceeds 20000 states')
                byte_indices[dest] = len(byte_states)
                byte_states.append(dest)
            result = byte_indices[dest]
        moves[key] = result
        return result

    transitions = [{start_id: 2}, {}]  # unconstrained root, terminal
    indices = {0: 2}
    pending = deque([0])
    total_edges = 1
    while pending:
        byte_state = pending.popleft()
        candidates = {}
        active, utf8 = byte_states[byte_state]
        if end in active and utf8 == 0:
            candidates[end_id] = 1
        stack = [(0, byte_state)]
        while stack:
            node, state = stack.pop()
            for token in leaves[node]:
                if state not in indices:
                    if len(indices) >= 4096:
                        raise GrammarTooLarge('tool grammar token DFA exceeds 4096 states')
                    indices[state] = len(indices) + 2
                    pending.append(state)
                candidates[token] = indices[state]
            for byte, child in trie[node].items():
                dest = move(state, byte)
                if dest >= 0:
                    stack.append((child, dest))
        total_edges += len(candidates)
        if total_edges > 4000000:
            raise GrammarTooLarge('tool grammar DFA exceeds 4000000 transitions')
        transitions.append(candidates)
    # Vocabulary without byte fallback can strand a syntactically valid byte
    # prefix. Remove states unable to reach the terminal, then their incoming
    # edges. Such tokens cannot complete a tool call with this tokenizer.
    reverse = [[] for _ in transitions]
    for src, edges in enumerate(transitions):
        for dst in set(edges.values()):
            reverse[dst].append(src)
    live = {1}
    pending = deque([1])
    while pending:
        for src in reverse[pending.popleft()]:
            if src not in live:
                live.add(src)
                pending.append(src)
    if 2 not in live:
        raise GrammarTooLarge('tokenizer cannot represent a complete tool call')
    live.add(0)
    remap = {old: new for new, old in enumerate(sorted(live))}
    return [{token: remap[dst] for token, dst in transitions[src].items() if dst in live}
            for src in sorted(live)]
