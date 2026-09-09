"""Replay recorded token paths through the DFA to measure forced-run opportunities."""
from pathlib import Path
import collections
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from needle2.tokenizer import RefTokenizer
from needle2.grammar import compile_tool_dfa


def main():
    report = json.loads((ROOT / 'reports/decode_projection_experiment.json').read_text())
    tokenizer = RefTokenizer.from_cact(ROOT / 'artifacts/official/needle2.cact')
    result = {}
    for suite, tools_file in [('basic', 'examples/tools.json'),
                              ('expanded', 'benchmarks/expanded_tools.json')]:
        dfa = compile_tool_dfa(json.loads((ROOT / tools_file).read_text()), tokenizer)
        types, offsets = dfa.state_types, dfa.candidate_offsets
        ids, next_states = dfa.candidate_tokens, dfa.next_states
        runs, sizes, cases = [], collections.Counter(), {}
        for row in report[suite]['rows']:
            if row['id'] in cases:
                continue
            state, run = dfa.initial_state, 0
            case_runs = []
            tokens = row['results']['baseline']['token_ids']
            for index, token in enumerate(tokens):
                count = int(offsets[state + 1] - offsets[state])
                # First token comes from prefill; only later tokens enter decode_loop.
                if index:
                    sizes[count if types[state] == 1 else 'open'] += 1
                    if types[state] == 1 and count == 1:
                        run += 1
                    elif run:
                        case_runs.append(run)
                        run = 0
                matches = [j for j in range(offsets[state], offsets[state + 1])
                           if ids[j] == token]
                state = int(next_states[matches[0]]) if matches else int(dfa.fallback_next_states[state])
                if state < 0:
                    raise ValueError(f'No transition for {suite}/{row["id"]}: {token}')
            if run:
                case_runs.append(run)
            runs.extend(case_runs)
            cases[row['id']] = {'decode_steps': len(tokens) - 1, 'forced_runs': case_runs}
        result[suite] = {'candidate_counts': dict(sizes),
                         'forced_run_lengths': dict(collections.Counter(runs)), 'cases': cases}
    target = ROOT / 'reports/decode_chunk_opportunities.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({suite: {k: v for k, v in data.items() if k != 'cases'}
                      for suite, data in result.items()}, indent=2))


if __name__ == '__main__':
    main()
