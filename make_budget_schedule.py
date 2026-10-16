"""Generate a predeclared multi-level budget schedule for identification replays.

Holds have equal length and the anchor budget opens and closes every schedule, so the reversed
sequence is a valid counterbalanced partner that starts at the same warmup cap. Candidates have no
repeated neighbours, hold counts that differ by at most one between levels, and every pair of
levels adjacent at least once. A seed selects one candidate deterministically (SHA-256, not the
Python RNG), after removing excluded schedules and their reversals.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from trace_control import validate_schedule

MAX_CANDIDATES = 2_000_000


def candidates(budgets, holds, anchor):
    levels = sorted(set(budgets))
    if len(levels) < 2 or anchor not in levels:
        raise ValueError('Need at least two budget levels, including the anchor')
    if holds < len(levels) + 1:
        raise ValueError('Too few holds to visit every level between anchor holds')
    if len(levels) ** (holds - 2) > MAX_CANDIDATES:
        raise ValueError('Too many holds to enumerate; use fewer levels or longer holds')
    pairs = set(itertools.combinations(levels, 2))
    result = []
    for middle in itertools.product(levels, repeat=holds - 2):
        sequence = (anchor, *middle, anchor)
        if any(a == b for a, b in zip(sequence, sequence[1:])):
            continue
        counts = [sequence.count(level) for level in levels]
        if max(counts) - min(counts) > 1:
            continue
        if {tuple(sorted(pair)) for pair in zip(sequence, sequence[1:])} == pairs:
            result.append(sequence)
    return result


def select(options, seed, exclude=()):
    excluded = {tuple(sequence) for sequence in exclude} | {tuple(reversed(sequence)) for sequence in exclude}
    remaining = [sequence for sequence in options if sequence not in excluded]
    if not remaining:
        raise ValueError('No candidate schedule remains after exclusions')
    draw = int(hashlib.sha256(f'budget-schedule:{seed}'.encode()).hexdigest(), 16)
    return remaining[draw % len(remaining)]


def number(value):
    return int(value) if float(value).is_integer() else float(value)


def schedule(sequence, hold):
    return [dict(at_seconds=number(index * hold), budget=number(budget)) for index, budget in enumerate(sequence)]


def dumps(steps):
    lines = ',\n'.join('  ' + json.dumps(step) for step in steps)
    return f'[\n{lines}\n]\n'


def sequence_of(path, duration):
    return tuple(step['budget'] for step in validate_schedule(json.loads(Path(path).read_text()), duration))


def generate(budgets, anchor, hold, duration, seed, exclude=()):
    if hold <= 0 or duration <= 0 or not (duration / hold).is_integer():
        raise ValueError('Duration must be a positive whole number of holds')
    options = candidates([float(b) for b in budgets], int(duration / hold), float(anchor))
    steps = schedule(select(options, seed, [tuple(map(float, s)) for s in exclude]), hold)
    validate_schedule(steps, duration)
    return steps, len(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budgets', type=float, nargs='+', required=True)
    parser.add_argument('--anchor', type=float, required=True, help='Opening/closing (warmup) budget')
    parser.add_argument('--hold-seconds', type=float, required=True)
    parser.add_argument('--duration-seconds', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--exclude', type=Path, nargs='*', default=[],
                        help='Schedules that, with their reversals, must not be selected')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reverse-output', type=Path, help='Also write the reversed (counterbalanced) schedule')
    args = parser.parse_args()
    try:
        exclude = [sequence_of(path, args.duration_seconds) for path in args.exclude]
        steps, count = generate(args.budgets, args.anchor, args.hold_seconds, args.duration_seconds,
                                args.seed, exclude)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    outputs = {args.output: steps}
    if args.reverse_output:
        outputs[args.reverse_output] = schedule([step['budget'] for step in reversed(steps)], args.hold_seconds)
    for path, content in outputs.items():
        if path.exists():
            parser.error(f'{path} exists; schedules are frozen once written')
    for path, content in outputs.items():
        path.write_text(dumps(content), encoding='utf-8', newline='\n')
        print(json.dumps(dict(path=str(path), candidates=count, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              budgets=[step['budget'] for step in content])))


if __name__ == '__main__':
    main()
