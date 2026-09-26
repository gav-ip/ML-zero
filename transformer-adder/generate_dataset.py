"""Generate unique addition problems with reversed answers and 12-character rows.

Operands occupy three characters each, keeping '=' at index 7 for the
existing training mask. Spaces vary around shorter operands; answers are
right-padded to four characters. Swapped operands count as the same problem.
"""

import argparse
from pathlib import Path
import random


def generate_rows(count=100_000, seed=0):
    if not 1 <= count <= 500_500:
        raise ValueError('count must be between 1 and 500500')
    rng = random.Random(seed)
    # Sample without replacement, including every operand-length combination.
    pairs = rng.sample([(a, b) for a in range(1000) for b in range(a, 1000)], count)

    def pad_operand(value):
        digits = str(value)
        spaces = 3 - len(digits)
        left = rng.randint(0, spaces)
        return ' ' * left + digits + ' ' * (spaces - left)

    rows = []
    for a, b in pairs:
        if rng.randrange(2):
            a, b = b, a
        answer = str(a + b)[::-1]
        rows.append(f'{pad_operand(a)}+{pad_operand(b)}={answer:<4}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('addition.txt'))
    args = parser.parse_args()
    rows = generate_rows(args.count, args.seed)
    args.output.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    print(f'Wrote {len(rows):,} unique problems to {args.output}')
