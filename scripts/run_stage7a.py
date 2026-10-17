"""Explicit frozen Stage7A run or separate offline evaluation; no implicit downloads."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'evaluate'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--official-repo')
    parser.add_argument('--checkpoint')
    parser.add_argument('--review-output', help='separate offline review directory; preserve prior evaluation')
    args = parser.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    if args.operation == 'run':
        if not args.official_repo or not args.checkpoint:
            parser.error('run requires --official-repo and --checkpoint')
        from georecon.matching.session import run_session
        ok = run_session(root, out, Path(args.official_repo).resolve(), Path(args.checkpoint).resolve())
        raise SystemExit(0 if ok else 1)
    from georecon.matching.evaluate import evaluate_session
    evaluate_session(root, out, Path(args.review_output).resolve() if args.review_output else None)


if __name__ == '__main__':
    main()
