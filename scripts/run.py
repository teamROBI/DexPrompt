"""Run a relocated script by its current or archived name; never runs on --list."""

import argparse
import runpy
import sys

from _bootstrap import GROUPS, ROOT, script_path, setup


def resolve(name):
    return script_path(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('script', nargs='?')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list:
        for path in sorted(path for group in GROUPS
                           for path in (ROOT / 'scripts' / group).glob('*.py')):
            print(path.relative_to(ROOT / 'scripts'))
        return
    if not args.script:
        parser.error('supply a script or --list')
    try:
        path = resolve(args.script)
    except ValueError as error:
        parser.error(str(error))
    setup()
    sys.argv = [str(path), *args.arguments]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
