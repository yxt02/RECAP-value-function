"""Lazy command dispatch: reporting does not require the training environment."""
import argparse
import importlib
import logging
import sys

COMMANDS = {
    'data': {
        'prepare': ('data', 'Audit/import data, write return labels and episode splits'),
        'survey': ('pairing', 'Compare same-session trajectory pairing feasibility'),
    },
    'value': {
        'train': ('training', 'Train the scalar value model; --prepare-cache builds features only'),
        'benchmark': ('benchmark', 'Measure GPU, loader or cached-head throughput'),
        'check-cache': ('cache_check', 'Check cached/online features and checkpoint reload'),
    },
    'report': {
        'evaluate': ('evaluation', 'Run fixed-checkpoint full-test evaluation'),
        'check': ('replay', 'Independently replay saved evaluation predictions'),
        'render': ('render', 'Build plots and HTML from saved evaluation arrays'),
        'videos': ('videos', 'Build browser-compatible previews from linked source videos'),
        'serve': ('serve', 'Serve a report on localhost with video seeking'),
    },
}


def main(argv=None):
    """Run one workflow with an explicit argument list, or use the process arguments.

    Returns the selected workflow's result. Argument errors exit with status 2;
    computation errors propagate rather than being reported as success.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog='recap', description=__doc__,
        epilog='Usage: recap GROUP COMMAND --help. Relative paths use the project root. See docs/scripts.md.')
    groups = parser.add_subparsers(dest='group', required=True)
    for group, commands in COMMANDS.items():
        section = groups.add_parser(group)
        actions = section.add_subparsers(dest='command', required=True)
        for command, (_, description) in commands.items():
            actions.add_parser(command, help=description, add_help=False)
    # Parse only the route; the workflow owns its options, including --help.
    route = parser.parse_args(argv[:2])
    module_name = COMMANDS[route.group][route.command][0]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    module = importlib.import_module(f'recap_value.workflows.{module_name}')
    return module.main(argv[2:])
