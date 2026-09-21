"""Lazy command routing for the local RECAP workflows.

The report commands must not import torch, so each module is imported only when its
command is selected. Relative paths are resolved against the project root.
"""
import argparse
import importlib
import logging
import sys
from pathlib import Path

# {group: {command: (module under scripts/, one-line description)}}
COMMANDS = {
    'data': {
        'prepare': ('data_workflow', '审计/导入数据，写入回报标签和轨迹划分'),
    },
    'value': {
        'train': ('training_workflow', '训练标量价值模型；--prepare-cache 只构建特征缓存'),
        'benchmark': ('training_workflow', '测量 GPU、数据加载器或缓存头的吞吐量'),
        'check-cache': ('training_workflow', '检查缓存/在线特征和 checkpoint 重载的一致性'),
    },
    'report': {
        'evaluate': ('evaluation_workflow', '使用固定 checkpoint 进行全测试集评估'),
        'render': ('evaluation_workflow', '从评估结果生成图表和 HTML 报告'),
    },
}


def function_name(command):
    """Workflow entry point for a command: `check-cache` -> `main_check_cache`."""
    return 'main_' + command.replace('-', '_')


def main(argv=None):
    """Route to the workflow for one command and return its result.

    Argument errors exit with status 2; workflow errors propagate instead of being
    reported as success.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog='recap',
        description='RECAP 价值函数统一命令入口',
        epilog='使用方法: recap GROUP COMMAND --help。相对路径使用项目根目录。详见 docs/scripts.md。'
    )
    groups = parser.add_subparsers(dest='group', required=True)
    for group, commands in COMMANDS.items():
        actions = groups.add_parser(group).add_subparsers(dest='command', required=True)
        for command, (_, description) in commands.items():
            actions.add_parser(command, help=description, add_help=False)

    # Parse the route only; the workflow parses its own flags.
    route = parser.parse_args(argv[:2])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    scripts_dir = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    module = importlib.import_module(COMMANDS[route.group][route.command][0])
    return getattr(module, function_name(route.command))(argv[2:])
