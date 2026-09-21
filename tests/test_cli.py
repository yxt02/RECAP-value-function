"""Public contract tests for the command router. Run with unittest discover -s tests -v."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from submodules import cli

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_each_route_forwards_arguments_without_mutating_argv(self):
        """Every route imports its module, calls its entry point with argv[2:] and leaves sys.argv alone."""
        for group, commands in cli.COMMANDS.items():
            for command, (module_name, _) in commands.items():
                workflow = Mock()
                func_mock = Mock(return_value='done')
                setattr(workflow, cli.function_name(command), func_mock)

                before = list(sys.argv)
                with patch.object(cli.importlib, 'import_module', return_value=workflow) as load:
                    self.assertEqual(cli.main([group, command, '--help']), 'done')
                load.assert_called_once_with(module_name)
                func_mock.assert_called_once_with(['--help'])
                self.assertEqual(sys.argv, before)

    def test_invalid_route_and_help_do_not_load_model_dependencies(self):
        for args, status in [([], 2), (['value', 'missing'], 2), (['--help'], 0), (['report', '--help'], 0)]:
            with patch.object(cli.importlib, 'import_module') as load, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    cli.main(args)
                self.assertEqual(exc.exception.code, status)
                load.assert_not_called()

    def test_absolute_entrypoint_runs_outside_project(self):
        """The absolute-path entry point works from another cwd and reaches the workflow's own parser."""
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'value_function.py'), 'report', 'render', '--help'],
            cwd='/tmp',
            text=True,
            capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('evaluation directory path', result.stdout)

    def test_workflow_failure_is_not_reported_as_success(self):
        workflow = Mock()
        func_mock = Mock(side_effect=ValueError('invalid input'))
        setattr(workflow, cli.function_name('render'), func_mock)

        with patch.object(cli.importlib, 'import_module', return_value=workflow):
            with self.assertRaisesRegex(ValueError, 'invalid input'):
                cli.main(['report', 'render', 'missing'])


if __name__ == '__main__':
    unittest.main()
