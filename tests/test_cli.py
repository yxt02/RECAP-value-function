"""Public command routing and lightweight report serving contracts."""
import contextlib
import functools
from http.server import ThreadingHTTPServer
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

from recap_value import cli
from recap_value.workflows.serve import Handler

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_each_route_forwards_arguments_without_mutating_argv(self):
        for group, commands in cli.COMMANDS.items():
            for command, (module, _) in commands.items():
                workflow = Mock()
                workflow.main.return_value = 'done'
                before = list(sys.argv)
                with patch.object(cli.importlib, 'import_module', return_value=workflow) as load:
                    self.assertEqual(cli.main([group, command, '--help']), 'done')
                load.assert_called_once_with(f'recap_value.workflows.{module}')
                workflow.main.assert_called_once_with(['--help'])
                self.assertEqual(sys.argv, before)

    def test_invalid_route_and_help_do_not_load_model_dependencies(self):
        for args, status in [([], 2), (['value', 'missing'], 2), (['--help'], 0), (['report', '--help'], 0)]:
            with patch.object(cli.importlib, 'import_module') as load, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    cli.main(args)
                self.assertEqual(exc.exception.code, status)
                load.assert_not_called()

    def test_absolute_entrypoint_runs_outside_project(self):
        result = subprocess.run([sys.executable, str(ROOT/'scripts/recap.py'), 'report', 'serve', '--help'],
                                cwd='/tmp', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--port', result.stdout)

    def test_workflow_failure_is_not_reported_as_success(self):
        workflow = Mock()
        workflow.main.side_effect = ValueError('invalid input')
        with patch.object(cli.importlib, 'import_module', return_value=workflow):
            with self.assertRaisesRegex(ValueError, 'invalid input'):
                cli.main(['report', 'check', 'missing'])

    def test_report_video_byte_range(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'clip.webm').write_bytes(b'0123456789')
            server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = Request(f'http://127.0.0.1:{server.server_port}/clip.webm', headers={'Range': 'bytes=2-5'})
                with urlopen(request) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers['Content-Range'], 'bytes 2-5/10')
                    self.assertEqual(response.read(), b'2345')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
