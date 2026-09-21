"""Public contract tests for the independent scripts. Run with unittest discover -s tests -v."""
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

# Scripts split by location
SCRIPTS_DIR = {
    'prepare_data.py': 'scripts',
    'train.py': 'scripts',
    'benchmark.py': 'tests',
    'check_cache.py': 'tests',
    'evaluate.py': 'tests',
    'render.py': 'tests',
}


class ScriptTests(unittest.TestCase):
    """Test that each script shows help without crashing."""

    def test_help_flag(self):
        """Each script should show help and exit 0."""
        for script, folder in SCRIPTS_DIR.items():
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / folder / script), '--help'],
                    cwd=str(ROOT),
                    text=True,
                    capture_output=True
                )
                self.assertEqual(result.returncode, 0,
                                 f'{script} --help failed: {result.stderr}')
                self.assertTrue(len(result.stdout) > 0,
                                f'{script} --help produced no output')

    def test_benchmark_requires_mode(self):
        """benchmark.py should fail without --mode."""
        result = subprocess.run(
            [sys.executable, str(ROOT / 'tests' / 'benchmark.py')],
            cwd=str(ROOT),
            text=True,
            capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_render_requires_output(self):
        """render.py should fail without output argument."""
        result = subprocess.run(
            [sys.executable, str(ROOT / 'tests' / 'render.py')],
            cwd=str(ROOT),
            text=True,
            capture_output=True
        )
        self.assertNotEqual(result.returncode, 0)

    def test_prepare_data_no_args_shows_help(self):
        """prepare_data.py without args should show help."""
        result = subprocess.run(
            [sys.executable, str(ROOT / 'scripts' / 'prepare_data.py')],
            cwd=str(ROOT),
            text=True,
            capture_output=True
        )
        self.assertEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
