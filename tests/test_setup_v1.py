import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('setup_v1', Path(__file__).parents[1] / 'scripts/setup_v1.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_descendant_finds_ninja_without_activation(self):
        import os
        import subprocess
        import sys
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / '.venv'
            bindir = venv / 'bin'
            bindir.mkdir(parents=True)
            ninja = bindir / 'ninja'
            ninja.write_text('#!/bin/sh\nprintf "venv-ninja\\n"\n')
            ninja.chmod(0o755)
            with patch.dict(os.environ, {'PATH': '/usr/bin:/bin', 'VIRTUAL_ENV': '/other',
                                         'PYTHONHOME': '/invalid'}):
                env = setup.venv_environment(venv)
                self.assertEqual(os.environ['VIRTUAL_ENV'], '/other')
                self.assertNotIn('PYTHONHOME', env)
                result = subprocess.run([sys.executable, '-c',
                    'import subprocess; subprocess.run(["ninja", "--version"], check=True)'],
                    env=env, check=True, capture_output=True, text=True)
                self.assertEqual(result.stdout.strip(), 'venv-ninja')

    def test_setup_passes_venv_environment_to_server(self):
        import argparse
        import tempfile
        from unittest.mock import patch, Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/python').touch()
            proc = Mock()
            proc.wait.return_value = 0
            proc.poll.return_value = 0
            def run_command(argv, **kwargs):
                if argv[0] == 'nvidia-smi':
                    return Mock(stdout='NVIDIA mock')
                self.assertEqual(kwargs['env']['VIRTUAL_ENV'], str(root / '.venv'))
                self.assertTrue(kwargs['env']['PATH'].startswith(str(root / '.venv/bin')))
                return Mock(stdout='{}')
            with patch.object(setup.platform, 'platform', return_value='Linux'), \
                 patch.object(setup, 'ROOT', root), patch.object(setup, 'check_port'), \
                 patch.object(setup.shutil, 'which', return_value=str(root / '.venv/bin/ninja')), \
                 patch.object(setup.subprocess, 'run', side_effect=run_command), \
                 patch.object(setup.subprocess, 'Popen', return_value=proc) as spawn, \
                 patch.object(setup, 'wait_ready', return_value={}):
                setup.run(argparse.Namespace(skip_install=False, startup_timeout=10))
            env = spawn.call_args.kwargs['env']
            self.assertEqual(env['VIRTUAL_ENV'], str(root / '.venv'))
            self.assertTrue(env['PATH'].startswith(str(root / '.venv/bin')))

    def test_launch_pins_context(self):
        args = setup.command(Path('/tmp/env/bin/python'))
        self.assertEqual(args[args.index('--max-model-len') + 1], '8192')
        self.assertIn(setup.MODEL, args)
        self.assertEqual(args[args.index('--host') + 1], '127.0.0.1')

    def test_rejects_wrong_or_missing_context(self):
        for value in (None, 262144, 4096):
            with self.assertRaises(RuntimeError):
                setup.check_models({'data': [{'id': setup.MODEL, 'max_model_len': value}]})
        with self.assertRaises(RuntimeError):
            setup.check_models({'data': [{'id': 'other', 'max_model_len': 8192}]})

    def test_waits_until_ready(self):
        class Process:
            def poll(self): return None
        calls = []
        def fetch():
            calls.append(1)
            if len(calls) == 1:
                raise OSError('loading')
            return {'data': [{'id': setup.MODEL, 'max_model_len': 8192}]}
        result = setup.wait_ready(Process(), 1, fetch, sleep=lambda _: None)
        self.assertEqual(len(calls), 2)
        setup.check_models(result)

    def test_port_reuse_and_live_listener(self):
        import socket
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        listener.listen()
        with self.assertRaises(OSError):
            setup.check_port(port)
        client = socket.create_connection(('127.0.0.1', port))
        connection, _ = listener.accept()
        connection.close()  # Server initiates close, leaving TIME_WAIT.
        client.recv(1)
        client.close()
        listener.close()
        setup.check_port(port)

    def test_exit_and_timeout(self):
        class Dead:
            def poll(self): return 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            setup.wait_ready(Dead(), 1, lambda: {})
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            setup.wait_ready(Dead(), 0, lambda: {})


if __name__ == '__main__':
    unittest.main()
