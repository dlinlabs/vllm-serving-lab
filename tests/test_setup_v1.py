import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('setup_v1', Path(__file__).parents[1] / 'scripts/setup_v1.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
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
