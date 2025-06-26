# tests/core/test_args.py

import pytest
import sys
from core.args import parse_args #

# Define test cases with expected arguments and their parsed values
test_cases = [
    ([], {'share': False, 'server': '127.0.0.1', 'port': None, 'inbrowser': False, 'allowed_output_paths': ''}),
    (['--share'], {'share': True, 'server': '127.0.0.1', 'port': None, 'inbrowser': False, 'allowed_output_paths': ''}),
    (['--server', '0.0.0.0'], {'share': False, 'server': '0.0.0.0', 'port': None, 'inbrowser': False, 'allowed_output_paths': ''}),
    (['--port', '8888'], {'share': False, 'server': '127.0.0.1', 'port': 8888, 'inbrowser': False, 'allowed_output_paths': ''}),
    (['--inbrowser'], {'share': False, 'server': '127.0.0.1', 'port': None, 'inbrowser': True, 'allowed_output_paths': ''}),
    (['--allowed_output_paths', '/path/to/output1,/path/to/output2'], {'share': False, 'server': '127.0.0.1', 'port': None, 'inbrowser': False, 'allowed_output_paths': '/path/to/output1,/path/to/output2'}),
    (['--share', '--server', 'localhost', '--port', '9000', '--inbrowser', '--allowed_output_paths', '/mnt/data'],
     {'share': True, 'server': 'localhost', 'port': 9000, 'inbrowser': True, 'allowed_output_paths': '/mnt/data'}),
]

@pytest.mark.parametrize("input_args, expected_output", test_cases)
def test_parse_args(monkeypatch, input_args, expected_output):
    """
    Tests the parse_args function with various command-line arguments.

    Args:
        monkeypatch: pytest fixture to modify system environment variables or attributes.
        input_args: A list of strings representing command-line arguments to simulate.
        expected_output: A dictionary representing the expected parsed arguments.
    """
    # Simulate command-line arguments by modifying sys.argv
    # The first element of sys.argv is typically the script name, so we add a dummy.
    monkeypatch.setattr(sys, 'argv', ['goan_app.py'] + input_args)

    # Call the function under test
    args = parse_args() #

    # Assert that the parsed arguments match the expected output
    for key, expected_value in expected_output.items():
        assert hasattr(args, key), f"Expected argument '{key}' not found in parsed args."
        assert getattr(args, key) == expected_value, f"Mismatch for '{key}': Expected {expected_value}, got {getattr(args, key)}"

    # Additionally, ensure no unexpected attributes are present (optional, but good for strictness)
    for key in vars(args):
        if key not in expected_output:
            # The 'help' argument might be implicitly added by argparse, ignore it for this check
            # For `inbrowser`, it will be `inbrowser` instead of `inbrowser_ui` as in the problem statement
            # So, only check for unexpected keys that aren't expected to be present.
            if key not in ['help']: # Add any other keys that argparse might introduce
                assert False, f"Unexpected argument '{key}' found in parsed args."
