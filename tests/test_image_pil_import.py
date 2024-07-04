

# run export script to check that it handles missing PIL Image module and --blank-psd-preview as expected

import subprocess
import os
import shutil

def run_cmd(cmd):
    print('running', cmd)
    #pylint: disable=subprocess-run-check
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def replace_import(filepath, old, new):
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = f.read()
    assert old in data, f'not found {repr(old)} in script' 
    with open(filepath, 'w', encoding='UTF-8') as f:
        f.write(data.replace(old, new))

def check_status(cmd, should_contain_error=None):
    result = run_cmd(cmd)
    if should_contain_error:
        assert should_contain_error in result.stderr, f"Expected error not found in stderr: {result.stderr}"
    else:
        assert result.returncode == 0, result.stderr

def main():
    script = '../clip_to_psd.py'
    script2 = '../clip_to_psd.tmp.test.py'
    input_file = 'test_export_all_features.clip'
    assert os.path.isfile(input_file), repr(input_file)

    cmd = f'python -u {script} {input_file} --log-level=error -o tmp.psd'
    cmd2 = cmd.replace(script, script2)

    check_status(cmd)

    shutil.copy(script, script2)
    replace_import(script2, 'from PIL import Image', 'from PIL import ImageNonExistingModule')

    # check that script which imports non-existing module handles this as expected:
    check_status(cmd2, should_contain_error='use --blank-psd-preview')
    check_status(cmd2 + ' --blank-psd-preview')

    os.remove('tmp.psd')
    os.remove(script2)

    print("All checks passed and replacement undone.")

main()

