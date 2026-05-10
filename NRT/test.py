# ===========================
# Path setup
# ===========================
import os
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ===========================
# Imports
# ===========================
from src.data.utils import (
    set_seed
)
SEED = 42
set_seed(SEED)


# ===========================
# Test Functions
# ===========================
def launch_nrt_tests():
    """
    Launch all NRT tests sequentially
    In each of the subfolders, run all the test.py files, which will load the datasets, train the models, deliver results, and plot the outputs. 
    """
    for subfolder in os.listdir(os.path.dirname(__file__)):
        subfolder_path = os.path.join(os.path.dirname(__file__), subfolder)
        if os.path.isdir(subfolder_path):
            test_file = os.path.join(subfolder_path, "test.py")
            if os.path.isfile(test_file):
                print(f"Running tests in {subfolder}...")
                result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("Errors:")
                    print(result.stderr)


if __name__ == "__main__":
    launch_nrt_tests()