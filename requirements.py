import os
import subprocess
import sys


def install(package):
    subprocess.check_call(["python", "-m", "pip", "install", package])


version = float(sys.version[0:3])
if version >= 3.6:
    install("numpy")
    install("matplotlib")
    install("tf-nightly")
    install("seaborn")
    install("scikit-learn")
    install("keras")
    install("tensorflow_model_optimization")
    install("pandas")
else:
    print("Python Verion Error, Try with version 3.7.7 or higher")

exit()
