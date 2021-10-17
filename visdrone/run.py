import os
import sys

whereIam = os.uname()[1]
assert whereIam in [
    "calculon",
    "astroboy",
    "flexo",
    "bender",
]

root = "/scratchf/"
if not os.path.exists(root + "VisDrone"):
    print("VisDrone not found")
    quit()

os.system("rm -rf build")
os.makedirs("build")

if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python train.py")
    os.system("/d/jcastillo/anaconda3/bin/python test.py")
