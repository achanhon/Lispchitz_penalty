import os
import sys

whereIam = os.uname()[1]
assert whereIam in [
    "calculon",
    "astroboy",
    "flexo",
    "bender",
    "ldtis706z",
]

if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratchf/"

if not os.path.exists(root + "AED"):
    print("aed not found")
    quit()

os.system("rm -rf build")
os.makedirs("build")

if whereIam == "ldtis706z":
    os.system("python3 train.py")
    os.system("python3 test.py")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python train.py")
    os.system("/d/jcastillo/anaconda3/bin/python test.py")
