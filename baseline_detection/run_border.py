import os
import sys
import datetime
import random

whereIam = os.uname()[1]
assert whereIam in [
    "wdtim719z",
    "calculon",
    "astroboy",
    "flexo",
    "bender",
    "super",
    "ldtis706z",
]

if whereIam in ["wdtim719z", "super"]:
    root = "/data/"
if whereIam == "ldtis706z":
    root = "/media/achanhon/bigdata/data/"
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    root = "/scratchf/"

if not os.path.exists(root + "CIA"):
    print("run merge before")
    quit()

if not os.path.exists("build"):
    os.makedirs("build")

today = datetime.date.today()
tmp = random.randint(0, 1000)
myhash = str(today) + "_" + str(tmp)
print(myhash)

if whereIam == "super":
    os.system("/data/anaconda3/bin/python train_border.py build/" + myhash + ".pth")
    os.system("/data/anaconda3/bin/python test_border.py build/" + myhash + ".pth")
if whereIam == "wdtim719z":
    os.system("/data/anaconda3/bin/python train_border.py build/" + myhash + ".pth")
    os.system("/data/anaconda3/bin/python test_border.py build/" + myhash + ".pth")
if whereIam == "ldtis706z":
    os.system("python3 train_border.py build/" + myhash + ".pth")
    os.system("python3 test_border.py build/" + myhash + ".pth")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    os.system("/d/jcastillo/anaconda3/bin/python train_border.py build/" + myhash)
    os.system("/d/jcastillo/anaconda3/bin/python test_border.py build/" + myhash)
