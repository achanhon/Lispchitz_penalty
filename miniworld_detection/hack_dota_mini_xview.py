import os
import sys

print("only use 30 images from xview - waiting for visual review")

# keep testx{0,31}-testx{2}+trainx{2}+testx{226}
# split in 2

os.system("cp -r /data/dota/dota /media/achanhon/bigdata/data/CIA/")

os.makedirs("/media/achanhon/bigdata/data/CIA/xview")
os.makedirs("/media/achanhon/bigdata/data/CIA/xview/train")
os.makedirs("/media/achanhon/bigdata/data/CIA/xview/test")

trainimage, testimage = 0, 0
for i in range(33):
    if i % 3 == 0:
        os.system(
            "cp /data/XVIEW1/little_xview/"
            + str(i)
            + "_x.png /media/achanhon/bigdata/data/CIA/xview/test/"
            + str(testimage)
            + "_x.png"
        )
        os.system(
            "cp /data/XVIEW1/little_xview/"
            + str(i)
            + "_y.png /media/achanhon/bigdata/data/CIA/xview/test/"
            + str(testimage)
            + "_y.png"
        )
        testimage += 1
    else:
        os.system(
            "cp /data/XVIEW1/little_xview/"
            + str(i)
            + "_x.png /media/achanhon/bigdata/data/CIA/xview/train/"
            + str(trainimage)
            + "_x.png"
        )
        os.system(
            "cp /data/XVIEW1/little_xview/"
            + str(i)
            + "_y.png /media/achanhon/bigdata/data/CIA/xview/train/"
            + str(trainimage)
            + "_y.png"
        )
        trainimage += 1
