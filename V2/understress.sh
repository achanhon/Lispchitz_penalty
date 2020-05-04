rm -r build
mkdir build
mkdir build/stress

/data/anaconda3/bin/python -u train.py | tee build/train.log
/data/anaconda3/bin/python -u noisetest_littleimage.py | tee build/noisetest_littleimage.log
