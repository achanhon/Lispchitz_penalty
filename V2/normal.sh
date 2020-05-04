rm -r build
mkdir build

/data/anaconda3/bin/python -u train.py | tee build/train.log
/data/anaconda3/bin/python -u fasttest.py | tee build/fasttest.log
/data/anaconda3/bin/python -u test.py | tee build/test.log
