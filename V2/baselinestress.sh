rm -r build
mkdir build
mkdir build/stress

/data/anaconda3/bin/python -u train.py | tee build/train.log
/data/anaconda3/bin/python -u noiselittletest.py | tee build/noiselittletest.log
