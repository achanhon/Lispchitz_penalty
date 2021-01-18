rm -r build
mkdir build

/data/anaconda3/bin/python train_AIRS_FSGM.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL1FSGMonISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL1FSGMonAIRS.txt

/data/anaconda3/bin/python train_AIRS_POST_FSGM.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL2FSGMonISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL2FSGMonAIRS.txt

/data/anaconda3/bin/python train_AIRS_POST_DFC_FSGM.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL3FSGMonISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL3FSGMonAIRS.txt

/data/anaconda3/bin/python train_AIRS.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL1onISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL1onAIRS.txt

/data/anaconda3/bin/python train_AIRS_POST.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL2onISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL2onAIRS.txt

/data/anaconda3/bin/python train_AIRS_POST_DFC.py 
/data/anaconda3/bin/python -u testISPRS_FSGM.py | tee build/ALL3onISPR.txt
/data/anaconda3/bin/python -u test_AIRS_FSGM.py | tee build/ALL3onAIRS.txt


