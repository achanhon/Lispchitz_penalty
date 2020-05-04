mkdir too_big
mkdir too_big/train too_big/test
mv build/train/images too_big/train/images
mv build/test/images too_big/test/images
mv build/train/centers too_big/train/centers
mv build/test/centers too_big/test/centers
mv build/vgg16-00b39a1b.pth too_big
rm -r build
mv too_big build

cp * build
cd build

export PATH="/home/achanhon/anaconda3/bin:$PATH"

mkdir preddiff trainpreddiff understresspreddiff

python -u train.py | tee train.txt
python -u csv_metric_under_stress.py | tee csv_metric_under_stress.txt

