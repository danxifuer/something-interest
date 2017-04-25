ver=$1
mkdir -p log/train
mkdir -p log/test
echo run {ver}/model/model.py
python ${ver}/model/model.py > log/train/train.log 2>&1 &
