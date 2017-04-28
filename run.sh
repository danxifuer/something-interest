ver=$1
echo run ${ver}/model/model.py
mkdir -p ${ver}/log/train
mkdir -p ${ver}/log/test
python ${ver}/model/model.py > ${ver}log/train/train.log 2>&1 &
