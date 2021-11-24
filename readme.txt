가상환경 activate 
   source /USER/USER_WORKSPACE/t1/bin/activate

train.py
   --img_size   = input 이미지 사이즈
   --datapaths = 학습 데이터경로
   --batch_size = 배치사이즈
   --lr = 학습률
   --EPOCH = EPOCH
   --pretrain_path = pretrain된 weight 경로

   EPOCH별로 학습 결과물을 저장하며 weights/ 폴더에 저장됩니다.

inference.py
   --img_size   = input 이미지 사이즈
   --datapaths = Test 데이터경로
   --pretrain_path = pretrain된 weight 경로
   --batch_size = 배치사이즈
   --check_path = checkpoint 경로
   --result_name = 최종 csv파일 이름

    OUTPUT은 result 폴더에 result.csv로 저장됩니다.

리더보드 스코어 및 등수
   - 스코어 : 0.9564468400
   - 등수 : 7등	

활용모델 
   - EfficientNet-B7 
   - Mixup training 
   - data augmentation (random  resize and crop / horizontal flip)

src 내 스크립트 목록 
   - models.py   inference시 모델 호출하여 사용
   - datasetutils.py  dataset class / csv 파일저장등 utility 기능 