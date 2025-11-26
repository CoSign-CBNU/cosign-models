# cosign-models

터미널 열고

pip install opencv-python mediapipe numpy tqdm pillow


## 실행 순서
python extract_keypoints.py
-> 영상 키포인트 추출, csv, npy파일 생성됨

<img width="319" height="257" alt="image" src="https://github.com/user-attachments/assets/e9399581-8293-4fb0-9d4b-42167cdfdb03" />

python build_dataset.py
-> 영상 데이터 to 라벨(폴더이름) 매핑 json파일 생성됨


python train_rf.py
-> RF 모델 학습, 5fold 교차검증 성능 출력, dataset_out폴더에 pkl 파일 생성됨


