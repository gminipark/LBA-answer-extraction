# LBA-answer-extraction

## Answer Extraction
- 응답 문장에서 정답을 추출하는 모델입니다.

### How to use
1. 다음과 같이 clone을 해주세요.
```
https://github.com/gminipark/LBA-answer-extraction.git
pip install -r requirements.txt
```
  
2. 학습된 모델[link](https://drive.google.com/drive/folders/1brledUiJ9tIgSaL_72mBSHvEZG66kYqn?usp=share_link)과 함께 아래와 같이 디렉토리를  준비해주세요,
``` 
LBA-answer-extraction/
	dataset.py
	extract.py
	requirements.txt
  model_dir/
```
3. Extract answer
```
python extract.py --model_path "model directory" --cuda 
```

## Input example 
```
[{
        "question" : "What is the color of pants that Dokyung is wearing?",
        "answer" : "The color of pants that Dokyung is wearing is gray."
}]
```
## Output example
```
gray
```

### Contact
	 - Gyu-Min Park (pgm1219@khu.ac.kr)
