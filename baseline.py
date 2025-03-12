import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForPreTraining, LlavaNextProcessor, AutoTokenizer
from PIL import Image
import os
import transformers
import json
import numpy as np
import cv2
import pyarrow as pa
import pyarrow.parquet as pq

prompt=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('lba_test')
image_dir = "image dir path

df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

def QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    switch = 0
    # target box 좌표 설정
    locations = sceneGraphs[str(image_id)]['objects'][str(entity_id)]
    width, height = sceneGraphs[str(image_id)]['width'], sceneGraphs[str(image_id)]['height']
    x, y, w, h = locations['x'], locations['y'], locations['w'], locations['h']
    x1, x2 = round(x/width,4), round((x+w)/width,4)
    y1, y2 = round(y/height,4), round((y+h)/height,4)
    tar_coordinate = (x, y, w, h)
    # 다른 entity box 좌표 설정
    keys = sceneGraphs[str(image_id)]['objects'].keys()
    objects=sceneGraphs[str(image_id)]['objects']
    i = 0
    coordinates = ""
    for k in keys:
        if k != str(entity_id) and objects[k]['name'] == locations['name']:
            i+=1
            v = objects[k]
            x1_ = round(v['x']/width , 4)
            y1_ = round(v['y']/height , 4)
            x2_ = round((v['x']+v['w'])/width , 4)
            y2_ = round((v['y']+v['h'])/height , 4)
            coordinates += f"The coordinates of non-target {v['name']+ str(i)} not in the question : ({x1_}, {y1_}, {x2_}, {y2_})\n"

    # 1번 질문
    text1 = f"""[INST] <image>
USER: {ambiguous_question}
ASSISTANT: [/INST]"""
    prompt = text1
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        # 이미지 로드 및 RGB로 변환
        image = Image.open(image_path).convert("RGB")
        # 프로세서 입력값 생성 (이미지를 PIL 이미지 객체가 아닌 텐서로 변환)
        inputs = processor(images=[image], text=text1, return_tensors="pt").to(device)
        image.save('img.jpg')
        # 모델 출력 생성
        generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        start=generated_text.find("[/INST]")
        generated_text = generated_text[start+8:]
        return generated_text, text1, ambiguous_question_answer

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""

import time
import datetime
total_start_time = time.time()
output_file = 'baseline.txt'
correct_li=[]
wrong_li=[]
print("테스트 시작")
score = 0
import shutil
with open(output_file,'w', encoding='utf-8') as f :
    f.write(f"prompt: {prompt}\n")
    f.write("테스트 시작")
    for i in range(1000):
        image_id = df[i]['image_id']
        ambiguous_question = df[i]['ambiguous_question']
        ambiguous_entity = df[i]['ambiguous_entity']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = df[i]['ambiguous_question_answer']
        target_question = df[i]['additional_question']
        no1= df[i]['no1']


        if os.path.exists(image_path):
            output, prompt, answer = QA(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
            output = output.lower()[:-1] ## 소문자로 맞춰주고 항상 출력 뒤에 빈 스트링 한 칸 차지하고 있어서 빼줌
            if answer in output :
                f.write("\ncorrect")
                score += 1
                correct_li.append(i+1)
            else :
                f.write("\nwrong")
                wrong_li.append(i+1)
            print()
            print(f"{i+1}th")
            print(ambiguous_question)
            print("output",output)
            print("answer",answer)
            print("score",score)
            f.write(f"\n{i+1}th")
            f.write(f"\n이미지 ID: {image_id}")
            f.write(f"\n엔티티 ID: {entity_id}")
            f.write(f"\n입력 질문:  {ambiguous_question}")
            f.write(f"\n출력 답: {output}")
            f.write(f"\n실제 답: {answer}")
            f.write(f"\n정답 질문: {target_question}")
            f.write(f"\n점수 : {score}")
            f.write('\n')
        else:
            print(f"이미지 {image_path}이(가) 존재하지 않습니다.")
            f.write(f"\n이미지 {image_path}이(가) 존재하지 않습니다.")

    total_elapsed_time = time.time() - total_start_time
    tot=str(datetime.timedelta(seconds=total_elapsed_time)).split(".")[0]
    f.write("\n========================================================")
    f.write(f"\n총 처리 시간 : {tot}, 최종 점수 : {score}")
    f.write(f"\nC:{correct_li}")
    f.write(f"\nW: {wrong_li}")