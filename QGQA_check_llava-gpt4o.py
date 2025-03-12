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
from openai import OpenAI
import base64
import requests
import time

#-------------

# yes cut
# gpt local image used
# QG : context, original question, ambiguous entity
# GPT : a generated question, ambiguous entity
# QA : context, original question

#-------------

gpt_prompt = ""
final_prompt = ""
prompt = ""

CLIENT_ID = "client id"
my_api_key="my api key"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('train_sceneGraphs.json', 'r', encoding='utf-8') as file :
    sceneGraphs = json.load(file)

from datasets import load_from_disk
dataset = load_from_disk('lba_test')
image_dir = "ambiguous_images"
boxed_img_dir = "300_boxes"


df = dataset

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = AutoModelForPreTraining.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8') 

# 결과 출력


def CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    headers = {
        'Authorization': f'Client-ID {CLIENT_ID}'
    }   

    boxed_img_path = os.path.join(boxed_img_dir, f"{image_id}.jpg")
    base64_image = encode_image(boxed_img_path)

    # 다른 entity box 좌표 설정
    keys = sceneGraphs[str(image_id)]['objects'].keys()
    objects=sceneGraphs[str(image_id)]['objects']
    i = 0
    q_count = 0
    context = ""
    switch = 0
    first_question = f"""[INST] <image>
USER : If you are uncertain or cannot determine the answer, respond with "I'm not sure." {ambiguous_question}
ASSISTANT : [/INST]"""
    
    first_output = generate_output(image_id, ambiguous_question, ambiguous_entity, first_question, entity_id, switch)[0]
    if 'not sure' in first_output.lower() :
        pass
    else : 
        q_count = 5
    while q_count < 3 :
        switch = 0  
        # 1번 질문
        additional_question2 = f"""[INST] <image>
You receive an original question that has an ambiguous entity difficult to specify.
The task is to use the provided context, and generate a new yes/no question asking about the {ambiguous_entity} referred to the original question.
In the generated question, you must distinguish the {ambiguous_entity} from the other '{ambiguous_entity}' in the image.
By answering the new yes/no question, one can identify which '{ambiguous_entity}' is referred to the original question.
When generating a new question, you can use the attributes of the {ambiguous_entity} or relative location to the other {ambiguous_entity} in the image.
If you created a question, answer it without saying it directly. If the answer is 'yes', show the question you created, and if it is 'no', do it again.
Make sure that the yes/no question allows you to distinguish {ambiguous_entity} from the other {ambiguous_entity}.

The original question: '{ambiguous_question}'
The ambiguous entity: '{ambiguous_entity}'

{context}
ASSISTANT: [/INST]"""
        prompt = additional_question2
        generated_question = generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question2, entity_id, switch)
        context += f'USER: {generated_question[0]}\n'


        # 대답 과정 (일단 지금은 사람이 대답/모델이 대답도 시켜볼 예정)
        switch = 'check'
#         additional_question3 = f"""[INST] <image>
# Keep in mind that the question is asking {ambiguous_entity} in the given coordinates
# The coordinates of the {ambiguous_entity} in the question : ({x1}, {y1}, {x2}, {y2})
# {context} Just answer yes or no.
# [/INST]"""

        gpt_prompt  = f"""You are an AI trained to support visual recognition tasks. Your job is to look at the image with s red bounding box around specific object and answer the question about it. You have to analyze the object inside the red bbox and user tells you the name of the object in the input text. Based on this object, you answer the question. 
Your goal
You can focus only on the object within the red bbox. Focus only on objects and properties when answering a question, without mentioning the bounding boxes in the image. 
Don't generate any new question.

When asked about direction or relative position, think twice and answer correctly.
If the question includes 'or', answer which one of the two options is correct.
If the question is wrong, you should correct it.
If the question is asking about the other {ambiguous_entity} which is not within the red bbox, you should correct it.

In addition, the answer should be simple and concise."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {my_api_key}"
        }
        payload = {
            "model": "gpt-4o",  # 이미지 처리가 가능한 모델 사용
            "messages": [
                {
                    "role": "system",
                    "content": gpt_prompt               },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Here is a question about the given image. The object in question is "{ambiguous_entity}".
Original question : {ambiguous_question}

The "{ambiguous_entity}" in the sub-question is in the red box in the image. Answer the sub-question. You can also give additional hints, if necessary, to help answer the original question in a sentence.
sub-question : {generated_question}
Answer : """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature":0,
            "top_p":0.1
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
        case_switch = response.json()['choices'][0]['message']['content'].lower()

        # yes인 경우
        if 'yes' in case_switch :
            context += f'ASSISTANT: Yes\n'
            break
        # no인 경우
        else : 
            context += f'ASSISTANT: {case_switch}\n'
        q_count += 1 
    additional_question_final = f"""[INST] <image>
{context}

USER: {ambiguous_question} Answer in one or two words.
ASSISTANT: [/INST]"""
    final_prompt = additional_question_final
    final_output = generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question_final, entity_id, switch)
    return first_output, final_output[0], final_output[1], ambiguous_question_answer, context


def generate_output(image_id, ambiguous_question, ambiguous_entity, additional_question, entity_id, switch):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    try:
        # 이미지 로드 및 RGB로 변환
        image = Image.open(image_path)


        # 프로세서 입력값 생성 (이미지를 PIL 이미지 객체가 아닌 텐서로 변환)
        inputs = processor(images=[image], text=additional_question, return_tensors="pt").to(device)

        image.save('img.jpg')
        # 모델 출력 생성
        generated_ids = model.generate(**inputs, max_new_tokens=80, pad_token_id=tokenizer.eos_token_id)
        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)
        start = generated_text.find("[/INST]")
        generated_text = generated_text[start+8:]
        return generated_text, additional_question

    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        return ""


import time
import datetime
total_start_time = time.time()
output_file = 'output_file.txt'
correct_li=[]
wrong_li=[]
print("테스트 시작")
score = 0
import shutil
with open(output_file,'w', encoding='utf-8') as f :
    f.write(f"prompt: {prompt}\n")
    f.write(f"gpt: {gpt_prompt}\n")
    f.write(f"final_prompt: {final_prompt}\n")
    f.write("테스트 시작")
    for i in range(300):
        image_id = df[i]['image_id']
        ambiguous_question = df[i]['ambiguous_question']
        ambiguous_entity = df[i]['ambiguous_entity']
        entity_id = df[i]['entity_id']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        ambiguous_question_answer = df[i]['ambiguous_question_answer']
        target_question = df[i]['additional_question']
        no1= df[i]['no1']


        if os.path.exists(image_path):
            switch, output, prompt, answer, contexts = CoT(image_id, ambiguous_question, ambiguous_entity, entity_id, ambiguous_question_answer)
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
            print(contexts[:-1])
            print(ambiguous_question)
            print("output",output)
            print("answer",answer)
            print("switch",switch)
            print("score",score)
            f.write(f"\n{i+1}th")
            f.write(f"\n이미지 ID: {image_id}")
            f.write(f"\n엔티티 ID: {entity_id}")
            f.write(f"\n입력 질문:  {ambiguous_question}")
            f.write(f"\n스위치: {switch}")
            f.write(f"\n출력 답: {output}")
            f.write(f"\n실제 답: {answer}")
            f.write(f"\n정답 질문: {target_question}")
            f.write(f"\n{contexts}")
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