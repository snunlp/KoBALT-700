#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# 에러문 일단 무시하도록 설정
logging.getLogger("transformers").setLevel(logging.ERROR)


# 인퍼런스 함수 
def kobalt_inference(text, model, tokenizer):
    messages = [
        {
            "role": "system", 
            "content": "당신은 문제를 해결하는 전문가입니다."
        },
        {
            "role": "user", 
            "content": (
                "다음 문제에 대해서 충분히 생각하고 추론하여, "
                "10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n\n"
                f"{text}\n\n"
                "답변은 반드시 다음 형식을 엄격히 지켜야 합니다: \"정답은 [정답 보기]입니다.\"로 끝나야하고, "
                "[정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다."
                "정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다."
            )
        }
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # 모델 기본 eos_token_id
    eos_token_id = tokenizer.eos_token_id
    
    # 생성 옵션 통일 버전 
    generate_kwargs = {
        "input_ids": inputs, 
        "max_new_tokens": 2048, 
        "do_sample": False
    }
    
    # eos_token_id가 있는 경우에만...명시적으로 설정
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id
    
    outputs = model.generate(**generate_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

    
# 정답 추출 함수 
def extract_answer(text):  
    # 1. 모든 "정답은 ~입니다" 구문 추출 
    pattern = r"정답은\s*(.*?)\s*입니다"
    matches = re.findall(pattern, text)
    
    valid_letters = []
    for content in matches:
        # 2. 알파벳 추출 
        letters = re.findall(r"[A-Z]", content)
        if letters:
            # 중복된 알파벳이 없는 경우에만 추가
            for letter in letters:
                if letter not in valid_letters:
                    valid_letters.append(letter)
    # 3. 복수 정답 고려
    if valid_letters:
        return ", ".join(valid_letters)
    return None


# 데이터셋으로 인퍼런스 수행 함수 
def process_kobalt(data, model_id, model, tokenizer):
    results = []
    
    for item in tqdm(data, desc=f"Inference with {model_id}"):
        id_value = item['ID']
        question = item['question']
        ground_truth = item['answer'].strip()
        main_category = item['대분류']
        sub_category = item['소분류']
        difficulty = item['난이도']
        
        model_output = kobalt_inference(question, model, tokenizer)
        predicted_answer = extract_answer(model_output)
        
        results.append({
            'ID': id_value, 
            '대분류': main_category,
            '소분류': sub_category,
            '난이도': difficulty,
            'question': question,
            'ground_truth_answer': ground_truth,
            'model_output': model_output,
            'predicted_answer': predicted_answer,
            'is_correct': ground_truth == predicted_answer,
            'model_id': model_id
        })
    return pd.DataFrame(results)


# 실행 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--input_file", type=str, default="전체_문제_목록_final_평가용.json")
    args = parser.parse_args()
    
    # 데이터 
    import json
    processed_data = json.load(open(args.input_file, encoding="utf-8"))
    

   
    
    print(f"\n===== Loading {args.model_id} =====")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto", 
        token=args.token
    )
    model.eval()
    safe_model_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', args.model_id)
    
    # 인퍼런스 실행
    df_results = process_kobalt(processed_data, args.model_id, model, tokenizer)
    
    # 결과 저장
    out_file = f"results_{safe_model_id}.csv"
    df_results.to_csv(out_file, index=False)
    
    # 정확도 계산
    accuracy = (df_results['is_correct'] == True).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\n=== ALL DONE! ===")
