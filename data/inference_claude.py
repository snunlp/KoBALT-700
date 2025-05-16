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

# 연구실 토큰 
import anthropic
client = anthropic.Anthropic(
    api_key="CLAUDE_API_KEY", 
)


#2) G2P 인퍼런스 함수
def gcp_inference(text, model):
    messages = [
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
    message = client.messages.create(
        model=model,
        max_tokens=2048,  
        temperature=0.0,
        system = "당신은 문제를 해결하는 전문가입니다.", 
        messages=messages
    )
    if isinstance(message.content, list):
        response_text = " ".join([block.text for block in message.content if hasattr(block, "text")])
    else:
        response_text = message.content
        
    return response_text.strip()


# 정답 추출 함수 
def extract_answer(text): 
    # None이거나 문자열이 아닐 경우 처리 (오류 대응)
    if text is None or not isinstance(text, str):
        print(f"Warning: Invalid text type - {type(text)}")
        return None
    
    # "정답은 ~입니다" 구문 추출 (마지막 일치 항목만)
    pattern = r"정답은\s*(.*?)\s*입니다"
    matches = re.findall(pattern, text)
    
    if not matches:
        return None
    
    # 마지막 매치만 사용
    last_match = matches[-1]
    
    # A~J 알파벳만 추출 (순서 유지 & 중복 제거)
    letters = re.findall(r"[A-J]", last_match)
    unique_letters = []
    for letter in letters:
        if letter not in unique_letters:
            unique_letters.append(letter)
    
    # 추출된 알파벳이 있으면 반환
    if unique_letters:
        return ", ".join(unique_letters)
    
    return None


#3) 데이터셋으로 인퍼런스 수행 함수 
def process_g2p(data, model_id, model):
    results = []
    
    for item in tqdm(data, desc=f"Inference with {model_id}"):
        id_value = item['ID']
        question = item['question']
        ground_truth = item['answer'].strip()
        main_category = item['대분류']
        sub_category = item['소분류']
        difficulty = item['난이도']
        
        model_output = gcp_inference(question, model)
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



#5) 실행 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="claude-3-7-sonnet-20250219")
    parser.add_argument("--input_file", type=str, default="전체_문제_목록_final_평가용.json")
    args = parser.parse_args()

    # 데이터 
    import json
    processed_data = json.load(open(args.input_file, encoding="utf-8"))

    print(f"\n===== Loading {args.model_id} =====")

    safe_model_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', args.model_id)

    # 인퍼런스 실행
    df_results = process_g2p(processed_data, args.model_id, args.model_id)
    
    # 결과 저장
    out_file = f"results_{safe_model_id}.csv"
    df_results.to_csv(out_file, index=False)
    
    # 정확도 계산
    accuracy = (df_results['is_correct'] == True).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\n=== ALL DONE! ===")