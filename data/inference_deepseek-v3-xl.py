# -*- coding: utf-8 -*-
from llama_cpp import Llama
import os
import re
import json
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import requests


def cpp_inference(text, model):
    prompt = (
        "다음 문제에 대해서 충분히 생각하고 추론하여, "
        "10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.\n\n"
        f"{text}\n\n"
        "답변은 반드시 다음 형식을 엄격히 지켜야 합니다: "
        "\"정답은 [정답 보기]입니다.\"로 끝나야하고, "
        "[정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.\n"
        "정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다."
    )

    messages = [
        {"role": "system", "content": "당신은 문제를 해결하는 전문가입니다."},
        {"role": "user", "content": prompt}
    ]

    response = model.create_chat_completion(
        messages = messages,
        max_tokens = 2048,
        temperature = 0.0,
        top_p = 1.0,
        top_k = 0,
        stop = ["<｜end▁of▁sentence｜>"],  # DeepSeek V3의 EOS 토큰
    )
    return response["choices"][0]["message"]['content']


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



# 데이터셋으로 인퍼런스 수행 함수
def process_kobalt(data, model_id, model, pretest):
    results = []
    for idx, item in enumerate(tqdm(data)):
        id_value = item['ID']
        question = item['question']
        ground_truth = item['answer'].strip()
        main_category = item['대분류']
        sub_category = item['소분류']
        difficulty = item['난이도']

        model_output = cpp_inference(question, model)
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

        if pretest: # 짧은 테스트용
            if idx == 0:
                print(f'\n\n예시_질문:\n{question}\n\n예시_답안:\n{model_output}\n\n')
            elif idx == 10:
                break

    return pd.DataFrame(results)


# 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--input_file", type=str, default="전체_문제_목록_final_평가용.json")
    parser.add_argument('--pretest', action='store_true', help='10문항으로 짧은 테스트 시행.')
    parser.add_argument("--n_gpu_layers", type=int, default=1, help="Number of layers to allocate per GPU")
    args = parser.parse_args()

    processed_data = json.load(open(args.input_file, encoding="utf-8"))
    model_name = args.model_path.split('/')[-2]

    print(f"\n===== Loading {model_name} =====")
    llm = Llama(
        model_path = args.model_path,
        n_ctx = 8192,
        n_gpu_layers = args.n_gpu_layers       
    )    

    print(f"\n===== Inference with {model_name} =====")

    df_results = process_kobalt(processed_data, model_name, llm, args.pretest)

    # 결과 저장
    out_file = f"results_{model_name.replace('/', '_').replace(':','_').replace('.','_')}.csv"
    df_results.to_csv(out_file, index=False)

    # 정확도 계산
    accuracy = (df_results['is_correct'] == True).mean()
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\n=== ALL DONE! ===")