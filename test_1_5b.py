from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import re
import os

# CUDA 확인
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 디바이스 수: {torch.cuda.device_count()}")
    print(f"현재 CUDA 디바이스: {torch.cuda.current_device()}")
    print(f"CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")

# 토큰 및 모델 설정
token = os.getenv("HF_TOKEN") # 환경 변수 사용
# 1.5B 모델 사용
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# 토크나이저와 모델 로드
print("모델 로딩 중... 잠시만 기다려주세요")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)

# 메모리 절약을 위해 float16 사용
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    token=token, 
    trust_remote_code=True,
    torch_dtype=torch.float16  # 반정밀도(16비트)로 로드
)

# GPU 사용 가능하면 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"모델이 로드된 디바이스: {device}")

print("\n모델 로딩 완료! 대화를 시작합니다. 종료하려면 '종료'를 입력하세요.\n")

# 시스템 메시지 설정
chat_history = [{"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."}]

# 응답 정제 함수 정의 (스트리밍에서는 사용하지 않지만, 히스토리 저장을 위해 남겨둠)
def clean_response(text):
    # 특수 토큰 제거
    text = re.sub(r'<\|im_start\|>assistant', '', text) # 시작 부분 assistant 태그 제거
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|endofturn\|>', '', text)
    text = re.sub(r'<\|stop\|>', '', text)
    text = re.sub(r'<\|pad\|>', '', text)
    text = re.sub(r'<\|unk\|>', '', text)
    text = re.sub(r'<\|mask\|>', '', text)
    text = re.sub(r'<\|sep\|>', '', text)
    text = re.sub(r'<\|cls\|>', '', text)
    
    # 혹시 남아있을 수 있는 assistant 키워드 제거 (대소문자 무시)
    text = re.sub(r'^\s*assistant\s*', '', text, flags=re.IGNORECASE).strip()
    
    # 빈 줄 제거
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

# 대화 루프
while True:
    # 사용자 입력 받기
    user_input = input("나: ")
    
    # 종료 조건
    if user_input.lower() == '종료':
        print("대화를 종료합니다.")
        break
    
    # 사용자 메시지 추가
    chat_history.append({"role": "user", "content": user_input})
    
    # 최근 대화만 사용 (메모리 절약 및 응답 품질 향상)
    recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history.copy()
    
    # 채팅 템플릿 적용
    inputs = tokenizer.apply_chat_template(
        recent_history, 
        return_tensors="pt", 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True  # 어텐션 마스크 포함
    )
    
    # 입력 토큰 길이 저장
    input_token_length = inputs["input_ids"].shape[1]
    
    # 입력을 모델과 같은 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 스트리머 설정
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 응답 생성 (스트리밍)
    print("\nAI: ", end="") # AI: 프롬프트 출력 후 줄바꿈 없이 대기
    output_ids = model.generate(
        **inputs, # 딕셔너리 언패킹으로 인자 전달
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,  # 샘플링 활성화
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id # pad_token_id 설정 추가
    )
    print() # 스트리밍 완료 후 줄바꿈
    
    # 히스토리 저장을 위한 전체 응답 생성 (스트리밍 완료 후)
    new_tokens = output_ids[0, input_token_length:]
    assistant_response_raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
    assistant_response = clean_response(assistant_response_raw)
    
    # 응답을 대화 기록에 추가 (정제된 응답으로)
    chat_history.append({"role": "assistant", "content": assistant_response}) 