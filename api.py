from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import uvicorn
from typing import List, Dict, Any
import os # 환경 변수 사용을 위해 추가

# --- 모델 및 토크나이저 로드 --- 

def load_model_and_tokenizer(model_size="0.5B"):
    token = os.getenv("HF_TOKEN") # 환경 변수 사용
    
    if model_size == "0.5B":
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    elif model_size == "1.5B":
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    elif model_size == "3B": # 3B 모델 옵션 추가
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B"
    else:
        raise ValueError("Unsupported model size. Choose '0.5B', '1.5B', or '3B'.") # 에러 메시지 업데이트

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        token=token, 
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on device: {device}")
    return tokenizer, model, device

# 전역 변수 또는 앱 상태로 모델/토크나이저 관리
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 환경 변수 'API_MODEL_SIZE' 에서 로드할 모델 크기 읽기 (기본값: 0.5B)
    model_size_to_load = os.getenv("API_MODEL_SIZE", "0.5B")
    print(f"Attempting to load model size: {model_size_to_load}")
    try:
        tokenizer, model, device = load_model_and_tokenizer(model_size=model_size_to_load)
        ml_models["tokenizer"] = tokenizer
        ml_models["model"] = model
        ml_models["device"] = device
    except ValueError as e:
        print(f"Error loading model: {e}")
        # 모델 로드 실패 시, 애플리케이션을 시작하지 않거나 기본 모델로 대체하는 등의 처리 가능
        # 여기서는 에러 로그만 남기고, API 호출 시 모델 없음을 알림
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        
    yield
    # 앱 종료 시 모델 정리 (선택적)
    ml_models.clear()
    print("Models cleared.")

app = FastAPI(lifespan=lifespan)

# --- 응답 정제 --- 

def clean_response(text):
    # 특수 토큰 제거
    text = re.sub(r'<\|im_start\|>assistant', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|endofturn\|>', '', text)
    text = re.sub(r'<\|stop\|>', '', text)
    text = re.sub(r'<\|pad\|>', '', text)
    text = re.sub(r'<\|unk\|>', '', text)
    text = re.sub(r'<\|mask\|>', '', text)
    text = re.sub(r'<\|sep\|>', '', text)
    text = re.sub(r'<\|cls\|>', '', text)
    
    # 혹시 남아있을 수 있는 assistant 키워드 제거
    text = re.sub(r'^\s*assistant\s*', '', text, flags=re.IGNORECASE).strip()
    
    # 빈 줄 제거 및 공백 정리
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    
    return text

# --- API 요청/응답 모델 --- 

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    user_input: str
    history: List[ChatMessage] = [] # 이전 대화 기록 (클라이언트가 관리)
    max_history: int = 5 # 유지할 최대 대화 턴 수

class ChatResponse(BaseModel):
    response: str

# --- API 엔드포인트 --- 

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    tokenizer = ml_models.get("tokenizer")
    model = ml_models.get("model")
    device = ml_models.get("device")

    if not all([tokenizer, model, device]):
        raise HTTPException(status_code=503, detail="Model is not available or failed to load. Check API server logs.")

    # 현재 사용자 입력을 history에 추가 (딕셔너리 형태로)
    current_history_dicts = [msg.dict() for msg in request.history] + [{"role": "user", "content": request.user_input}]
    
    # 시스템 메시지 추가 (history에 없으면 추가)
    if not any(msg["role"] == "system" for msg in current_history_dicts):
         current_history_dicts.insert(0, {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."})

    # 최근 대화만 유지
    if len(current_history_dicts) > request.max_history * 2 + 1: # 시스템 프롬프트 + (사용자+AI)*턴수
         # 시스템 메시지는 유지하고, 오래된 사용자/AI 메시지 제거
         system_message = [msg for msg in current_history_dicts if msg["role"] == "system"]
         recent_messages = current_history_dicts[-(request.max_history * 2):]
         recent_history_dicts = system_message + recent_messages
    
    try:
        # 채팅 템플릿 적용
        inputs = tokenizer.apply_chat_template(
            recent_history_dicts,
            return_tensors="pt", 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True
        )
        
        input_token_length = inputs["input_ids"].shape[1]
        inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

        # 응답 생성
        with torch.no_grad():
            output_ids = model.generate(
                **inputs_on_device,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 새로 생성된 토큰만 추출 및 디코딩
        new_tokens = output_ids[0, input_token_length:]
        assistant_response_raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
            
        # 응답 정제
        assistant_response = clean_response(assistant_response_raw)
        
        return ChatResponse(response=assistant_response)

    except Exception as e:
        print(f"Error during chat generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat generation.")

# --- 서버 실행 (직접 실행 시) ---
if __name__ == "__main__":
    # API 서버 실행 명령어: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    # 환경 변수 API_MODEL_SIZE 로 로드할 모델 설정 가능 (예: API_MODEL_SIZE=1.5B uvicorn ...)
    uvicorn.run("api:app", host="0.0.0.0", port=8000) 