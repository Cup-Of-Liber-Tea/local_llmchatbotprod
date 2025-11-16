import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from dotenv import load_dotenv # dotenv 사용 방식으로 복귀

# 키 파일 경로를 상수로 정의
# KEYFILE_PATH = "/code/app/.keyfile" # 이 방식 대신 dotenv 사용

model = None
tokenizer = None
device = None
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

# 모듈 레벨에서 사용할 토큰 변수 초기화
loaded_token = None

# dotenv를 사용하여 app/.keyfile 로드 시도 (verbose 추가)
# 이 코드는 모듈 로드 시점에 실행됨
keyfile_full_path = os.path.join(os.path.dirname(__file__), '.keyfile')
if os.path.exists(keyfile_full_path):
    print(f"Found keyfile at: {keyfile_full_path}. Attempting to load...")
    # override=True: 시스템 환경변수보다 .keyfile 우선
    # verbose=True: 로드된 변수 확인
    load_dotenv(dotenv_path=keyfile_full_path, override=True, verbose=True) 
    # 로드 후 HYPERCLOVAX_API_KEY 값을 읽어 전역 변수에 저장
    loaded_token = os.getenv("HYPERCLOVAX_API_KEY")
    print(f"HYPERCLOVAX_API_KEY from .keyfile: {'Loaded' if loaded_token else 'Not found'}")
else:
    print(f"Warning: dotenv key file '{keyfile_full_path}' not found.")

def load_model_and_tokenizer():
    """모델과 토크나이저를 로드하고 전역 변수에 저장합니다."""
    # 전역 변수 사용 명시
    global model, tokenizer, device, loaded_token

    try:
        # 함수 내에서 load_dotenv() 호출 제거
        # 대신 모듈 레벨에서 로드된 loaded_token 사용
        
        # 모듈 레벨에서 로드된 토큰 확인
        print(f"Token check inside load_model_and_tokenizer: {'Available' if loaded_token else 'Unavailable'}")
        
        if not loaded_token:
            # 시스템 환경변수에서 다시 한번 확인 (선택적)
            loaded_token = os.getenv("HYPERCLOVAX_API_KEY")
            print(f"Re-checking system env var HYPERCLOVAX_API_KEY: {'Found' if loaded_token else 'Not found'}")
            if not loaded_token:
                raise ValueError("HyperCLOVAX API 키를 .keyfile 또는 환경 변수에서 찾을 수 없습니다.")
        
        # 디바이스 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 토크나이저 로드
        print("Loading tokenizer...")
        # 로드 시 loaded_token 사용
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=loaded_token, 
            trust_remote_code=True
        )
        
        # 모델 로드
        print("Loading model...")
        # 로드 시 loaded_token 사용
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=loaded_token,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        ).to(device)
        
        print("Model and tokenizer loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def clean_response(text):
    """모델 응답에서 불필요한 특수 토큰, 시스템 프롬프트, 과거 대화 내용을 제거합니다."""
    # 모든 input/output 패턴에서 마지막 assistant 응답만 추출
    
    # user/assistant 패턴 찾기 (과거 대화 내용 제거)
    if "user" in text and "assistant" in text:
        # 마지막 assistant 이후의 텍스트만 추출
        parts = text.split("assistant")
        if len(parts) > 1:
            # 마지막 assistant 부분만 사용
            text = "assistant" + parts[-1]
    
    # 마커를 기준으로 응답 내용 추출
    # system 마커 제거 (시스템 프롬프트 전체 제거)
    if "<|im_start|>system" in text and "<|im_start|>assistant" in text:
        # system 마커와 assistant 마커 사이의 모든 내용 제거
        system_start = text.find("<|im_start|>system")
        assistant_start = text.find("<|im_start|>assistant", system_start)
        if system_start >= 0 and assistant_start > system_start:
            # system 부분 제거하고 assistant 부분부터 시작
            text = text[assistant_start:]
    
    # 여전히 assistant 마커가 있으면 그 이후 내용만 추출
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    
    # 기타 특수 토큰 제거
    text = re.sub(r'<\|im_start\|>assistant', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|endofturn\|>', '', text)
    text = re.sub(r'<\|stop\|>', '', text)
    
    # "user" 문자열 이전의 모든 내용 제거 (과거 대화 필터링)
    if "user" in text:
        user_parts = text.split("user")
        if len(user_parts) > 1:
            # 맨 마지막 user 다음에 나오는 assistant 찾기
            last_user_part = user_parts[-1]
            if "assistant" in last_user_part:
                asst_parts = last_user_part.split("assistant")
                if len(asst_parts) > 1:
                    # 맨 마지막 assistant 이후 내용만 사용
                    text = asst_parts[-1].strip()
            else:
                # user 부분만 있고 assistant 부분이 없는 경우 (비정상)
                # 전체 텍스트에서 마지막 assistant 이후 내용 찾기
                if "assistant" in text:
                    text = text.split("assistant")[-1].strip()
    
    # 마커 제거 후에도 여전히 "system" 단어로 시작하는 경우 (시스템 프롬프트 누출)
    if text.strip().startswith("system"):
        # "assistant" 단어를 찾아 이후 내용만 사용
        assistant_pos = text.find("assistant")
        if assistant_pos > 0:
            # "assistant" 단어 이후의 첫 번째 개행 이후 내용을 사용
            newline_pos = text.find("\n", assistant_pos)
            if newline_pos > 0:
                text = text[newline_pos+1:]
    
    # "assistant" 문자열로 시작하는 경우 제거
    text = re.sub(r'^assistant\s*[:：]?\s*', '', text.strip())
    
    # 응답에서 시작/끝 공백 제거 및 반환
    return text.strip()

def run_inference(chat_history: list):
    """주어진 대화 기록을 바탕으로 모델 추론을 실행합니다."""
    global model, tokenizer, device

    if model is None or tokenizer is None:
        raise RuntimeError("모델 또는 토크나이저가 로드되지 않았습니다.")

    try:
        inputs = tokenizer.apply_chat_template(
            chat_history,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True, # 모델이 응답을 생성하도록 프롬프트 추가
            return_dict=True
        ).to(device)
    except Exception as e:
        print(f"입력 처리 중 오류: {e}")
        return "입력 처리 중 오류가 발생했습니다.", 0, 0, 0 # 오류 메시지 반환

    input_token_length = inputs["input_ids"].shape[1]

    # 스트리밍 없이 전체 결과 생성
    with torch.no_grad():
        try:
            # max_new_tokens 값을 적절히 조절 (1024는 일반적으로 안전한 값)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024, # 너무 큰 값에서 발생하는 오류 방지
                do_sample=True,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        except Exception as e:
            print(f"추론 중 오류: {e}")
            return "추론 중 오류가 발생했습니다.", input_token_length, 0, input_token_length # 오류 메시지 반환

    total_token_length = output_ids.shape[1]
    output_token_length = total_token_length - input_token_length

    # 생성된 텍스트 디코딩 및 정제
    try:
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        ai_response = clean_response(generated_text)
        return ai_response, input_token_length, output_token_length, total_token_length
    except Exception as e:
        print(f"디코딩 중 오류: {e}")
        return "응답 처리 중 오류가 발생했습니다.", input_token_length, output_token_length, total_token_length 

def format_chat_history(chat_history):
    """채팅 기록의 형식을 검증하고 필요한 경우 수정합니다."""
    try:
        # 입력의 총 토큰 길이 추정 (디버깅용)
        if tokenizer:
            formatted_chat = tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=True
            )
            token_len = len(tokenizer(formatted_chat)["input_ids"])
            print(f"포맷된 채팅 기록의 토큰 길이: {token_len}")
            
            # 너무 긴 경우 경고 출력
            if token_len > 2048:  # 안전한 임계값
                print(f"경고: 입력 토큰 길이가 너무 깁니다 ({token_len} 토큰). 일부 내용이 잘릴 수 있습니다.")
        
        return chat_history
    except Exception as e:
        print(f"채팅 기록 포맷 중 오류: {e}")
        # 오류 발생 시 원본 반환
        return chat_history 

def run_inference_stream(chat_history: list):
    """주어진 대화 기록을 바탕으로 모델 추론을 스트리밍 방식으로 실행하고 토큰을 생성합니다."""
    global model, tokenizer, device

    if model is None or tokenizer is None:
        raise RuntimeError("모델 또는 토크나이저가 로드되지 않았습니다.")

    try:
        inputs = tokenizer.apply_chat_template(
            chat_history,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True
        ).to(device)
    except Exception as e:
        print(f"입력 처리 중 오류: {e}")
        yield f"오류: 입력 처리 중 오류가 발생했습니다. ({e})" # 오류 메시지 yield
        return

    # 스트리밍에 사용할 전체 응답 텍스트 (필터링용)
    full_response = ""
    # 마지막으로 yield한 텍스트 크기 (증분만 제공하기 위해)
    last_length = 0
    
    # skip_prompt=True: 입력 프롬프트를 건너뛰고 생성된 텍스트만 스트리밍
    # skip_special_tokens=True: 특수 토큰을 건너뜀
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # model.generate를 별도 스레드에서 실행
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 스트리머에서 토큰을 yield
    try:
        for new_text in streamer:
            # 모든 토큰을 full_response에 추가
            full_response += new_text

            # 전체 응답 텍스트를 정제 (시스템 프롬프트, 특수 토큰, 과거 대화 제거)
            cleaned_full = clean_response(full_response)
            
            # 이전에 보낸 것보다 긴 경우에만 증분 부분을 yield
            if len(cleaned_full) > last_length:
                incremental_text = cleaned_full[last_length:]
                yield incremental_text
                last_length = len(cleaned_full)
                
    except Exception as e:
        print(f"스트리밍 중 오류: {e}")
        yield f"오류: 스트리밍 중 오류가 발생했습니다. ({e})" # 오류 메시지 yield
    finally:
        # 스레드가 종료될 때까지 대기
        thread.join() 