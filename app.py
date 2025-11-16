import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import re
from threading import Thread
import os

# 페이지 설정
st.set_page_config(
    page_title="HyperCLOVA X SEED 챗봇",
    page_icon="💬",
    layout="centered"
)

# 응답 정제 함수 정의 (스트리밍 후 전체 텍스트에 적용)
def clean_response(text):
    # 특수 토큰 제거 (streamer의 skip_special_tokens=True 로 대체될 수 있음)
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

# 모델 로드 함수 (캐싱 사용)
@st.cache_resource
def load_model(model_size="0.5B"):
    # 토큰 설정 (환경 변수 우선 사용)
    token = os.getenv("HF_TOKEN") # 환경 변수 우선 사용
    
    if model_size == "0.5B":
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    elif model_size == "1.5B":
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    elif model_size == "3B": # 3B 모델 옵션 추가
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-3B"
    else:
        # 기본값으로 0.5B 사용 또는 에러 처리
        print(f"Warning: Unsupported model size '{model_size}'. Defaulting to 0.5B.")
        model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
    
    with st.spinner(f"{model_size} 모델 로딩 중... 잠시만 기다려주세요"):
        # 토크나이저와 모델 로드
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
        st.session_state.device = device # 디바이스 정보 세션 상태에 저장
        
    return tokenizer, model

def generate_response(chat_history, tokenizer, model, device):
    # 전체 채팅 히스토리 딕셔너리 변환 (최근 5개 메시지만 유지)
    history_dicts = [msg for msg in chat_history if msg["role"] != "system"] # 시스템 메시지 제외하고 시작
    recent_history_dicts = history_dicts[-10:] # 최근 5턴 (사용자+AI = 10개 메시지)
    
    # 시스템 메시지 추가 (항상 맨 앞에)
    system_message = next((msg for msg in chat_history if msg["role"] == "system"), None)
    if system_message:
        recent_history_dicts.insert(0, system_message)
    else: # 혹시 시스템 메시지 빠졌으면 기본값 추가
        recent_history_dicts.insert(0, {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."})

    # 채팅 템플릿 적용
    try:
        inputs = tokenizer.apply_chat_template(
            recent_history_dicts,
            return_tensors="pt", 
            tokenize=True, 
            add_generation_prompt=True,
            return_dict=True
        ).to(device) # 바로 디바이스로 이동
    except Exception as e:
        st.error(f"채팅 템플릿 적용 중 오류 발생: {e}")
        return None # 오류 발생 시 streamer 대신 None 반환
    
    # 스트리머 설정
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 생성 파라미터 설정
    generation_kwargs = dict(
        inputs=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"), # attention_mask 추가
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 별도 스레드에서 모델 생성 실행
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 스트리머 객체 반환
    return streamer

def create_chat_ui():
    st.title("HyperCLOVA X SEED 챗봇")
    
    # 사이드바 - 모델 선택
    st.sidebar.title("설정")
    model_size = st.sidebar.radio(
        "모델 크기 선택",
        ["0.5B", "1.5B", "3B"], # 3B 옵션 추가
        index=0, # 기본 0.5B
        key="model_size_selection", # 상태 유지를 위한 키
        help="0.5B, 1.5B, 3B 모델 중 선택 (3B는 고사양 필요)"
    )
    
    # 모델 로드 상태 관리
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_model_size" not in st.session_state:
        st.session_state.current_model_size = None

    # 모델 로드 버튼 또는 모델 크기 변경 시 모델 로드
    if st.sidebar.button("모델 로드/변경") or st.session_state.current_model_size != model_size:
        try:
            st.session_state.tokenizer, st.session_state.model = load_model(model_size)
            st.session_state.model_loaded = True
            st.session_state.current_model_size = model_size
            # 모델 변경 시 채팅 히스토리 초기화
            st.session_state.chat_history = [{"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."}]
            st.sidebar.success(f"{model_size} 모델 로딩 완료!")
        except Exception as e:
            st.sidebar.error(f"모델 로딩 실패: {e}")
            st.session_state.model_loaded = False
            st.session_state.current_model_size = None
        st.rerun() # UI 갱신
        
    # CUDA 정보 표시
    st.sidebar.subheader("CUDA 정보")
    cuda_available = torch.cuda.is_available()
    st.sidebar.write(f"CUDA 사용 가능: {'✅' if cuda_available else '❌'}")
    if cuda_available and "device" in st.session_state:
        try:
            st.sidebar.write(f"CUDA 디바이스: {torch.cuda.get_device_name(st.session_state.device)}")
        except Exception:
            st.sidebar.write("CUDA 디바이스 정보 로드 실패")
    
    # 대화 초기화 버튼
    if st.sidebar.button("대화 초기화"):
        if "chat_history" in st.session_state:
            # 시스템 메시지만 남기고 초기화
            st.session_state.chat_history = [msg for msg in st.session_state.chat_history if msg["role"] == "system"]
        st.rerun() # UI 갱신
    
    # 세션 상태에 채팅 히스토리 초기화 (앱 시작 시 한 번)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."}]
        
    # 채팅 내역 표시 (시스템 메시지 제외)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
    
    # 사용자 입력 처리
    if st.session_state.model_loaded:
        user_input = st.chat_input("메시지를 입력하세요...")
        if user_input:
            # 사용자 메시지를 히스토리에 추가하고 UI에 표시
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="🧑"):
                st.write(user_input)
            
            # 응답 생성 및 스트리밍 표시
            with st.chat_message("assistant", avatar="🤖"):
                streamer = generate_response(
                    st.session_state.chat_history,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    st.session_state.device
                )
                if streamer: # streamer가 정상적으로 반환되었을 때만 실행
                    with st.spinner("생각 중..."): # 스피너를 스트리밍 영역과 함께 배치
                        # 스트리밍 출력을 표시하고 완료 후 전체 텍스트 받기
                        full_response = st.write_stream(streamer)
                    
                    # 스트리밍 완료 후, 전체 응답을 히스토리에 추가
                    cleaned_response = clean_response(full_response) # 간단한 정제 적용
                    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
                    # 스트리밍 완료 후에는 rerun이 필요 없음 (write_stream이 UI 업데이트)
                else:
                    st.error("응답 생성 중 오류가 발생했습니다.") 
            
    else:
        st.info("먼저 사이드바에서 모델을 로드/변경 버튼을 클릭해주세요.")

if __name__ == "__main__":
    # 환경 변수 HF_TOKEN 설정 확인 (선택 사항)
    if not os.getenv("HF_TOKEN"):
        print("Warning: Hugging Face token (HF_TOKEN) environment variable not set. Using default token.")
    create_chat_ui() 