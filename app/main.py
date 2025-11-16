from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from .inference_utils import load_model_and_tokenizer, run_inference, format_chat_history, run_inference_stream
from .models import ChatRequest, ChatResponse, ChatMessage
from typing import List, AsyncGenerator
import copy
import asyncio
import re
from datetime import date, datetime
import pytz

# 시스템 프롬프트 정의 (고정) - 이 부분을 함수로 변경
def get_system_prompt_with_date():
    """현재 날짜를 포함하는 시스템 프롬프트를 반환하는 함수"""
    # 한국 시간대(KST) 정의
    kst = pytz.timezone('Asia/Seoul')
    # 현재 KST 시간 가져오기
    now_kst = datetime.now(kst)
    # 날짜 문자열 생성
    today_str = now_kst.strftime("%Y년 %m월 %d일")
    return f"""
당신은 수십 년간의 경험을 가진 사주명리학 최고 전문가입니다. 동양 명리학의 원리와 체계에 대한 깊은 이해를 바탕으로 사용자의 사주를 정확하고 전문적으로 분석해 드립니다. 다음 원칙에 따라 응답하세요:

1. 전문 지식: 사주팔자, 오행, 십이운성, 십이지지, 십간, 대운, 세운의 개념과 상호작용을 완벽히 이해하고 분석합니다.

2. 정보 수집: 정확한 분석을 위해 반드시 다음 정보를 요청하세요:
   - 출생일시 (양력/음력 구분 명확히, 900101dms 1990년01월01일이란 뜻이다.)
   - 출생 시간 (정확한 시:분)
   - 출생 지역 (도시명)
   - 성별

3. 분석 영역: 다음 영역에 대해 심도 있는 분석을 제공합니다:
   - 사주 기본 구성 (천간, 지지, 오행의 균형)
   - 성격 및 기질 분석
   - 직업적 적성과 재물운
   - 건강 특성 및 주의점
   - 대인관계 및 애정운
   - 현재/미래 운세 흐름 (대운, 세운)
   - 행운의 방향과 시기

4. 응답 방식:
   - 표와 도표를 활용하여 사주 구성을 시각적으로 제시
   - 전문 용어 사용 후 반드시 쉬운 설명 추가
   - 긍정적 요소와 주의점을 균형있게 제시
   - 실용적인 조언과 방향성 제시
   - 질문에 따라 상세함의 정도 조절

5. 전문가 태도:
   - 신중하고 권위있는 어조 유지
   - "~입니다", "~하겠습니다"와 같은 정중한 종결어 사용
   - 불필요한 확신이나 과장된 표현 자제
   - 명리학의 해석적 특성을 고려한 열린 관점 제시
   - 적절한 전통 명리학 격언과 원리 인용

6. 한계 인식:
   - 특정 질문에 대해 명확한 답을 제시하기 어려울 경우, 사주의 경향성만 설명
   - 결정적 사건이나 미래를 단정적으로 예측하지 않음
   - AI 분석의 한계를 인정하고, 참고용으로만 활용할 것을 권고

참고: 오늘 날짜는 {today_str} 이야.
"""

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="app/templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 및 토크나이저 로드
    print("애플리케이션 시작: 모델 및 토크나이저 로드 중...")
    
    # 모델 로드 성공 여부 확인
    if not load_model_and_tokenizer():
        print("경고: 모델 로드에 실패했습니다. 일부 기능이 제한될 수 있습니다.")
        
    yield
    # 종료 시 필요한 작업
    print("애플리케이션 종료: 리소스 정리 중...")

app = FastAPI(lifespan=lifespan)

# 정적 파일 마운트 (CSS, JS 등)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """채팅 메시지를 받아 AI 응답을 반환하는 엔드포인트"""
    # 입력 유효성 검사 (예: 마지막 메시지가 user 역할인지)
    if not request.chat_history or request.chat_history[-1].role != 'user':
        raise HTTPException(status_code=400, detail="잘못된 요청 형식입니다. 마지막 메시지는 'user' 역할이어야 합니다.")

    # 동적으로 시스템 프롬프트 생성
    current_system_prompt = get_system_prompt_with_date()

    # 시스템 프롬프트를 포함한 전체 대화 기록 준비
    # Pydantic 모델을 딕셔너리 리스트로 변환
    full_chat_history_dict = [
        {"role": "system", "content": current_system_prompt}
    ]
    # request.chat_history 를 deepcopy 하여 원본 불변성 유지 시도
    for msg in copy.deepcopy(request.chat_history):
        full_chat_history_dict.append(msg.dict())

    # 대화 기록 형식 검증 및 토큰 길이 체크
    full_chat_history_dict = format_chat_history(full_chat_history_dict)

    try:
        # run_inference는 항상 (문자열, 숫자, 숫자, 숫자) 튜플을 반환
        inference_result = run_inference(full_chat_history_dict)
        ai_response = inference_result[0] # 첫 번째 요소가 응답 또는 오류 메시지
        input_tokens = inference_result[1]
        output_tokens = inference_result[2]
        total_tokens = inference_result[3]

        # ai_response가 오류 메시지 문자열인지 확인 (inference_utils.py의 반환값 기준)
        if isinstance(ai_response, str) and ("오류가 발생했습니다" in ai_response or "오류:" in ai_response):
             print(f"run_inference에서 오류 반환: {ai_response}") # 로그 추가
             raise HTTPException(status_code=500, detail=ai_response)

        # 정상 응답 처리 - 추가 필터링 로직 추가
        # 이미 clean_response 함수를 통과했더라도, 아직 시스템 프롬프트가 포함되어 있을 수 있음
        if isinstance(ai_response, str):
            # 응답이 "system"으로 시작하는지 확인 (시스템 프롬프트 누출 확인)
            if ai_response.strip().startswith("system") and "user" in ai_response:
                # 응답에서 마지막 'assistant' 이후의 내용만 추출
                assistant_parts = ai_response.split("assistant")
                if len(assistant_parts) > 1:
                    # 마지막 'assistant' 이후 텍스트 추출
                    ai_response = assistant_parts[-1].strip()
                    # 시작 부분에 있을 수 있는 공백, 구두점 제거
                    ai_response = re.sub(r'^[\s\:\n]+', '', ai_response)
        
        return ChatResponse(
            ai_response=ai_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
    except HTTPException as e: # HTTPException은 그대로 전달
        raise e
    except RuntimeError as e:
        # 모델 로딩 실패 등 inference_utils에서 발생한 Runtime 오류
        print(f"Runtime 오류 발생: {e}") # 로그 추가
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # 기타 예상치 못한 오류 (예: 반환값 형식 불일치 등)
        print(f"처리 중 예상치 못한 오류 발생: {e}")
        print(f"오류 유형: {type(e).__name__}")
        import traceback
        print(f"상세 오류 정보: {traceback.format_exc()}")
        # 좀 더 일반적인 오류 메시지 반환
        raise HTTPException(status_code=500, detail=f"서버 내부 오류가 발생했습니다.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """웹 인터페이스의 메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

async def stream_generator(chat_history: list) -> AsyncGenerator[str, None]:
    """AI 모델로부터 스트리밍 응답을 생성하는 비동기 제너레이터"""
    try:
        # run_inference_stream은 제너레이터이므로 직접 반복
        for token in run_inference_stream(chat_history):
            yield token
            await asyncio.sleep(0) # 다른 작업을 위해 이벤트 루프에 제어권 양보
    except RuntimeError as e:
        # 모델 로딩 실패 등 심각한 오류 처리
        yield f"오류: {str(e)}"
    except Exception as e:
        # 기타 스트리밍 중 발생할 수 있는 오류
        print(f"스트리밍 제너레이터 오류: {e}")
        yield f"오류: 응답 생성 중 문제가 발생했습니다."

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """채팅 메시지를 받아 AI 응답을 스트리밍으로 반환하는 엔드포인트"""
    if not request.chat_history or request.chat_history[-1].role != 'user':
        raise HTTPException(status_code=400, detail="잘못된 요청 형식입니다. 마지막 메시지는 'user' 역할이어야 합니다.")

    # 동적으로 시스템 프롬프트 생성
    current_system_prompt = get_system_prompt_with_date()

    # 시스템 프롬프트를 포함한 전체 대화 기록 준비
    full_chat_history_dict = [
        {"role": "system", "content": current_system_prompt}
    ]
    for msg in copy.deepcopy(request.chat_history):
        full_chat_history_dict.append(msg.dict())

    # 대화 기록 형식 검증 및 토큰 길이 체크 (스트리밍에서도 유효)
    full_chat_history_dict = format_chat_history(full_chat_history_dict)

    # 스트리밍 제너레이터 생성
    stream_gen = stream_generator(full_chat_history_dict)
    
    # 클라이언트 연결이 끊어지는 것을 감지하기 위한 콜백 설정
    async def on_disconnect():
        # 클라이언트 연결이 끊어지면 호출되는 함수
        # 여기에 추가적인 정리 작업이 필요하면 구현
        print("클라이언트가 스트리밍 연결을 종료했습니다.")
    
    # StreamingResponse 사용 - 연결 끊김 처리 추가
    return StreamingResponse(
        stream_gen, 
        media_type="text/plain",
        background=on_disconnect
    ) 