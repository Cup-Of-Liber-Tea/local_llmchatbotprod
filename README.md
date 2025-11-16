# HyperCLOVA X SEED 챗봇

네이버 HyperCLOVA X SEED 모델을 사용한 챗봇 애플리케이션입니다. 웹 UI(Streamlit) 또는 API 서버(FastAPI) 형태로 실행할 수 있으며, 0.5B, 1.5B, 3B 모델 크기를 지원합니다.

## 설치 방법

1.  **가상 환경 생성 (권장):**
    ```bash
    python -m venv .env
    source .env/bin/activate  # Linux/macOS
    # .env\Scripts\activate  # Windows
    ```

2.  **필요한 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **참고:** 서버 환경의 CUDA 버전에 따라 PyTorch 설치 명령이 다를 수 있습니다. `requirements.txt`의 `torch` 라인을 주석 처리하고 [PyTorch 공식 웹사이트](https://pytorch.org/)에서 환경에 맞는 설치 명령을 확인하여 직접 설치하는 것을 권장합니다.

3.  **Hugging Face 토큰:** 코드 내에 기본 토큰이 포함되어 있으나, 환경 변수 `HF_TOKEN`을 설정하여 사용하는 것이 더 안전하고 권장됩니다.

## 실행 방법

### API 서버 (FastAPI)

API 서버는 `/chat` 엔드포인트를 통해 챗봇 기능을 제공합니다. 다른 웹 애플리케이션에서 이 API를 호출하여 사용할 수 있습니다.

```bash
# 0.5B 모델 로드 (기본값)
uvicorn api:app --host 0.0.0.0 --port 8000

# 1.5B 모델 로드 (환경 변수 사용)
API_MODEL_SIZE=1.5B uvicorn api:app --host 0.0.0.0 --port 8000

# 3B 모델 로드 (환경 변수 사용)
API_MODEL_SIZE=3B uvicorn api:app --host 0.0.0.0 --port 8000

# 개발 환경 (코드 변경 시 자동 재시작, 0.5B 로드)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

*   서버가 시작되면 `http://<서버 IP>:8000/docs` 에서 API 문서를 확인할 수 있습니다.
*   API 서버 시작 시 환경 변수 `API_MODEL_SIZE`를 설정하여 로드할 모델 크기 (0.5B, 1.5B, 3B)를 지정할 수 있습니다. 설정하지 않으면 기본값(0.5B)이 사용됩니다.

**API 요청 예시 (Python `requests` 사용):**

```python
import requests

api_url = "http://localhost:8000/chat"

user_query = "오늘 날씨 어때?"
chat_history = [
    # {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."}, # 시스템 메시지는 API 서버가 관리
    {"role": "user", "content": "안녕"},
    {"role": "assistant", "content": "안녕하세요!"}
]

data = {
    "user_input": user_query,
    "history": chat_history, # 클라이언트는 시스템 메시지 제외하고 전달
    "max_history": 5 # 최근 5턴의 대화만 고려
}

try:
    response = requests.post(api_url, json=data)
    response.raise_for_status() # 오류 발생 시 예외 발생
    result = response.json()
    print("AI 응답:", result['response'])

    # 다음 요청을 위해 현재 질문과 응답을 history에 추가
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": result['response']})

except requests.exceptions.RequestException as e:
    print(f"API 요청 오류: {e}")
except Exception as e:
    print(f"오류 발생: {e}")
```

### 웹 UI 버전 (Streamlit)

```bash
streamlit run app.py
```

*   웹 브라우저가 자동으로 열리고 챗봇 UI가 표시됩니다.
*   사이드바에서 원하는 모델 크기(0.5B, 1.5B, 3B)를 선택하고 "모델 로드/변경" 버튼을 클릭합니다.
*   채팅 입력 필드에 메시지를 입력하고 Enter 키를 누릅니다.

### 콘솔 버전

```bash
python test.py      # 0.5B 모델
python test_1_5b.py # 1.5B 모델
python test_3b.py   # 3B 모델
```

## 주요 기능

*   0.5B, 1.5B, 3B 모델 선택 가능 (API는 환경 변수로 설정)
*   채팅 히스토리 지원 (API는 클라이언트 측 관리)
*   응답 정제 기능
*   실시간 응답 스트리밍 (콘솔 및 웹 UI)
*   API 엔드포인트 제공 (FastAPI)
*   웹 기반 인터페이스 (Streamlit)

## 참고 사항

*   GPU 메모리가 충분해야 모델이 원활하게 실행됩니다.
    *   0.5B: 4GB+ VRAM 권장
    *   1.5B: 8GB+ VRAM 권장
    *   **3B: 10GB+ VRAM 권장 (매우 고사양 필요)**
*   시스템 RAM도 충분해야 합니다 (모델 크기에 따라 16GB ~ 32GB+ 권장).
*   모델 로딩에 시간이 걸릴 수 있습니다. API 서버는 시작 시 지정된 모델을 미리 로드합니다.
*   API 서버는 상태 없이(stateless) 작동하므로, 대화 기록 관리는 API를 호출하는 클라이언트 측에서 담당해야 합니다. 