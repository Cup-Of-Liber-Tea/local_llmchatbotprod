# 필요한 라이브러리들을 가져옵니다.
# transformers: 허깅페이스(Hugging Face) 라이브러리로, 사전 훈련된 모델과 토크나이저를 쉽게 사용할 수 있게 해줍니다.
# AutoTokenizer: 모델에 맞는 토크나이저를 자동으로 로드하는 클래스입니다. 토크나이저는 텍스트를 모델이 이해할 수 있는 숫자(토큰 ID)로 변환합니다.
# AutoModelForCausalLM: 인과 관계 언어 모델(Causal LM)을 자동으로 로드하는 클래스입니다. Causal LM은 이전 단어들을 바탕으로 다음 단어를 예측하는 방식으로 텍스트를 생성합니다.
# TextStreamer: 모델이 생성하는 텍스트를 실시간으로(토큰 단위로) 출력해주는 유틸리티입니다.
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# torch: 파이토치(PyTorch) 라이브러리로, 딥러닝 모델 구축 및 연산을 위한 핵심 라이브러리입니다.
import torch
import re
import os
# accelerate 라이브러리는 device_map="auto" 사용 시 내부적으로 활용될 수 있습니다.
from accelerate import Accelerator # 명시적 임포트 (필수는 아닐 수 있음)

# --- 모델 및 토크나이저 설정 ---
# 사용할 LLM의 이름을 지정합니다. 허깅페이스 모델 허브에 등록된 이름을 사용합니다.
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

# 허깅페이스 API 토큰을 환경 변수 'HF_TOKEN'에서 읽어옵니다.
# 비공개 모델이나 특정 모델 접근 시 필요할 수 있습니다. 
# 환경 변수 설정이 권장되지만, 직접 코드에 입력할 수도 있습니다(보안상 비권장).
token = os.getenv("HF_TOKEN") # 환경 변수 사용

# 만약 환경 변수에 토큰이 없다면 사용자에게 경고 메시지를 출력합니다.
if not token:
    print("경고: HF_TOKEN 환경 변수가 설정되지 않았습니다. 필요시 로그인해야 할 수 있습니다.")
    # token = "hf_..." # 여기에 직접 토큰을 입력할 수도 있습니다.

# --- CUDA(GPU) 확인 및 연산 장치 설정 ---
# torch.cuda.is_available(): 현재 시스템에서 NVIDIA GPU를 파이토치가 사용할 수 있는지 확인합니다.
use_cuda = torch.cuda.is_available()
# torch.device(): 모델 연산을 수행할 장치(device)를 설정합니다.
# GPU 사용이 가능하면 'cuda'(GPU)를, 불가능하면 'cpu'(CPU)를 사용하도록 설정합니다.
device = torch.device("cuda" if use_cuda else "cpu")
# 설정된 장치 정보를 출력합니다.
print(f"CUDA 사용 가능: {use_cuda}, 사용 디바이스: {device}")
# 만약 GPU를 사용한다면, 첫 번째 GPU의 이름을 출력합니다 (참고용).
if use_cuda:
    print(f"CUDA 디바이스 이름: {torch.cuda.get_device_name(0)}")

# --- 모델 및 토크나이저 로드 (8비트 양자화 적용) ---
print(f"{model_name} 모델 로딩 중 (8-bit 양자화 적용)...")
# GPU 메모리 절약을 위해 8비트 양자화 및 자동 디바이스 매핑 사용
# torch_dtype = torch.float16 if use_cuda else torch.float32 # 양자화 시 주석 처리 또는 제거
try:
    # AutoTokenizer.from_pretrained(): 지정된 모델 이름에 맞는 토크나이저를 허깅페이스 허브에서 다운로드하고 로드합니다.
    # token: 필요한 경우 허깅페이스 API 토큰을 전달합니다.
    # trust_remote_code=True: 일부 모델은 허깅페이스 라이브러리에 없는 사용자 정의 코드를 포함하는데, 이 코드를 신뢰하고 실행할지 여부를 결정합니다. (보안에 유의해야 함)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    # AutoModelForCausalLM.from_pretrained(): 지정된 이름의 Causal LM 모델을 허깅페이스 허브에서 다운로드하고 로드합니다.
    # token, trust_remote_code: 토크나이저와 동일한 이유로 사용됩니다.
    # torch_dtype: 위에서 설정한 데이터 타입(float16 또는 float32)으로 모델 가중치를 로드하도록 지정합니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        device_map="auto",  # 모델 레이어를 GPU/CPU에 자동으로 분산 배치
        load_in_8bit=True,  # 모델 가중치를 8비트로 양자화하여 로드 (bitsandbytes 필요)
        # torch_dtype=torch_dtype # 8비트 로딩 시 주석 처리 또는 제거
    )
# 만약 모델 로딩 중 오류가 발생하면,
except Exception as e:
    # 오류 메시지를 출력하고,
    print(f"모델 로딩 중 오류 발생: {e}")
    # 사용자에게 가능한 원인(토큰 문제, 로그인 필요)을 안내한 후,
    print("Hugging Face 토큰이 유효한지, 또는 로그인이 필요한지 확인하세요.")
    # 프로그램을 종료합니다.
    exit()

# --- 모델 설정 및 이동 ---
# model.to(device): 로드된 모델의 모든 파라미터(가중치)를 위에서 설정한 연산 장치(GPU 또는 CPU)로 이동시킵니다.
# 이렇게 해야 해당 장치에서 모델 연산이 가능해집니다.
# model.to(device)
# model.eval(): 모델을 평가(evaluation) 모드로 설정합니다.
# 이 모드에서는 학습 시 사용되는 드롭아웃(Dropout)이나 배치 정규화(Batch Normalization) 등이 비활성화되어, 일관된 추론 결과를 얻을 수 있습니다.
model.eval()
# 모델이 로드된 주 장치를 확인하여 출력 (참고용, 실제로는 여러 장치에 분산될 수 있음)
print(f"모델 로딩 완료. (메인 디바이스 추정: {model.device})")

# --- 간단한 응답 정제 함수 정의 ---
# 모델이 생성한 텍스트에는 종종 특별한 의미를 가진 토큰(예: 문장 시작/끝, 역할 구분)이 포함될 수 있습니다.
# 이 함수는 사용자에게 보여주기 전에 이러한 불필요한 특수 토큰들을 제거하는 역할을 합니다.
def clean_response(text):
    # re.sub(pattern, replacement, string): 문자열(string)에서 정규식 패턴(pattern)에 맞는 부분을 찾아 다른 문자열(replacement)로 바꿉니다.
    # 여기서는 <|im_start|>assistant, <|im_end|> 등의 특수 토큰들을 빈 문자열('')로 바꾸어 제거합니다.
    text = re.sub(r'<\|im_start\|>assistant', '', text) # 어시스턴트 역할 시작 태그 제거
    text = re.sub(r'<\|im_end\|>', '', text) # 메시지 끝 태그 제거
    text = re.sub(r'<\|endofturn\|>', '', text) # 대화 턴 끝 태그 제거
    text = re.sub(r'<\|stop\|>', '', text) # 생성 중지 토큰 제거
    # text.strip(): 문자열 앞뒤의 공백(띄어쓰기, 줄바꿈 등)을 제거합니다.
    text = text.strip()
    # 정제된 텍스트를 반환합니다.
    return text

# --- 대화 기록 초기화 (시스템 프롬프트 포함) ---
# chat_history: 사용자와 AI 간의 대화 내용을 저장하는 리스트입니다.
# 각 대화 턴은 딕셔너리 형태로 저장되며, 'role'(역할: system, user, assistant)과 'content'(내용) 키를 가집니다.
chat_history = [
    # 시스템 메시지 (System Prompt): 대화 시작 전에 AI에게 역할을 부여하거나, 따라야 할 지침을 제공하는 메시지입니다.
    # 이 프롬프트는 AI의 응답 스타일과 내용에 큰 영향을 줍니다.
    # 현재 프롬프트는 AI에게 4단계 사고 과정을 설명하도록 지시합니다.
    {"role": "system", "content": """
당신은 신뢰성 높은 추론과 자기검증 능력을 갖추고 질문에 답변할 때 다음 구조화된 프레임워크를 사용하여 환각을 최소화하고 논리적 일관성을 유지하는 AI 어시스턴트입니다.

응답 구조 (반드시 이 순서와 형식을 따르세요):

1. **[지식 상태 확인]**
   - 질문에 대한 자신의 지식 수준을 평가합니다(높음/중간/낮음).
   - 불확실한 부분이 있다면 명확히 표시합니다.
   - (자기점검: "이 주제에 대한 나의 지식 한계를 정직하게 인정했는가?")

2. **[문제 분해]**
   - 질문을 더 작은 하위 질문으로 분해합니다.
   - 핵심 개념과 용어를 명확히 정의합니다.
   - (자기점검: "질문의 모든 중요 측면을 포함했는가?")

3. **[관련 지식 활성화]**
   - 질문 해결에 필요한 관련 개념, 원리, 사실을 간략히 나열합니다.
   - 각 정보의 확실성 수준을 표시합니다.
   - (자기점검: "제시한 지식이 정확하고 관련성이 있는가?")

4. **[단계적 추론]**
   - 명확한 논리적 단계를 통해 답변을 도출합니다.
   - 각 단계마다 이유와 근거를 제시합니다.
   - (자기점검: "각 추론 단계가 논리적으로 연결되는가?")

5. **[대안 가설 검토]**
   - 최소 한 가지 대안적 관점이나 접근법을 고려합니다.
   - 왜 이 대안이 주 추론보다 약한지 설명합니다.
   - (자기점검: "편향 없이 다른 가능성을 고려했는가?")

6. **[논리적 오류 검증]**
   - 자신의 추론에서 가능한 오류, 비약, 환각을 검토합니다.
   - 확인되지 않은 가정을 명시적으로 표시합니다.
   - (자기점검: "내 추론에 논리적 오류나 근거 없는 주장이 있는가?")

7. **[결론 도출]**
   - 앞선 단계를 종합하여 균형 잡힌 결론을 제시합니다.
   - 남아있는 불확실성이나 제한사항을 명시합니다.
   - (자기점검: "결론이 분석에서 논리적으로 도출되었는가?")

8. **[최종 답변]**
   - 이전 7단계의 추론 과정을 종합하여 다음 요소를 체계적으로 포함한 포괄적 답변을 제공합니다:
     
     a) **핵심 결론 요약** (3-5개 핵심 포인트)
        • 각 결론에 가장 강력한 근거 1-2개 첨부
        • 결론의 확실성 수준 표시 (확실함/가능성 높음/추정됨)
     
     b) **실용적 적용 방안** (2-4개)
        • 구체적이고 실행 가능한 조치나 권장사항
        • 가능한 경우 단계별 접근법 또는 우선순위 제시
     
     c) **잠재적 한계 및 고려사항** (1-3개)
        • 결론 적용 시 주의해야 할 제약사항
        • 추가 정보가 필요한 영역이나 불확실성
     
     d) **상황별 조정 가이드** (선택적)
        • 다양한 상황에 따른 접근법 조정 방안
        • "만약 ~라면, ~하는 것이 좋습니다" 형식의 조건부 조언
   
   - 모든 내용은 일반인도 이해할 수 있는 명확하고 구체적인 언어로 작성하되, 필요한 전문 용어는 간결히 설명합니다.
   - 전체 최종 답변은 약 15-20문장 정도로, 중요도에 따라 비중을 조절하여 작성합니다.


답변 작성 시 지침:
- [최종 답변]을 제외한 각 단계는 간결하게 유지하며, 쉬운 언어와 명확한 예시를 사용하세요.
- 알지 못하는 내용은 솔직히 인정하고, 추측과 사실을 명확히 구분하세요.
- 특정 사실에 대한 불확실성이 있을 경우 "~일 수 있습니다", "~로 추정됩니다"와 같은 표현을 사용하세요.
- 추론 과정에서 자기일관성(self-consistency)을 유지하고, 모순되는 진술을 하지 마세요.

항상 이 구조와 자기점검을 포함하여, 명확하고 친절한 한국어로 답변하세요.

    """}
] 

# --- 스트리머 설정 ---
# TextStreamer 객체를 생성합니다.
# tokenizer: 모델이 생성한 토큰 ID를 다시 텍스트로 변환(디코딩)하기 위해 필요합니다.
# skip_prompt=True: 모델 입력으로 들어간 프롬프트 부분은 출력하지 않도록 합니다.
# skip_special_tokens=True: 디코딩 시 특수 토큰(<|im_end|> 등)을 자동으로 제거하고 출력합니다.
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 사용자에게 추론 시작을 알리는 메시지를 출력합니다.
print("\n--- 추론 시작 ('종료' 입력 시 종료) ---")

# --- 메인 대화 루프 ---
# 사용자가 '종료'를 입력할 때까지 무한히 반복합니다.
while True:
    # 사용자로부터 입력을 받습니다. "나: " 라는 프롬프트를 보여줍니다.
    user_input = input("나: ")
    # 사용자가 입력한 내용을 소문자로 변환하여 '종료'인지 확인합니다.
    if user_input.lower() == '종료':
        # '종료'가 맞으면 종료 메시지를 출력하고,
        print("프로그램을 종료합니다.")
        # break 키워드로 while 루프를 빠져나갑니다.
        break

    # 사용자의 입력을 딕셔너리 형태로 만들어 대화 기록(chat_history) 리스트에 추가합니다.
    chat_history.append({"role": "user", "content": user_input})

    # --- 모델 입력 준비 ---
    # try-except 블록: 입력 처리 중 발생할 수 있는 오류를 처리합니다.
    try:
        # tokenizer.apply_chat_template(): 대화 기록(chat_history) 리스트를 모델이 이해할 수 있는 형식으로 변환합니다.
        #   chat_history: 시스템 프롬프트와 전체 대화 내용이 담긴 리스트입니다.
        #   return_tensors="pt": 결과를 파이토치 텐서(PyTorch Tensor) 형태로 반환하도록 지정합니다.
        #   tokenize=True: 텍스트를 토큰 ID로 변환합니다.
        #   add_generation_prompt=True: 모델에게 이제 응답을 생성할 차례임을 알리는 프롬프트를 추가합니다.
        #   return_dict=True: 결과를 딕셔너리 형태로 반환합니다 (예: {'input_ids': ..., 'attention_mask': ...}).
        # .to(device): 변환된 입력 텐서들을 모델이 있는 연산 장치(GPU 또는 CPU)로 이동시킵니다.
        inputs = tokenizer.apply_chat_template(
            chat_history,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True
        ).to(model.device) # 입력도 모델의 주 디바이스로 이동
    # 입력 처리 중 오류 발생 시,
    except Exception as e:
        # 오류 메시지를 출력하고,
        print(f"입력 처리 중 오류 발생: {e}")
        # 방금 추가했던 사용자 입력을 대화 기록에서 제거합니다 (오류 유발 입력을 남기지 않기 위함).
        chat_history.pop()
        # continue 키워드로 현재 루프 반복을 중단하고 다음 반복으로 넘어갑니다 (다음 사용자 입력을 기다림).
        continue

    # 모델 생성(generate) 함수에 입력된 토큰의 길이를 저장합니다.
    # 나중에 모델이 생성한 새로운 토큰들만 추출하기 위해 사용됩니다.
    input_token_length = inputs["input_ids"].shape[1]

    # --- 모델 추론 실행 (텍스트 생성) ---
    # AI의 응답이 시작됨을 알리기 위해 "AI: "를 먼저 출력합니다. end=""는 줄바꿈 없이 출력하라는 의미입니다.
    print("AI: ", end="")
    # torch.no_grad(): 이 컨텍스트 블록 안에서는 파이토치가 기울기(gradient)를 계산하지 않습니다.
    # 추론(텍스트 생성) 시에는 모델 학습이 아니므로 기울기 계산이 필요 없고, 이를 비활성화하면 메모리 사용량을 줄이고 계산 속도를 약간 높일 수 있습니다.
    with torch.no_grad():
        # try-except 블록: 모델 추론 중 발생할 수 있는 오류(예: CUDA 메모리 부족)를 처리합니다.
        try:
            # model.generate(): 모델에게 텍스트 생성을 지시하는 핵심 함수입니다.
            #   **inputs: 준비된 입력 딕셔너리(inputs)의 내용('input_ids', 'attention_mask')을 함수의 인자로 풀어 전달합니다.
            #   streamer=streamer: 위에서 설정한 TextStreamer 객체를 전달하여, 생성되는 토큰을 실시간으로 출력하도록 합니다.
            #   max_new_tokens=512: 모델이 새로 생성할 최대 토큰 수를 제한합니다. 응답 길이를 조절합니다.
            #   do_sample=True: 다음 토큰을 선택할 때 확률적 샘플링을 사용합니다. False면 가장 확률 높은 토큰만 선택하여 덜 창의적인 결과가 나옵니다.
            #   temperature=0.7: 샘플링 시 확률 분포를 조절하는 값입니다. 낮을수록 확률 높은 단어 위주로, 높을수록 다양한 단어가 선택될 확률이 높아집니다. (보통 0.7~1.0 사용)
            #   eos_token_id=tokenizer.eos_token_id: 문장 끝(End Of Sentence)을 나타내는 토큰의 ID를 지정합니다. 모델이 이 토큰을 생성하면 생성을 멈춥니다.
            #   pad_token_id=tokenizer.pad_token_id: 입력 길이를 맞추기 위해 사용되는 패딩 토큰의 ID를 지정합니다. generate 함수 내부에서 필요할 수 있습니다.
            output_ids = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        # 추론 중 오류 발생 시,
        except Exception as e:
            # 오류 메시지 앞에 줄바꿈을 추가하여 출력합니다 (스트리밍 중일 수 있으므로).
            print(f"\n추론 중 오류 발생: {e}")
            # 오류 유발 입력을 대화 기록에서 제거합니다.
            chat_history.pop()
            # 다음 사용자 입력을 기다립니다.
            continue
            
    # 스트리밍 출력이 끝나면 줄바꿈을 해줍니다.
    print()

    # --- 생성된 응답 처리 (대화 기록 저장용) ---
    # 모델이 생성한 전체 output_ids에는 입력 부분도 포함되어 있습니다.
    # 따라서 입력 길이를 기준으로 그 이후의 ID들만 추출하여 모델이 '새로 생성한' 토큰들만 얻습니다.
    new_tokens = output_ids[0, input_token_length:]
    # tokenizer.decode(): 토큰 ID 시퀀스(new_tokens)를 사람이 읽을 수 있는 텍스트 문자열로 변환(디코딩)합니다.
    # skip_special_tokens=True: 디코딩 과정에서 특수 토큰을 제거합니다. (streamer가 이미 처리했겠지만, 기록용으로도 제거)
    assistant_response_raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # clean_response() 함수를 호출하여 디코딩된 텍스트에서 혹시 남아있을 수 있는 특수 토큰이나 불필요한 공백을 추가로 정제합니다.
    assistant_response = clean_response(assistant_response_raw)

    # 최종 응답 출력은 streamer가 이미 했으므로 주석 처리합니다.
    # print(f"AI: {assistant_response}")

    # --- 대화 기록 업데이트 ---
    # 정제된 AI의 최종 응답 내용이 비어있지 않은 경우에만,
    if assistant_response:
        # AI의 응답을 딕셔너리 형태로 만들어 대화 기록(chat_history) 리스트에 추가합니다.
        # 이렇게 해야 다음 사용자 입력 시 AI가 이전 자신의 답변도 참고할 수 있습니다.
        chat_history.append({"role": "assistant", "content": assistant_response})

    # --- (선택 사항) 메모리 관리용 대화 기록 제한 ---
    # 대화가 너무 길어지면 입력 토큰 수가 증가하여 메모리 부족이나 속도 저하가 발생할 수 있습니다.
    # 주석 처리된 이 부분은 대화 기록의 길이를 제한하는 예시입니다.
    # if len(chat_history) > 10: # 예: 최대 10개 턴(시스템 메시지 포함) 유지
    #     # 시스템 메시지(chat_history[0])는 항상 유지하고, 가장 최근의 9개 턴만 남깁니다.
    #     chat_history = [chat_history[0]] + chat_history[-9:] 