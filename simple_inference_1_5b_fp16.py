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
# 사용할 LLM의 이름을 지정합니다. 여기서는 1.5B 텍스트 모델을 사용합니다.
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

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

# --- 모델 및 토크나이저 로드 (양자화 없음) ---
print(f"{model_name} 모델 로딩 중...") # 로딩 메시지 수정
# 모델을 로드할 때 사용할 데이터 타입(dtype)을 설정합니다.
# GPU를 사용할 때는 torch.float16 (반정밀도 부동소수점)을 사용하여 메모리 사용량을 줄이고 속도를 높입니다.
# CPU를 사용할 때는 torch.float32 (단정밀도 부동소수점)를 사용하여 호환성을 확보합니다.
torch_dtype = torch.float16 if use_cuda else torch.float32 # dtype 설정 복원
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    # AutoModelForCausalLM.from_pretrained(): 양자화 옵션 없이 로드합니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        torch_dtype=torch_dtype # dtype 설정 사용
    )
except Exception as e:
    # 오류 메시지를 출력하고,
    print(f"모델 로딩 중 오류 발생: {e}")
    # 사용자에게 가능한 원인(토큰 문제, 로그인 필요)을 안내한 후,
    print("Hugging Face 토큰이 유효한지, 또는 로그인이 필요한지 확인하세요.")
    # 프로그램을 종료합니다.
    exit()

# --- 모델 설정 및 이동 ---
# model.to(device): 로드된 모델을 설정된 연산 장치(GPU 또는 CPU)로 이동시킵니다.
model.to(device) # 복원
# model.eval(): 모델을 평가(evaluation) 모드로 설정합니다.
model.eval()
print(f"모델 로딩 완료. 디바이스: {device}") # 로딩 완료 메시지 수정

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
<?xml version="1.0" encoding="UTF-8"?>
<system_prompt>
  <introduction>
    신뢰성 높은 추론과 자기검증 능력을 갖추고, 구조화된 프레임워크로 환각을 최소화하는 AI 어시스턴트입니다.
  </introduction>
  <instructions>
    1-7단계는 간결하게, 8단계는 상세한 보고서로 작성하세요. 전체의 70% 이상은 최종 답변(8단계)에 할애하세요.
  </instructions>
  <response_structure>
    <step id="1" name="지식_상태_확인" max_words="100">
      <description>지식 수준 평가 및 한계 명시</description>
      <self_check>정직한 한계 인지</self_check>
    </step>
    <step id="2" name="문제_분해" max_words="150">
      <description>하위 질문 분해 및 핵심 용어 정의</description>
      <self_check>중요 측면 모두 포함</self_check>
    </step>
    <step id="3" name="관련_지식_활성화" max_words="200">
      <description>관련 개념 및 확실성 표시</description>
      <self_check>정확성 및 관련성 점검</self_check>
    </step>
    <step id="4" name="단계적_추론" max_words="200">
      <description>논리적 단계별 근거</description>
      <self_check>논리적 연결성 확인</self_check>
    </step>
    <step id="5" name="대안_가설_검토" max_words="150">
      <description>대안적 접근 및 한계 설명</description>
      <self_check>편향 없이 고려</self_check>
    </step>
    <step id="6" name="논리적_오류_검증" max_words="150">
      <description>오류 및 환각 검토</description>
      <self_check>근거 없는 주장 명시</self_check>
    </step>
    <step id="7" name="결론_도출" max_words="200">
      <description>종합 결론 및 제한사항</description>
      <self_check>논리적 도출 여부 확인</self_check>
    </step>
    <step id="8" name="최종_답변" min_words="1500">
    <description>이전 7단계의 추론 과정을 종합하여 다음 요소를 포함한 포괄적인 보고서를 제공한다:</description>
    <importance>이 단계는 전체 응답의 70% 이상을 차지해야 하며, 반드시 1500자 이상이어야 합니다. 이 길이 요구사항은 필수적입니다.</importance>
    <elements>
        <element>주제의 역사적 배경 및 발전 과정 (최소 200자)</element>
        <element>핵심 개념 및 원리에 대한 상세한 설명 (최소 300자)</element>
        <element>다양한 관점과 이론적 접근 (최소 200자)</element>
        <element>현재 동향 및 미래 전망 (최소 200자)</element>
        <element>실제 사례 및 응용 (최소 200자)</element>
        <element>관련 통계 데이터 및 연구 결과 (가능한 경우)</element>
        <element>분야별 세부 내용 및 상호 관계 (최소 200자)</element>
    </elements>
    <formatting_guidelines>
        <guideline>정보는 논리적으로 구조화하고, 적절한 소제목, 번호 매기기, 표, 글머리 기호 등을 사용하여 가독성을 높인다.</guideline>
        <guideline>모든 내용은 일반인도 이해할 수 있는 명확하고 구체적인 언어로 작성하되, 필요한 전문 용어는 간결히 설명한다.</guideline>
        <guideline>글자 수가 1500자 미만일 경우 더 자세히 설명하고 예시를 추가하여 최소 글자 수를 충족해야 한다.</guideline>
    </formatting_guidelines>
    </step>

    <additional_guidelines>
    <guideline>각 추론 단계(1-7)는 간결하게 유지하며, 단계별로 단어 제한을 준수한다.</guideline>
    <guideline>최종 답변(8단계)은 매우 상세하고 포괄적이어야 하며, 반드시 1500자 이상이어야 한다. 각 섹션별 최소 글자 수를 반드시 지킬 것.</guideline>
    <guideline>알지 못하는 내용은 솔직히 인정하고, 추측과 사실을 명확히 구분한다.</guideline>
    <guideline>추론 과정에서 자기일관성(self-consistency)을 유지하고, 모순되는 진술을 하지 않는다.</guideline>
    </additional_guidelines>

  <language_instruction>
    항상 이 구조와 자기점검을 포함하여, 명확하고 친절한 한국어로 답변
  </language_instruction>
</system_prompt>



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
        # .to(device)를 사용하여 설정된 장치로 이동합니다.
        inputs = tokenizer.apply_chat_template(
            chat_history,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True
        ).to(device) # .to(device) 사용
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
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
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

    # --- 결과 지표 계산 및 출력 (추가 부분) ---
    # 생성된 총 토큰 수 (입력 + 출력)
    total_token_length = output_ids.shape[1]
    # 새로 생성된 출력 토큰 수 계산
    output_token_length = total_token_length - input_token_length

    # 결과 지표 출력
    print(f"--- 결과 지표 ---")
    print(f"입력 토큰 수: {input_token_length}")
    print(f"출력 토큰 수: {output_token_length}")
    print(f"총 토큰 수: {total_token_length}")
    print(f"-----------------\n")

    # --- 대화 기록에 AI 응답 추가 ---
    # 모델이 생성한 전체 텍스트를 디코딩합니다.
    # output_ids[0]는 배치 크기가 1이라고 가정하고 첫 번째(그리고 유일한) 시퀀스를 선택합니다.
    # skip_special_tokens=False로 설정하여 EOS 같은 특수 토큰 포함 여부를 확인하고,
    # 후처리에서 제거하는 것이 더 안전할 수 있습니다. (여기서는 True 유지)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 입력 프롬프트를 제외한 실제 생성된 응답 부분만 추출합니다.
    # generated_text에는 입력 프롬프트가 포함되어 있을 수 있으므로,
    # 입력 길이를 기준으로 잘라내거나, 더 정확하게는 특정 시작 토큰 이후를 추출해야 합니다.
    # 여기서는 clean_response 함수가 역할을 하므로 전체 텍스트를 넘깁니다.
    ai_response = clean_response(generated_text)

    # 정제된 AI 응답을 대화 기록에 추가합니다.
    chat_history.append({"role": "assistant", "content": ai_response})

    # --- (선택 사항) 메모리 관리용 대화 기록 제한 ---
    # 대화가 너무 길어지면 입력 토큰 수가 증가하여 메모리 부족이나 속도 저하가 발생할 수 있습니다.
    # 주석 처리된 이 부분은 대화 기록의 길이를 제한하는 예시입니다.
    # if len(chat_history) > 10: # 예: 최대 10개 턴(시스템 메시지 포함) 유지
    #     # 시스템 메시지(chat_history[0])는 항상 유지하고, 가장 최근의 9개 턴만 남깁니다.
    #     chat_history = [chat_history[0]] + chat_history[-9:] 