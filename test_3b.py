from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoProcessor
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
model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B"

# 모델 로드
print(f"모델 로딩 중 ({model_name})... 잠시만 기다려주세요")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # bfloat16 사용 (float16 대신)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)
    preprocessor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"모델이 {device}에 로드되었습니다.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    exit(1)

# 응답 정제 함수
def clean_response(text):
    # 특수 토큰 제거
    special_tokens = ['<|im_start|>', '<|im_end|>', '<|endofturn|>', 
                     '<|stop|>', '<|pad|>', '<|unk|>', '<|mask|>', 
                     '<|sep|>', '<|cls|>']
    
    for token in special_tokens:
        text = text.replace(token, '')
    
    # 빈 줄 제거
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 앞뒤 공백 제거
    return text.strip()

# 이미지 처리 및 응답 생성 함수
def process_image_and_generate(image_path, prompt="이 이미지에 대해 설명해주세요"):
    try:
        # 이미지 입력 준비
        vlm_chat = [
            {"role": "system", "content": {"type": "text", "text": "당신은 도움이 되는 AI 어시스턴트입니다. 한국어로 친절하게 답변해주세요."}},
            {"role": "user", "content": {"type": "text", "text": prompt}},
            {
                "role": "user",
                "content": {
                    "type": "image",
                    "filename": os.path.basename(image_path),
                    "image": image_path
                }
            }
        ]
        
        new_vlm_chat, all_images, is_video_list = preprocessor.load_images_videos(vlm_chat)
        preprocessed = preprocessor(all_images, is_video_list=is_video_list)
        
        # bfloat16 타입으로 변환 (preprocessed 내 텐서 처리)
        for key in preprocessed:
            if isinstance(preprocessed[key], torch.Tensor):
                preprocessed[key] = preprocessed[key].to(device=device, dtype=torch.bfloat16)
        
        input_ids = tokenizer.apply_chat_template(
            new_vlm_chat, return_tensors="pt", tokenize=True, add_generation_prompt=True
        )
        
        # 생성 파라미터
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "top_p": 0.6,
            "temperature": 0.5,
            "repetition_penalty": 1.0,
        }
        
        # 입력을 모델과 같은 디바이스로 이동 (bfloat16 타입으로)
        input_ids = input_ids.to(device=device, dtype=torch.bfloat16)
        
        # 응답 생성
        print("\nAI 응답 생성 중...")
        output_ids = model.generate(
            input_ids=input_ids,
            **generation_config,
            **preprocessed
        )
        
        # 응답 디코딩
        response = tokenizer.batch_decode(output_ids)[0]
        return clean_response(response)
        
    except Exception as e:
        return f"이미지 처리 중 오류 발생: {e}"

# 메인 대화 루프
print("\n모델 로딩 완료! 대화를 시작합니다.")
print("이미지 경로를 입력하거나, '종료'를 입력하여 프로그램을 종료할 수 있습니다.")

while True:
    try:
        user_input = input("\n이미지 경로 또는 명령을 입력하세요: ")
        
        if user_input.lower() == '종료':
            print("프로그램을 종료합니다.")
            break
        
        # 기본 이미지 프롬프트
        img_prompt = "이 이미지에 대해 설명해주세요"
        
        # 이미지 경로 검사
        if os.path.exists(user_input):
            # 추가 프롬프트 입력 받기
            prompt_input = input("이미지에 대한 질문을 입력하세요 (기본: 이미지 설명): ")
            if prompt_input.strip():
                img_prompt = prompt_input
            
            # 이미지 처리 및 응답 생성
            response = process_image_and_generate(user_input, img_prompt)
            print(f"\nAI: {response}")
        else:
            print(f"오류: 파일을 찾을 수 없습니다 - {user_input}")
            print("올바른 이미지 경로를 입력하세요.")
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
        break
    except Exception as e:
        print(f"오류 발생: {e}")
        continue 