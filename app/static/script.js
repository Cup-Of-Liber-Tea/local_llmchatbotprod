const chatbox = document.getElementById('chatbox');
const messageForm = document.getElementById('message-form');
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const stopButton = document.getElementById('stop-button');

// 전역 변수로 AbortController 선언
let activeController = null;

// 대화 기록을 저장할 배열 (JavaScript 내에서 관리)
let chatHistory = [];
// 스트리밍 사용 여부 (true: 스트리밍 사용, false: 일반 API 호출)
const useStreaming = true; 

// 메시지를 채팅창에 추가하는 함수
function addMessage(sender, text, type = 'normal') {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');

    const paragraph = document.createElement('p');
    paragraph.textContent = text;

    if (sender === 'user') {
        messageDiv.classList.add('user-message');
    } else {
        if (type === 'error') {
            messageDiv.classList.add('error-message');
        } else if (type === 'loading') {
            messageDiv.classList.add('loading-message');
        } else {
            messageDiv.classList.add('bot-message');
        }
    }

    messageDiv.appendChild(paragraph);
    chatbox.appendChild(messageDiv);
    chatbox.scrollTop = chatbox.scrollHeight;

    // 로딩 메시지인 경우, 해당 요소를 반환하여 나중에 제거할 수 있도록 함
    if (type === 'loading') {
        return messageDiv; 
    }
    return null; // 그 외에는 null 반환
}

// 마크다운을 HTML로 변환하는 간단한 함수
function markdownToHtml(markdown) {
    if (!markdown) return '';
    
    // 1. 코드 블록 변환 (```code```)
    markdown = markdown.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
    
    // 2. 인라인 코드 변환 (`code`)
    markdown = markdown.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // 3. 헤더 변환
    markdown = markdown.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    markdown = markdown.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    markdown = markdown.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // 4. 볼드 변환 (**text**)
    markdown = markdown.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // 5. 이탤릭 변환 (*text*)
    markdown = markdown.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // 6. 리스트 변환
    const lines = markdown.split('\n');
    for (let i = 0; i < lines.length; i++) {
        // 번호 있는 리스트 (1. item)
        if (lines[i].match(/^\d+\. /)) {
            lines[i] = lines[i].replace(/^\d+\. (.*)$/, '<li>$1</li>');
            if (i === 0 || !lines[i-1].includes('<li>')) {
                lines[i] = '<ol>' + lines[i];
            }
            if (i === lines.length - 1 || !lines[i+1].includes('<li>')) {
                lines[i] = lines[i] + '</ol>';
            }
        }
        // 번호 없는 리스트 (- item 또는 * item)
        else if (lines[i].match(/^[\-\*] /)) {
            lines[i] = lines[i].replace(/^[\-\*] (.*)$/, '<li>$1</li>');
            if (i === 0 || !lines[i-1].includes('<li>')) {
                lines[i] = '<ul>' + lines[i];
            }
            if (i === lines.length - 1 || !lines[i+1].includes('<li>')) {
                lines[i] = lines[i] + '</ul>';
            }
        }
    }
    markdown = lines.join('\n');
    
    // 7. 단락 변환 (빈 줄로 구분된 텍스트를 <p> 태그로 감싸기)
    markdown = markdown.replace(/\n\n/g, '</p><p>');
    if (!markdown.startsWith('<h') && !markdown.startsWith('<ul') && !markdown.startsWith('<ol') && !markdown.startsWith('<pre')) {
        markdown = '<p>' + markdown;
    }
    if (!markdown.endsWith('</h1>') && !markdown.endsWith('</h2>') && !markdown.endsWith('</h3>') && !markdown.endsWith('</ul>') && !markdown.endsWith('</ol>') && !markdown.endsWith('</pre>')) {
        markdown = markdown + '</p>';
    }
    
    // 8. 개행 문자를 <br> 태그로 변환 (단락 내에서)
    markdown = markdown.replace(/\n/g, '<br>');
    
    return markdown;
}

// 중지 버튼 이벤트 핸들러 초기 설정
stopButton.onclick = () => {
    if (activeController) {
        console.log("중지 버튼 클릭됨, 스트리밍 중단 시도...");
        activeController.abort(); // 스트리밍 중단
    }
};

// 폼 제출 이벤트 처리
messageForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const userMessage = messageInput.value.trim();
    if (!userMessage) return;

    addMessage('user', userMessage);
    chatHistory.push({ role: 'user', content: userMessage });
    messageInput.value = '';
    
    // 버튼 상태 변경: 전송 비활성화, 중지 활성화
    sendButton.disabled = true;
    stopButton.classList.remove('hidden');
    
    // 로딩 메시지를 추가하고 해당 요소를 변수에 저장
    const loadingMessageElement = addMessage('bot', '메시지 생성 중...', 'loading');
    
    if (useStreaming) {
        // 스트리밍 API 사용 시
        try {
            // 로딩 메시지 제거
            if (loadingMessageElement && chatbox.contains(loadingMessageElement)) {
                chatbox.removeChild(loadingMessageElement);
            }
            
            // 스트리밍 응답을 위한 메시지 요소 생성 (이제 중지 버튼 추가 안 함)
            const streamMessageDiv = document.createElement('div');
            streamMessageDiv.classList.add('message', 'bot-message');
            const paragraph = document.createElement('p');
            streamMessageDiv.appendChild(paragraph);
            
            chatbox.appendChild(streamMessageDiv);
            
            // 새로운 AbortController 생성
            activeController = new AbortController();
            const signal = activeController.signal;
            
            // 서버에 스트리밍 요청
            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_history: chatHistory }),
                signal // AbortController의 signal 추가
            });
            
            if (!response.ok) {
                throw new Error(`HTTP 오류! 상태: ${response.status}`);
            }
            
            // 응답 스트림을 읽기 위한 reader 설정
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            let botResponse = ''; // 전체 응답을 저장할 변수
            
            // 스트림 처리 및 UI 업데이트
            while (true) {
                try {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const text = decoder.decode(value, { stream: true });
                    const cleanedText = cleanBotMessage(text);
                    botResponse += cleanedText;
                    
                    // 마크다운 변환하여 HTML로 출력
                    paragraph.innerHTML = markdownToHtml(botResponse);
                    
                    // 스크롤 맨 아래로 조정
                    chatbox.scrollTop = chatbox.scrollHeight;
                } catch (readError) {
                    if (readError.name === 'AbortError') {
                        console.log('사용자가 생성을 중지했습니다.');
                        botResponse += '\n\n*생성이 중지되었습니다.*';
                        paragraph.innerHTML = markdownToHtml(botResponse);
                        break;
                    } else {
                        throw readError;
                    }
                }
            }
            
            // 대화 기록에 봇 응답 추가
            chatHistory.push({ role: 'assistant', content: botResponse });
            
        } catch (error) {
            if (error.name !== 'AbortError') { // 취소가 아닌 다른 오류인 경우에만 오류 메시지 표시
                console.error('스트리밍 오류 발생:', error);
                addMessage('bot', `오류: ${error.message}`, 'error');
            }
        } finally {
            // 버튼 상태 복원: 전송 활성화, 중지 비활성화
            sendButton.disabled = false;
            stopButton.classList.add('hidden');
            // AbortController 초기화 (중요!)
            activeController = null; 
            messageInput.focus();
        }
    } else {
        // 일반 API 호출 (여기는 중지 버튼 관련 로직 없음)
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ chat_history: chatHistory }),
            });

            // fetch 후 로딩 메시지 제거
            if (loadingMessageElement && chatbox.contains(loadingMessageElement)) {
                chatbox.removeChild(loadingMessageElement);
            }

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP 오류! 상태: ${response.status}`);
            }

            const data = await response.json();
            let botMessage = data.ai_response;

            // 클라이언트 측에서 추가 필터링
            botMessage = cleanBotMessage(botMessage);

            // 봇 메시지 추가 (마크다운 변환하여 HTML로 출력)
            const botMessageElement = document.createElement('div');
            botMessageElement.classList.add('message', 'bot-message');
            const paragraph = document.createElement('p');
            paragraph.innerHTML = markdownToHtml(botMessage);
            botMessageElement.appendChild(paragraph);
            chatbox.appendChild(botMessageElement);
            chatbox.scrollTop = chatbox.scrollHeight;
            
            // 대화 기록에 봇 메시지 추가
            chatHistory.push({ role: 'assistant', content: botMessage });

        } catch (error) {
            console.error('오류 발생:', error);
            addMessage('bot', `오류: ${error.message}`, 'error');
        } finally {
            // 일반 API 호출 후에도 전송 버튼만 활성화
            sendButton.disabled = false;
            stopButton.classList.add('hidden'); // 일반 호출 완료 시 중지 버튼 숨김
            messageInput.focus();
        }
    }
});

// 초기 메시지 추가 (선택 사항, HTML에도 있음)
// addMessage('bot', '안녕하세요! 무엇을 도와드릴까요?'); 

// 봇 메시지 정제 함수 (시스템 프롬프트 및 내부 지시 제거)
function cleanBotMessage(text) {
    if (!text) return "";

    // 시스템 프롬프트 및 관련 내용 제거
    if (text.startsWith('system')) {
        // 'assistant' 다음 내용만 추출
        const assistantParts = text.split('assistant');
        if (assistantParts.length > 1) {
            text = assistantParts[assistantParts.length - 1].trim();
            // 콜론(:) 이나 개행 문자로 시작하면 제거
            text = text.replace(/^[\s\:\n]+/, '');
        }
    }

    // [지식 상태 확인], [문제 분해] 등 구조화된 프레임워크 마커 제거
    text = text.replace(/\*\*\[\s*[^]]*\]\*\*\s*\([^)]*\)/g, '');

    // 자기점검 마커 제거
    text = text.replace(/\([^)]*자기점검[^)]*\)/g, '');

    return text;
} 