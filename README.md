# Corrective RAG with Web Search

이 프로젝트는 LangGraph를 사용하여 웹 검색 기능이 통합된 RAG(Retrieval-Augmented Generation) 시스템을 구현한 애플리케이션입니다.

## 주요 기능

- 로컬 벡터 DB에서 정보 검색
- 문서 관련성 평가
- 관련성이 낮을 때 자동 웹 검색
- 질문 최적화 재작성
- 소스 인용이 포함된 답변 생성
- 한국어 및 영어 지원

## 설치 방법

1. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

2. API 키 설정:

환경 변수로 설정:

```bash
# Windows
set GOOGLE_API_KEY=your_google_api_key
set TAVILY_API_KEY=your_tavily_api_key

# Linux/Mac
export GOOGLE_API_KEY=your_google_api_key
export TAVILY_API_KEY=your_tavily_api_key
```

또는 코드에서 직접 설정 (권장하지 않음):
```python
import os
os.environ['GOOGLE_API_KEY'] = "your_google_api_key"
os.environ['TAVILY_API_KEY'] = "your_tavily_api_key"
```

## 사용 방법

### 웹 인터페이스 (Streamlit)

웹 인터페이스를 실행하려면:

```bash
python main.py --web
```

또는 간단히:

```bash
python main.py
```

Streamlit 앱이 시작되면:
1. 질문을 입력하세요
2. 사이드바에서 사용자 지정 지식 베이스 URL을 추가할 수 있습니다
3. 디버그 정보 표시 옵션을 활성화하여 시스템 동작을 볼 수 있습니다

### 명령줄 인터페이스

명령줄에서 사용하려면:

```bash
python main.py --cli "질문을 여기에 입력하세요"
```

또는 대화형 모드:

```bash
python main.py --cli
```

그러면 질문을 입력하라는 메시지가 표시됩니다.

## 시스템 구성

애플리케이션은 다음과 같은 주요 컴포넌트로 구성됩니다:

1. **벡터 데이터베이스**: 문서를 저장하고 벡터 검색을 수행
2. **질문 재작성**: 웹 검색에 최적화된 질문 생성
3. **문서 관련성 평가**: 검색된 문서의 관련성 판단
4. **웹 검색**: Tavily를 사용한 실시간 웹 검색
5. **답변 생성**: 관련 정보를 사용하여 정보가 풍부한 답변 생성

## 워크플로우

1. 사용자가 질문을 입력
2. 로컬 벡터 DB에서 관련 문서 검색
3. 문서 관련성 평가
4. 관련성이 낮으면 질문 재작성 및 웹 검색 수행
5. 모든 관련 정보를 활용하여 답변 생성
6. 웹 검색 결과가 사용된 경우 출처 인용 추가

## 코드 구조

- `improved_web_search.py`: 주요 RAG 및 웹 검색 기능 구현
- `streamlit_app.py`: Streamlit 웹 인터페이스
- `main.py`: 명령줄 및 웹 인터페이스 실행 옵션
