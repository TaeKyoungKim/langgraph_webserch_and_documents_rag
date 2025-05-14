# 웹 검색 기능이 통합된 교정형 RAG 시스템 상세 설명

## 프로젝트 개요

이 프로젝트는 LangGraph를 활용하여 자동 웹 검색 기능을 갖춘 교정형 RAG(Retrieval-Augmented Generation) 시스템을 구현합니다. 기존 RAG 시스템의 한계점인 로컬 지식 기반만으로는 답변할 수 없는 질문에 대응하기 위해, 문서 관련성 평가 및 웹 검색을 통한 확장된 지식 검색 기능을 구현하였습니다.

## 주요 기능 및 장점

1. **하이브리드 정보 검색**: 로컬 벡터 데이터베이스 검색과 웹 검색을 결합하여 보다 포괄적인 정보 제공
2. **문서 관련성 평가**: LLM 기반의 검색 결과 관련성 평가로 저품질 정보 필터링
3. **질문 최적화**: 웹 검색을 위한 질문 재작성으로 검색 효율성 향상
4. **Gemini 및 Groq LLM 지원**: 다양한 LLM 백엔드 활용 가능
5. **한국어/영어 지원**: 다국어 질의응답 지원
6. **웹 및 CLI 인터페이스**: 다양한 사용 환경 지원
7. **출처 인용**: 웹 검색 결과 활용 시 출처 정보 제공
8. **디버그 모드**: 시스템 작동 과정 시각화를 통한 투명성 제공

## 기술 스택

- **[LangGraph](https://python.langchain.com/docs/langgraph)**: 복잡한 상태 머신 및 조건부 흐름 관리
- **[LangChain](https://python.langchain.com/)**: 기본적인 RAG 파이프라인 구현
- **[Gemini API](https://ai.google.dev/gemini-api)**: 기본 LLM 및 임베딩 모델
- **[Groq API](https://console.groq.com/docs/quickstart)**: 보조 LLM(Llama 3.1 8B)
- **[Chroma DB](https://www.trychroma.com/)**: 벡터 데이터베이스 저장 및 검색
- **[Tavily](https://tavily.com/)**: 웹 검색 API
- **[Streamlit](https://streamlit.io/)**: 웹 인터페이스 구현

## 시스템 아키텍처

교정형 RAG 시스템은 다음과 같은 주요 컴포넌트로 구성됩니다:

### 1. 상태 관리 (State Management)

LangGraph의 `StateGraph` 및 `TypedDict`를 사용하여 다음 상태 정보를 관리합니다:

```python
class State(TypedDict):
    question: str               # 현재 질문
    original_question: str      # 원래 질문(재작성 전)
    documents: List[Document]   # 검색된 문서 목록
    web_search: str             # 웹 검색 필요 여부 플래그
    generation: str             # 생성된 답변
    web_results: List[Dict]     # 웹 검색 결과
    relevance_score: str        # 문서 관련성 점수
```

### 2. 노드 및 엣지 구성

LangGraph 워크플로우는 다음 노드들로 구성됩니다:

- **retrieve**: 로컬 벡터 DB에서 문서 검색
- **grade_documents**: 검색된 문서의 관련성 평가
- **transform_query**: 웹 검색을 위한 질문 최적화
- **web_search_node**: Tavily API를 통한 웹 검색 수행
- **generate**: 최종 답변 생성

이들 노드는 다음과 같은 실행 흐름으로 연결됩니다:

```
START → retrieve → grade_documents → [조건부 분기]
    ├── 관련성 높음 → generate → END
    └── 관련성 낮음 → transform_query → web_search_node → generate → END
```

### 3. 문서 관련성 평가

문서 관련성 평가는 다음과 같은 로직으로 수행됩니다:

1. LLM 기반 평가기를 구성하여 각 문서의 관련성을 이진 점수('yes'/'no')로 평가
2. 최소 2개 이상의 관련 문서가 필요하며, 그렇지 않을 경우 웹 검색 트리거

```python
# 관련 문서가 너무 적으면 웹 검색 필요
if relevant_count < 2:  # 최소 2개 이상의 관련 문서가 있어야 함
    web_search = "Yes"
    print(f"---ONLY {relevant_count} RELEVANT DOCUMENTS, WEB SEARCH NEEDED---")
```

### 4. 웹 검색 및 통합

웹 검색 프로세스는 다음과 같은 단계로 수행됩니다:

1. 질문 재작성으로 웹 검색에 최적화된 쿼리 생성
2. Tavily API를 통한 웹 검색 수행
3. 검색 결과를 Document 객체로 변환하여 기존 문서와 통합
4. 출처 정보 보존하여 최종 답변에 인용 포함

```python
# 웹 검색 결과를 문서 형태로 변환
web_docs = []
for result in search_results:
    content = result.get("content", "")
    title = result.get("title", "")
    if content:
        metadata = {"source": result.get("url", ""), "title": title}
        doc = Document(page_content=content, metadata=metadata)
        web_docs.append(doc)
```

## 구현 세부 사항

### 벡터 데이터베이스 설정

웹 페이지에서 문서를 로드하여 청크로 분할하고 임베딩하여 Chroma 벡터 데이터베이스에 저장합니다:

```python
def setup_vectordb(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    return vectorstore.as_retriever()
```

### 질문 재작성 최적화

웹 검색에 최적화된 질문을 생성하기 위한 프롬프트 엔지니어링:

```python
system = """You are a question re-writer that converts an input question to a better version that is optimized 
for web search. Look at the input and try to reason about the underlying semantic intent / meaning.

Your task is to improve the question to maximize the chance of finding relevant information. Focus on:
1. Clarifying ambiguous terms
2. Adding specific keywords related to the topic
3. Phrasing it in a way that would match informational content

Keep the question concise and focused on the main topic.
"""
```

### 다중 LLM 지원

Gemini 및 Groq LLM을 모두 지원하여 다양한 모델을 활용할 수 있도록 구현:

```python
# Gemini 모델
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Groq 모델 (Llama 3.1)
llm_grop = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
```

### 출처 인용 구현

웹 검색 결과가 사용된 경우 답변 하단에 출처 정보를 포함하여 신뢰성을 향상:

```python
# 웹 검색 결과 사용 여부 표시
if state.get("web_search") == "Completed" and state.get("web_results"):
    sources = []
    for idx, result in enumerate(state.get("web_results", [])[:3], 1):
        if "url" in result:
            sources.append(f"{idx}. {result.get('title', 'Source ' + str(idx))}: {result['url']}")
    
    if sources:
        generation += "\n\nSources:\n" + "\n".join(sources)
```

## 사용자 인터페이스

### 웹 인터페이스 (Streamlit)

Streamlit을 활용한 웹 인터페이스는 다음과 같은 기능을 제공합니다:

1. **질문 입력**: 사용자가 자연어로 질문 입력
2. **커스텀 URL 추가**: 사이드바에서 사용자 지정 지식 베이스 URL 추가 가능
3. **디버그 정보**: 시스템 작동 과정 실시간 표시 옵션
4. **API 키 관리**: 환경 변수 또는 Streamlit 시크릿을 통한 안전한 API 키 관리

```python
# URL inputs
custom_urls = []
use_default = st.sidebar.checkbox("Use default knowledge base", value=True)

if not use_default:
    st.sidebar.markdown("Enter URLs to use as knowledge base:")
    for i in range(3):  # Allow adding up to 3 custom URLs
        url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
        if url:
            custom_urls.append(url)
```

### 명령줄 인터페이스 (CLI)

명령줄에서 간편하게 사용할 수 있는 인터페이스도 제공:

```python
def run_cli():
    """Run the command-line interface version"""
    from improved_web_search import setup_and_run
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Enter your question: ", end="")
        question = input()
    
    answer = setup_and_run(question)
    print("\n=== FINAL ANSWER ===")
    print(answer)
```

## 확장성 및 추가 기능

이 프로젝트는 다음과 같은 방향으로 확장 가능합니다:

1. **다중 검색 엔진 통합**: Google, Bing 등 다양한 검색 엔진 API 통합
2. **구조화된 데이터 추출**: 웹 검색 결과에서 테이블, 목록 등 구조화된 데이터 추출 기능
3. **멀티모달 지원**: 이미지 검색 및 분석 기능 추가
4. **추론 체인 시각화**: 복잡한 추론 과정을 시각적으로 표현하는 기능
5. **다중 언어 모델 앙상블**: 여러 LLM을 조합하여 최적의 답변 생성
6. **사용자 피드백 루프**: 사용자 피드백을 통한 시스템 자가 개선 메커니즘

## 성능 및 한계점

### 성능 지표

- **검색 정확도**: 관련 문서 검색 및 평가 정확도
- **답변 품질**: 생성된 답변의 정확성, 관련성, 완전성
- **실행 시간**: 전체 파이프라인 실행 시간 및 각 단계별 지연 시간

### 현재 한계점

1. **API 의존성**: Gemini, Groq, Tavily API 의존성으로 인한 비용 및 가용성 문제
2. **문서 평가 복잡성**: 복잡한 질문의 경우 문서 관련성 평가가 어려울 수 있음
3. **검색 결과 품질**: 웹 검색 API의 품질에 의존적인 결과
4. **모델 할루시네이션**: LLM의 특성상 존재하지 않는 정보를 생성할 가능성
5. **컨텍스트 창 제한**: 대량의 검색 결과를 처리할 때 LLM의 컨텍스트 창 제한에 영향 받음

## 코드 구조 및 패턴

### 주요 모듈 및 파일

- `improved_web_search.py`: 핵심 RAG 및 LangGraph 로직 구현
- `streamlit_app.py`: Streamlit 웹 인터페이스
- `main.py`: CLI 및 웹 인터페이스 진입점
- `requirements.txt`: 의존성 관리

### 설계 패턴 및 원칙

1. **상태 기반 워크플로우**: LangGraph를 통한 상태 관리 및 조건부 실행 흐름
2. **모듈성**: 각 기능을 모듈화하여 유지보수성 향상
3. **타입 힌팅**: TypedDict 및 타입 힌팅을 통한 코드 안정성 강화
4. **지연 초기화**: 필요할 때만 컴포넌트 초기화하여 리소스 효율성 향상
5. **에러 처리**: 단계별 예외 처리로 견고성 제공

```python
# 지연 초기화 패턴 예시
def initialize_components():
    """Initialize global components if they haven't been initialized yet"""
    global retriever, retrieval_grader, question_rewriter, web_search_tool, llm, embeddings
    
    if llm is not None:
        return  # Already initialized
    
    # 컴포넌트 초기화...
```

## 설치 및 실행 가이드

### 환경 설정

1. **패키지 설치**:
```bash
pip install -r requirements.txt
```

2. **API 키 설정**:
```bash
# Windows
set GOOGLE_API_KEY=your_google_api_key
set TAVILY_API_KEY=your_tavily_api_key
set GROQ_API_KEY=your_groq_api_key

# Linux/Mac
export GOOGLE_API_KEY=your_google_api_key
export TAVILY_API_KEY=your_tavily_api_key
export GROQ_API_KEY=your_groq_api_key
```

### 실행 방법

1. **웹 인터페이스**:
```bash
python main.py
# 또는
python main.py --web
```

2. **명령줄 인터페이스**:
```bash
python main.py --cli "질문을 여기에 입력하세요"
# 또는 대화형 모드
python main.py --cli
```

## 라이센스 및 참고자료

- 이 프로젝트는 MIT 라이센스를 따릅니다.
- LangGraph 및 LangChain 공식 문서를 참고하여 개발되었습니다.
- Lilian Weng의 에이전트 및 프롬프트 엔지니어링에 관한 블로그 포스트를 지식 베이스로 활용하였습니다.

## 결론

이 교정형 RAG 시스템은 로컬 지식 한계를 웹 검색으로 확장하여 보다 포괄적인 질의응답 기능을 제공합니다. LangGraph를 활용한 상태 관리 및 조건부 워크플로우는 복잡한 RAG 파이프라인을 체계적으로 구현할 수 있게 합니다. 또한 다양한 LLM 지원 및 문서 관련성 평가를 통해 정보의 품질을 유지하면서 웹 검색과의 통합을 실현하였습니다. 향후 다중 검색 엔진 통합, 멀티모달 지원 등으로 기능을 확장할 수 있는 기반을 마련하였습니다. 