#!/usr/bin/env python
"""
웹 검색 기능이 통합된 교정형 RAG 시스템 사용 예제
"""

import os
import sys
from improved_web_search import setup_and_run

def main():
    """Example usage of the corrective RAG system with web search"""
    
    # Check API keys
    if os.environ.get("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY" or not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Set it in your environment variables.")
    
    if os.environ.get("TAVILY_API_KEY") == "YOUR_TAVILY_API_KEY" or not os.environ.get("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not set. Set it in your environment variables.")

    # Ask for a question if none provided
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("\n=== 웹 검색 기능이 통합된 교정형 RAG 시스템 데모 ===")
        print("질문을 입력하세요 (예: '태종실록에 대해 알려줘' 또는 'What is agent memory in LLMs?')")
        print("질문: ", end="")
        question = input().strip()
    
    if not question:
        print("질문을 입력해주세요.")
        return
    
    print(f"\n질문: {question}")
    print("\n처리 중...\n")
    
    # 샘플 URL 목록 (기본값 대신 다른 URL 사용 가능)
    custom_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://python.langchain.com/docs/expression_language/cookbook/retrieval",
        "https://python.langchain.com/docs/integrations/retrievers/",
    ]
    
    # RAG 시스템 실행
    answer = setup_and_run(question, custom_urls)
    
    print("\n=== 답변 ===")
    print(answer)
    
    print("\n=== 참고 사항 ===")
    print("이 예제는 다음을 보여줍니다:")
    print("1. 로컬 지식에서 관련 정보 검색")
    print("2. 관련성이 낮은 경우 자동 웹 검색 수행")
    print("3. 검색된 정보를 바탕으로 LLM 답변 생성")
    print("4. 웹 검색 결과 사용 시 출처 표시")

if __name__ == "__main__":
    main() 