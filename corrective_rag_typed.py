from typing import TypedDict, Annotated, List, Optional, Union, Dict, Any
from langchain_core.documents import Document
import operator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the GraphState based on the user query
class GraphState(TypedDict):
    """Type hints for the graph state."""
    context: Annotated[List[Document], operator.add]
    answer: Annotated[List[Document], operator.add]
    question: Annotated[str, "user question"]
    sql_query: Annotated[str, "sql query"]
    binary_score: Annotated[str, "binary score yes or no"]
    rewritten_question: Annotated[str, "optimized question for search"]
    web_search: Annotated[str, "whether web search is needed"]
    language: Annotated[str, "language of the question"]
    error: Annotated[Optional[str], "error message if any"]

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    
    Args:
        text (str): Input text to detect language
        
    Returns:
        str: Detected language code ('ko' for Korean, 'en' for English, etc.)
    """
    # Simple language detection based on character codes
    # This is a basic implementation - in production, use a proper language detection library
    korean_chars = 0
    for char in text:
        if '\uAC00' <= char <= '\uD7A3':  # Korean Unicode range
            korean_chars += 1
    
    # If more than 20% of characters are Korean, assume Korean
    if korean_chars > len(text) * 0.2:
        return "ko"
    return "en"

def rewrite_question(state: GraphState) -> GraphState:
    """
    Rewrites the input question to a better version optimized for web search.
    Supports both English and Korean language questions.

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with rewritten question
    """
    print("---REWRITING QUESTION---")
    
    question = state["question"]
    
    # Detect language
    language = state.get("language", detect_language(question))
    
    try:
        # Create the question rewriter components with language-specific prompts
        if language == "ko":
            system = """당신은 입력 질문을 웹 검색에 최적화된 더 나은 버전으로 변환하는 질문 재작성자입니다.
                입력을 살펴보고 기본 의미론적 의도/의미를 추론해 보세요. 한국어로 응답해 주세요."""
        else:
            system = """You are a question re-writer that converts an input question to a better version that is optimized
                for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
            
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        
        # Assuming llm is defined elsewhere and imported
        question_rewriter = re_write_prompt | llm | StrOutputParser()
        
        # Get the rewritten question
        rewritten_question = question_rewriter.invoke({"question": question})
        
        error = None
    except Exception as e:
        # Handle any errors that occur during rewriting
        print(f"Error rewriting question: {str(e)}")
        rewritten_question = question  # Fall back to original question
        error = f"Error in question rewriting: {str(e)}"
    
    # Return with the GraphState constructor
    return GraphState(
        context=state.get("context", []),  # Preserve context
        answer=state.get("answer", []),    # Preserve answer
        question=question,                 # Keep original question
        rewritten_question=rewritten_question,  # Add the rewritten question
        sql_query=state.get("sql_query", ""),   # Preserve SQL query
        binary_score=state.get("binary_score", ""),  # Preserve binary score
        web_search=state.get("web_search", "No"),  # Default to No for web search
        language=language,  # Add detected language
        error=error  # Add error if any
    )

def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve documents with type annotations

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with retrieved documents
    """
    print("---RETRIEVE---")
    
    question = state["question"]
    error = None
    
    try:
        # Use rewritten question if available, otherwise use original question
        search_question = state.get("rewritten_question", question) if not state.get("error") else question
        
        # Assuming retriever is defined elsewhere and imported
        documents = retriever.get_relevant_documents(search_question)
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        documents = []
        error = f"Error in document retrieval: {str(e)}"
    
    # Return with the GraphState constructor
    return GraphState(
        context=documents,  # Store retrieved documents in context
        answer=[],  # Empty answer initially
        question=question,  # Keep original question
        rewritten_question=state.get("rewritten_question", ""),  # Preserve rewritten question
        sql_query=state.get("sql_query", ""),  # Preserve existing SQL query
        binary_score="",  # Empty binary score initially
        web_search=state.get("web_search", "No"),  # Preserve web search flag
        language=state.get("language", detect_language(question)),  # Preserve or detect language
        error=error  # Add error if any
    )

def grade_documents(state: GraphState) -> GraphState:
    """
    Determines whether the retrieved documents are relevant to the question.
    Uses type annotations with GraphState.

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with graded documents
    """
    
    print("---CHECKING DOCUMENT RELEVANT IS TO QUESTION OR NOT---")
    
    question = state["question"]
    documents = state["context"]  # Get documents from context
    error = None

    # Score each doc
    filtered_docs = []
    binary_score = "no"  # Default to no
    web_search = "No"  # Default to No
    
    if not documents:
        print("No documents to grade, may need web search")
        web_search = "Yes"
        error = state.get("error", "No documents found to grade")
    else:
        try:
            for d in documents:
                # Assuming retrieval_grader is defined elsewhere and imported
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                    binary_score = "yes"  # At least one document is relevant
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    web_search = "Yes"  # Need web search if doc not relevant
            
            # If no documents were found relevant, we need web search
            if not filtered_docs:
                web_search = "Yes"
        except Exception as e:
            print(f"Error grading documents: {str(e)}")
            filtered_docs = documents  # Fall back to all documents
            binary_score = "yes"  # Assume relevance to avoid web search failures
            error = f"Error in document grading: {str(e)}"
            
    # Return with the GraphState constructor
    return GraphState(
        context=filtered_docs,  # Update context with filtered docs
        answer=[],  # Still empty answer
        question=question,
        rewritten_question=state.get("rewritten_question", ""),  # Preserve rewritten question
        sql_query=state.get("sql_query", ""),  # Preserve existing SQL query
        binary_score=binary_score,  # Update binary score
        web_search=web_search,  # Update web search flag
        language=state.get("language", detect_language(question)),  # Preserve language
        error=error  # Add error if any
    )

def web_search(state: GraphState) -> GraphState:
    """
    Perform a web search when documents are not relevant.
    Supports both English and Korean language searches.

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with web search results
    """
    print("---PERFORMING WEB SEARCH---")
    
    question = state["question"]
    language = state.get("language", detect_language(question))
    error = None
    
    try:
        # Use rewritten question if available for better web search
        search_question = state.get("rewritten_question", question) if not state.get("error") else question
        
        # This is a placeholder for an actual web search implementation
        # In a real system, you would integrate with a search API
        if language == "ko":
            result_prefix = f"'{search_question}'에 대한 웹 검색 결과: "
        else:
            result_prefix = f"Web search result for: '{search_question}'. "
            
        web_results = [
            Document(page_content=result_prefix + 
                                 "This is placeholder content that would normally come from a web search API.")
        ]
    except Exception as e:
        print(f"Error in web search: {str(e)}")
        web_results = [Document(page_content="Failed to retrieve web search results.")]
        error = f"Error in web search: {str(e)}"
    
    return GraphState(
        context=web_results,  # Replace context with web results
        answer=[],  # Still empty answer
        question=question,
        rewritten_question=state.get("rewritten_question", ""),
        sql_query=state.get("sql_query", ""),
        binary_score="yes",  # Web results are considered relevant
        web_search="Completed",  # Mark web search as completed
        language=language,
        error=error
    )

def generate_sql_query(state: GraphState) -> GraphState:
    """
    Generate an SQL query based on the user question.
    Supports both English and Korean language questions.

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with SQL query
    """
    print("---GENERATING SQL QUERY---")
    
    question = state["question"]
    language = state.get("language", detect_language(question))
    error = None
    
    try:
        # Create language-specific prompt
        if language == "ko":
            prompt_text = f"""
            다음 질문에 대한 SQL 쿼리를 생성하세요.
            관련 테이블이 있는 데이터베이스가 있다고 가정합니다.
            
            질문: {question}
            
            SQL 쿼리:
            """
        else:
            prompt_text = f"""
            Generate a SQL query for the following question. 
            Assume we have a database with relevant tables.
            
            Question: {question}
            
            SQL Query:
            """
        
        # Generate SQL query
        response = llm.invoke(prompt_text)
        sql_query = response.content
    except Exception as e:
        print(f"Error generating SQL query: {str(e)}")
        sql_query = "-- Failed to generate SQL query"
        error = f"Error in SQL query generation: {str(e)}"
    
    # Return with the GraphState constructor
    return GraphState(
        context=state.get("context", []),  # Preserve context
        answer=state.get("answer", []),  # Preserve answer
        question=question,
        rewritten_question=state.get("rewritten_question", ""),  # Preserve rewritten question
        sql_query=sql_query,  # Update SQL query
        binary_score=state.get("binary_score", ""),  # Preserve binary score
        web_search=state.get("web_search", "No"),  # Preserve web search flag
        language=language,
        error=error
    )

def generate_answer(state: GraphState) -> GraphState:
    """
    Generate an answer based on the context and question.
    Supports both English and Korean language questions.

    Args:
        state (GraphState): The current graph state with context and question

    Returns:
        GraphState: Updated state with answer
    """
    print("---GENERATING ANSWER---")
    
    question = state["question"]
    context = state["context"]
    language = state.get("language", detect_language(question))
    error = None
    
    try:
        if not context:
            if language == "ko":
                answer_text = "이 질문에 답변할 만한 충분한 관련 정보가 없습니다."
            else:
                answer_text = "I don't have enough relevant information to answer this question."
            answer_doc = Document(page_content=answer_text)
        else:
            # Format context for the LLM
            context_text = "\n\n".join([doc.page_content for doc in context])
            
            # Create language-specific prompt
            if language == "ko":
                prompt_text = f"""
                제공된 컨텍스트를 기반으로 다음 질문에 답변하세요.
                
                컨텍스트:
                {context_text}
                
                질문:
                {question}
                
                답변:
                """
            else:
                prompt_text = f"""
                Answer the following question based on the provided context.
                
                Context:
                {context_text}
                
                Question:
                {question}
                
                Answer:
                """
            
            # Generate answer
            response = llm.invoke(prompt_text)
            answer_doc = Document(page_content=response.content)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        if language == "ko":
            answer_text = "죄송합니다. 답변을 생성하는 동안 오류가 발생했습니다."
        else:
            answer_text = "Sorry, an error occurred while generating the answer."
        answer_doc = Document(page_content=answer_text)
        error = f"Error in answer generation: {str(e)}"
    
    # Return with the GraphState constructor
    return GraphState(
        context=state.get("context", []),  # Preserve context
        answer=[answer_doc],  # Add the answer as a Document
        question=question,
        rewritten_question=state.get("rewritten_question", ""),  # Preserve rewritten question
        sql_query=state.get("sql_query", ""),  # Preserve existing SQL query
        binary_score=state.get("binary_score", ""),  # Preserve binary score
        web_search=state.get("web_search", "No"),  # Preserve web search flag
        language=language,
        error=error
    )

# Adding a generate function similar to the original notebook
def generate(state: GraphState) -> GraphState:
    """
    Generate answer

    Args:
        state (GraphState): The current graph state

    Returns:
        GraphState: Updated state with generation
    """
    
    print("---GENERATE---")
    
    question = state["question"]
    context = state["context"]
    language = state.get("language", detect_language(question))
    error = None
    
    try:
        # Assuming rag_chain is defined elsewhere and imported
        generation = rag_chain.invoke({"context": context, "question": question})
        answer_doc = Document(page_content=generation)
    except Exception as e:
        print(f"Error in RAG generation: {str(e)}")
        if language == "ko":
            answer_text = "죄송합니다. 답변을 생성하는 동안 오류가 발생했습니다."
        else:
            answer_text = "Sorry, an error occurred while generating the answer."
        answer_doc = Document(page_content=answer_text)
        error = f"Error in RAG generation: {str(e)}"
    
    # Return with the GraphState constructor
    return GraphState(
        context=context,
        answer=[answer_doc],  # Store generation as a Document in answer
        question=question,
        rewritten_question=state.get("rewritten_question", ""),  # Preserve rewritten question
        sql_query=state.get("sql_query", ""),
        binary_score=state.get("binary_score", ""),
        web_search=state.get("web_search", "No"),  # Preserve web search flag
        language=language,
        error=error
    )

# Define the router function
def router(state: GraphState) -> str:
    """
    Route to the appropriate next step based on the state.
    
    Args:
        state (GraphState): The current graph state
        
    Returns:
        str: The name of the next node in the graph
    """
    # If there's an error and not in web search yet, try web search as fallback
    if state.get("error") and state.get("web_search") != "Completed":
        return "web_search"
    
    # Check if we need to do a web search
    if state.get("web_search") == "Yes":
        return "web_search"
    
    # If documents are relevant, generate answer
    elif state.get("binary_score") == "yes":
        return "generate"
    
    # If we've completed web search, generate
    elif state.get("web_search") == "Completed":
        return "generate"
    
    # Default case
    else:
        return "generate"

# Example of setting up a StateGraph flow (commented out)
"""
from langgraph.graph import StateGraph, START, END

# Build the graph
graph = StateGraph(GraphState)

# Define nodes
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", grade_documents)
graph.add_node("web_search", web_search)
graph.add_node("generate_sql", generate_sql_query)
graph.add_node("generate", generate)

# Add edges
graph.add_edge(START, "rewrite_question")
graph.add_edge("rewrite_question", "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_edge("grade_documents", router)
graph.add_edge("web_search", "generate")
graph.add_edge("generate_sql", "retrieve")
graph.add_edge("generate", END)

# Compile the graph
app = graph.compile()
"""

# Example of a complete workflow with conditional paths
# 
# # Initialize with an initial state
# initial_state = GraphState(
#     context=[], 
#     answer=[], 
#     question="직원들의 평균 급여는 얼마인가요?",  # "What is the average salary of employees?"
#     rewritten_question="",
#     sql_query="",
#     binary_score="",
#     web_search="No",
#     language="",  # Will be auto-detected
#     error=None
# )
# 
# # Step 1: Detect language and rewrite the question for better search
# state_with_rewritten_q = rewrite_question(initial_state)
# 
# # Step 2: Generate SQL query for the question
# state_with_sql = generate_sql_query(state_with_rewritten_q)
# 
# # Step 3: Retrieve relevant documents
# state_with_docs = retrieve(state_with_sql)
# 
# # Step 4: Grade the retrieved documents
# state_with_graded_docs = grade_documents(state_with_docs)
# 
# # Step 5: Conditional path based on document relevance
# if state_with_graded_docs["web_search"] == "Yes":
#     # If documents aren't relevant, do web search
#     state_with_web_results = web_search(state_with_graded_docs)
#     # Then generate answer using web results
#     final_state = generate_answer(state_with_web_results)
# else:
#     # If documents are relevant, generate answer directly
#     final_state = generate_answer(state_with_graded_docs)
# 
# # The answer is in final_state["answer"][0].page_content
# # The SQL query is in final_state["sql_query"]
# # The rewritten question is in final_state["rewritten_question"] 