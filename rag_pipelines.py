# rag_pipelines.py
import logging
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# Import from our modules
from config import DEFAULT_RAG_PROMPT_TEMPLATE, REFINEMENT_PROMPT_TEMPLATE, FULL_DOC_FALLBACK_PROMPT_TEMPLATE

def format_docs_for_context(docs: List[Document]) -> str:
    if not docs: return "제공된 문맥 정보가 없습니다."
    formatted_list = []
    for i, doc in enumerate(docs):
        doc_type = doc.metadata.get('doc_type', 'unknown')
        header, content = "", doc.page_content or ""
        if doc_type == "qa_pair":
            q = doc.metadata.get('original_question', '질문 없음')
            header = f"참고 Q&A {i+1} (출처: {doc.metadata.get('source_file', 'Excel')}, 원본 질문: \"{q[:40].strip()}...\")"
        elif doc_type == "legal_document_chunk":
            title = doc.metadata.get('article_title_full', '전체 문서 일부')
            header = f"법률 문서 {i+1}: '{doc.metadata.get('source_file', '출처 불명')}' (관련 조항 추정: {title})"
        else: # full_document_context or unknown
            header = f"참고 문서 {i+1}: '{doc.metadata.get('source_file', '출처 불명')}'"
        
        content_preview = (content[:700] + "...") if len(content) > 700 else content
        formatted_list.append(f"{header}\n추출 내용:\n{content_preview}")
    return "\n\n---\n\n".join(formatted_list)

def generate_answer_with_rag(question: str, vectorstore: Chroma, llm) -> Dict:
    try:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
        prompt = PromptTemplate.from_template(DEFAULT_RAG_PROMPT_TEMPLATE)
        rag_chain = {"context": retriever | format_docs_for_context, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        answer = rag_chain.invoke(question)
        docs = retriever.invoke(question)
        return {"result": answer, "source_documents": docs}
    except Exception as e:
        return {"result": f"RAG 답변 생성 중 오류: {e}", "source_documents": []}

def generate_direct_llm_answer(question: str, llm) -> Dict:
    try:
        return {"result": llm.invoke(f"법률 전문가로서 다음 질문에 대해 답변해 주세요:\n\n{question}").content, "source_documents": None}
    except Exception as e:
        return {"result": f"LLM 직접 답변 생성 중 오류: {e}", "source_documents": None}

def generate_llm_answer_with_given_context(question: str, llm, context_str: str, source_desc: str) -> Dict:
    try:
        prompt = PromptTemplate.from_template(FULL_DOC_FALLBACK_PROMPT_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context_str, "source_description": source_desc})
        return {"result": answer}
    except Exception as e:
        return {"result": f"전체 문서 기반 답변 생성 중 오류: {e}"}

def generate_blended_answer(question: str, vectorstore: Chroma, llm, full_texts_map: Dict) -> Dict:
    rag_result = generate_answer_with_rag(question, vectorstore, llm)
    rag_answer = rag_result.get("result", "")
    rag_docs = rag_result.get("source_documents", [])

    is_not_proper = not rag_answer or "오류" in rag_answer or "답변할 수 없습니다" in rag_answer
    if is_not_proper and rag_docs:
        logging.info("RAG 답변 부적절. 전체 문서로 폴백 시도.")
        top_source_file = rag_docs[0].metadata.get("source_file")
        if top_source_file and top_source_file in full_texts_map:
            full_content = full_texts_map[top_source_file]
            fallback_result = generate_llm_answer_with_given_context(question, llm, full_content, top_source_file)
            fallback_docs = [Document(page_content=full_content[:1000]+"...", metadata={"source_file": top_source_file, "doc_type": "full_document_context"})]
            return {**fallback_result, "source_documents": fallback_docs, "mode_applied": "fallback_full_doc"}
        return rag_result

    # Refinement logic
    try:
        refinement_prompt = PromptTemplate.from_template(REFINEMENT_PROMPT_TEMPLATE)
        refinement_chain = refinement_prompt | llm | StrOutputParser()
        refined_answer = refinement_chain.invoke({
            "original_question": question,
            "retrieved_context": format_docs_for_context(rag_docs),
            "rag_answer": rag_answer
        })
        return {"result": refined_answer, "source_documents": rag_docs, "mode_applied": "rag_plus_refinement"}
    except Exception as e:
        logging.warning(f"답변 개선 중 오류: {e}. 초기 RAG 답변 반환.")
        return rag_result