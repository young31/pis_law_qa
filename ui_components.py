# ui_components.py
import streamlit as st
from langchain_core.documents import Document

def display_results(result_data: dict, answer_mode: str):
    if not result_data:
        st.error("결과 데이터가 없습니다."); return
    
    mode_applied = result_data.get("mode_applied", answer_mode)
    st.markdown(f"##### 🤖 AI 답변 ({mode_applied})")
    answer = result_data.get("result", "답변 생성 실패/결과 없음.")
    st.markdown(answer if answer.strip() else "AI가 답변을 생성하지 못했습니다.")
    st.markdown("---")

    source_documents = result_data.get("source_documents")
    if source_documents:
        st.markdown("##### 📚 참고 문헌 (검색된 문서)")
        for i, doc in enumerate(source_documents):
            if not isinstance(doc, Document): continue
            doc_type = doc.metadata.get('doc_type', 'unknown')
            title = ""
            if doc_type == "qa_pair":
                q = doc.metadata.get('original_question', "질문 없음")
                title = f"참고 Q&A {i+1}: \"{q[:40].strip()}...\""
            elif doc_type == "full_document_context":
                title = f"참고 전체 문서 {i+1}: {doc.metadata.get('source_file')}"
            else: # legal_document_chunk
                article_title = doc.metadata.get('article_title_full', '조항 정보 없음')
                title = f"법률 문서 {i+1}: {doc.metadata.get('source_file')} - {article_title}"

            with st.expander(title):
                st.text(doc.page_content)
                meta = {k: v for k, v in doc.metadata.items() if k not in ["source_file", "article_title_full", "original_question"]}
                if meta: st.json(meta)