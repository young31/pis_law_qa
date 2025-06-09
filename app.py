import streamlit as st
import os
import re
from pathlib import Path
import logging

# Import from our new modules
from config import *
from doc_processing import load_text_documents_from_directory, load_pdf_documents_from_directory, load_qa_from_excel
from llm_services import get_ollama_llm, get_gemini_llm, get_ollama_embeddings, get_google_embeddings, create_or_load_vectorstore
from rag_pipelines import generate_answer_with_rag, generate_blended_answer, generate_direct_llm_answer
from ui_components import display_results
# --- Main Streamlit Application ---
def main():
    st.set_page_config(page_title="AI 법률 질의응답 시스템", layout="wide", initial_sidebar_state="expanded")
    st.title("⚖️ AI 법률 질의응답 시스템")

    # --- Session State Initialization ---
    session_defaults = {
        "llm": None,
        "llm_provider_name_loaded": "Ollama",
        "llm_model_name": DEFAULT_OLLAMA_LLM_MODEL,
        "embeddings": None,
        "embedding_provider_name_loaded": "Ollama",
        "embedding_model_name_loaded": DEFAULT_OLLAMA_EMBEDDING_MODEL,
        "vectorstore": None,
        "vectorstore_config_key_loaded": None,
        "db_status_message": "DB가 로드되지 않았습니다. 설정을 확인하고 'DB 준비' 버튼을 클릭하세요.",
        "db_ready": False,
        "texts_with_sources_cache": None,
        "full_texts_map_cache": {},
        "last_loaded_doc_dir": DEFAULT_DOC_DIRECTORY,
        "last_loaded_file_type": "txt",
        "uploaded_qa_file_name": None,
        "qa_documents_cache": None,
        "sidebar_doc_dir": DEFAULT_DOC_DIRECTORY,
        "sidebar_file_type": "txt",
        "sidebar_emb_provider": "Ollama",
        "sidebar_ollama_emb_model": DEFAULT_OLLAMA_EMBEDDING_MODEL,
        "sidebar_google_emb_model": DEFAULT_GOOGLE_EMBEDDING_MODEL,
        "sidebar_llm_provider": "Ollama",
        "sidebar_ollama_llm_model": DEFAULT_OLLAMA_LLM_MODEL,
        "sidebar_gemini_llm_model": DEFAULT_GEMINI_LLM_MODEL,
        "sidebar_chunk_size": CHUNK_SIZE_DEFAULT,
        "sidebar_chunk_overlap": CHUNK_OVERLAP_DEFAULT,
        "sidebar_force_recreate_db": False,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.caption(f"LLM ({st.session_state.llm_provider_name_loaded}: {st.session_state.llm_model_name if st.session_state.llm else '로드 안됨'}), "
               f"임베딩 ({st.session_state.embedding_provider_name_loaded}: "
               f"{st.session_state.embedding_model_name_loaded if st.session_state.embeddings else '로드 안됨'})")

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("⚙️ 시스템 설정")

        st.subheader("1. 법률 문서 로딩")
        st.session_state.sidebar_doc_dir = st.text_input("문서 디렉토리", value=st.session_state.sidebar_doc_dir, key="doc_dir_sb")
        st.session_state.sidebar_file_type = st.selectbox("문서 타입", ["txt", "pdf"], index=["txt", "pdf"].index(st.session_state.sidebar_file_type), key="file_type_sb")

        st.subheader("2. 임베딩 모델")
        st.session_state.sidebar_emb_provider = st.selectbox("제공자", ["Ollama", "Google"], index=["Ollama", "Google"].index(st.session_state.sidebar_emb_provider), key="emb_prov_sb")
        current_embedding_model_to_use = ""
        if st.session_state.sidebar_emb_provider == "Ollama":
            st.session_state.sidebar_ollama_emb_model = st.text_input("Ollama 모델명", value=st.session_state.sidebar_ollama_emb_model, key="ollama_emb_sb", help=f"예: {DEFAULT_OLLAMA_EMBEDDING_MODEL}")
            current_embedding_model_to_use = st.session_state.sidebar_ollama_emb_model
        elif st.session_state.sidebar_emb_provider == "Google":
            st.session_state.sidebar_google_emb_model = st.text_input("Google 모델명", value=st.session_state.sidebar_google_emb_model, key="google_emb_sb", disabled=not GOOGLE_API_KEY, help=f"예: {DEFAULT_GOOGLE_EMBEDDING_MODEL}")
            if not GOOGLE_API_KEY: st.warning("⚠️ GOOGLE_API_KEY 없음.")
            current_embedding_model_to_use = st.session_state.sidebar_google_emb_model

        st.subheader("3. LLM (답변 생성)")
        st.session_state.sidebar_llm_provider = st.selectbox(
            "LLM 제공자", ["Ollama", "Gemini"],
            index=["Ollama", "Gemini"].index(st.session_state.sidebar_llm_provider),
            key="llm_provider_sb", help="답변 생성에 사용할 LLM의 종류를 선택하세요."
        )
        if st.session_state.sidebar_llm_provider == "Ollama":
            st.session_state.sidebar_ollama_llm_model = st.text_input("Ollama LLM 모델명", value=st.session_state.sidebar_ollama_llm_model, key="ollama_llm_sb", help=f"예: {DEFAULT_OLLAMA_LLM_MODEL}")
        elif st.session_state.sidebar_llm_provider == "Gemini":
            st.session_state.sidebar_gemini_llm_model = st.text_input("Gemini LLM 모델명", value=st.session_state.sidebar_gemini_llm_model, key="gemini_llm_sb", disabled=not GOOGLE_API_KEY, help="예: gemini-2.0-flash, gemini-2.0-flash-lite-001")
            if not GOOGLE_API_KEY: st.warning("⚠️ GOOGLE_API_KEY 없음.")

        st.subheader("4. 벡터 DB 설정")
        st.session_state.sidebar_chunk_size = st.slider("Chunk Size", 200, 4000, st.session_state.sidebar_chunk_size, 50, key="chunk_size_sb")
        st.session_state.sidebar_chunk_overlap = st.slider("Chunk Overlap", 0, 1000, st.session_state.sidebar_chunk_overlap, 50, key="chunk_overlap_sb")
        
        st.subheader("5. 이전 Q&A 데이터 (선택)")
        uploaded_qa_file_ui = st.file_uploader("Q&A 참고자료 Excel (.xlsx, .xls)", type=["xlsx", "xls"], key="qa_file_uploader_sb")
        
        st.session_state.sidebar_force_recreate_db = st.checkbox("DB 강제 재생성", value=st.session_state.sidebar_force_recreate_db, key="force_recreate_sb")

    # --- LLM Loading/Re-loading Logic ---
    desired_llm_provider = st.session_state.sidebar_llm_provider
    if desired_llm_provider == "Ollama":
        desired_llm_model = st.session_state.sidebar_ollama_llm_model
    else:  # Gemini
        desired_llm_model = st.session_state.sidebar_gemini_llm_model

    llm_needs_update = (
        st.session_state.llm is None or
        st.session_state.llm_provider_name_loaded != desired_llm_provider or
        st.session_state.llm_model_name != desired_llm_model
    )
    if llm_needs_update and desired_llm_model:
        with st.spinner(f"{desired_llm_provider} LLM ({desired_llm_model}) 로드 중..."):
            llm_instance = None
            if desired_llm_provider == "Ollama":
                llm_instance = get_ollama_llm(desired_llm_model)
            elif desired_llm_provider == "Gemini":
                llm_instance = get_gemini_llm(desired_llm_model)

            if llm_instance:
                st.session_state.llm = llm_instance
                st.session_state.llm_provider_name_loaded = desired_llm_provider
                st.session_state.llm_model_name = desired_llm_model
                logging.info(f"LLM switched/loaded: {desired_llm_provider} - {desired_llm_model}")
            else:
                if st.session_state.llm:
                    st.sidebar.error(f"새 LLM({desired_llm_provider} - {desired_llm_model}) 로드 실패. 이전 LLM을 계속 사용합니다.")
                else:
                    st.session_state.llm = None
                    st.session_state.llm_provider_name_loaded = desired_llm_provider
                    st.session_state.llm_model_name = desired_llm_model
                    st.sidebar.error(f"LLM({desired_llm_provider} - {desired_llm_model}) 로드 실패. 답변 생성이 불가능합니다.")

    if st.session_state.llm:
        st.sidebar.success(f"LLM ({st.session_state.llm_provider_name_loaded}: {st.session_state.llm_model_name}) 준비 완료")
    else:
        failed_provider = st.session_state.sidebar_llm_provider
        failed_model = st.session_state.sidebar_ollama_llm_model if failed_provider == "Ollama" else st.session_state.sidebar_gemini_llm_model
        st.sidebar.error(f"LLM ({failed_provider}: {failed_model}) 로드 실패 또는 안됨.")

    # --- Main Area ---
    col1, col2 = st.columns([1.2, 2])
    with col1:
        st.subheader("💾 데이터베이스 관리")
        st.info(st.session_state.db_status_message)

        safe_sidebar_emb_model_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', current_embedding_model_to_use).replace('/', '_').replace(':','_')
        current_sidebar_db_config_key = (f"chroma_db_{st.session_state.sidebar_emb_provider.lower()}_{safe_sidebar_emb_model_name}_cz{st.session_state.sidebar_chunk_size}_co{st.session_state.sidebar_chunk_overlap}")
        
        if st.session_state.db_ready and st.session_state.vectorstore_config_key_loaded != current_sidebar_db_config_key:
            st.warning("⚠️ DB 설정 변경됨. DB 재로드 필요.")

        button_text = f"'{st.session_state.sidebar_doc_dir}'({st.session_state.sidebar_file_type.upper()}) 로드 및 DB 준비"
        if st.session_state.sidebar_force_recreate_db: button_text += " (강제 재생성)"
        
        if st.button(button_text, key="prepare_db_main", use_container_width=True):
            st.session_state.db_ready = False
            st.session_state.db_status_message = "DB 처리 시작..."
            
            # 1. Initialize Embeddings
            embeddings_instance, emb_init_success = None, False
            with st.spinner(f"{st.session_state.sidebar_emb_provider} 임베딩 ({current_embedding_model_to_use}) 초기화..."):
                if st.session_state.sidebar_emb_provider == "Ollama":
                    if current_embedding_model_to_use: embeddings_instance = get_ollama_embeddings(current_embedding_model_to_use)
                    else: st.error("Ollama 임베딩 모델명 미지정.")
                elif st.session_state.sidebar_emb_provider == "Google":
                    if not GOOGLE_API_KEY: st.error("Google API 키 없음.")
                    elif current_embedding_model_to_use: embeddings_instance = get_google_embeddings(current_embedding_model_to_use)
                    else: st.error("Google 임베딩 모델명 미지정.")
                if embeddings_instance: emb_init_success = True
            
            if not emb_init_success:
                st.session_state.db_status_message = "❌ 임베딩 모델 초기화 실패."
                st.session_state.embeddings = None
                st.rerun()

            st.session_state.embeddings = embeddings_instance
            st.session_state.embedding_provider_name_loaded = st.session_state.sidebar_emb_provider
            st.session_state.embedding_model_name_loaded = current_embedding_model_to_use
            logging.info(f"Embeddings set: {st.session_state.embedding_provider_name_loaded} - {st.session_state.embedding_model_name_loaded}")
            
            # 2. Load Q&A documents
            current_qa_docs = []
            if uploaded_qa_file_ui is not None:
                if (st.session_state.uploaded_qa_file_name != uploaded_qa_file_ui.name) or \
                   st.session_state.qa_documents_cache is None or st.session_state.sidebar_force_recreate_db:
                    with st.spinner(f"'{uploaded_qa_file_ui.name}' Q&A 데이터 로드..."):
                        loaded_qa_docs = load_qa_from_excel(uploaded_qa_file_ui)
                    st.session_state.qa_documents_cache = loaded_qa_docs if loaded_qa_docs else []
                    st.session_state.uploaded_qa_file_name = uploaded_qa_file_ui.name
                    if loaded_qa_docs: st.success(f"'{uploaded_qa_file_ui.name}'에서 {len(loaded_qa_docs)}개 Q&A 로드 완료.")
                current_qa_docs = st.session_state.qa_documents_cache
            else:
                if st.session_state.uploaded_qa_file_name is not None: st.info("Q&A Excel 파일 제거됨.")
                st.session_state.qa_documents_cache, st.session_state.uploaded_qa_file_name = [], None

            current_qa_docs = st.session_state.qa_documents_cache if st.session_state.qa_documents_cache else []

            # 3. Load primary documents
            texts_sources_for_db = None
            should_load_fresh_docs = (st.session_state.last_loaded_doc_dir != st.session_state.sidebar_doc_dir or
                                      st.session_state.last_loaded_file_type != st.session_state.sidebar_file_type or
                                      st.session_state.texts_with_sources_cache is None or st.session_state.sidebar_force_recreate_db)
            if should_load_fresh_docs:
                if not os.path.isdir(st.session_state.sidebar_doc_dir):
                    st.error(f"문서 디렉토리 '{st.session_state.sidebar_doc_dir}' 찾을 수 없음."); st.session_state.texts_with_sources_cache = None; st.rerun()
                load_func = load_text_documents_from_directory if st.session_state.sidebar_file_type == "txt" else load_pdf_documents_from_directory
                with st.spinner(f"'{st.session_state.sidebar_doc_dir}' ({st.session_state.sidebar_file_type.upper()}) 로드..."):
                    texts_sources_for_db = load_func(st.session_state.sidebar_doc_dir)
                if texts_sources_for_db:
                    st.session_state.texts_with_sources_cache = texts_sources_for_db
                    st.session_state.full_texts_map_cache = dict(texts_sources_for_db)
                    st.session_state.last_loaded_doc_dir, st.session_state.last_loaded_file_type = st.session_state.sidebar_doc_dir, st.session_state.sidebar_file_type
                else: st.warning(f"'{st.session_state.sidebar_doc_dir}' ({st.session_state.sidebar_file_type})에서 문서 로드 실패."); st.session_state.texts_with_sources_cache = None; st.session_state.full_texts_map_cache = {}; st.rerun()
            else:
                texts_sources_for_db = st.session_state.texts_with_sources_cache
                if texts_sources_for_db: st.info(f"캐시된 법률 문서 {len(texts_sources_for_db)}개 사용.")

            # 4. Create/Load Vectorstore
            if (texts_sources_for_db or current_qa_docs) or not st.session_state.sidebar_force_recreate_db:
                with st.spinner("벡터 DB 구성 중... (시간 소요될 수 있음)"):
                    vector_db_instance = create_or_load_vectorstore(
                        texts_sources_for_db, current_qa_docs, st.session_state.embeddings,
                        st.session_state.embedding_provider_name_loaded, st.session_state.embedding_model_name_loaded,
                        st.session_state.sidebar_chunk_size, st.session_state.sidebar_chunk_overlap,
                        st.session_state.sidebar_force_recreate_db
                    )
                if vector_db_instance:
                    st.session_state.vectorstore, st.session_state.vectorstore_config_key_loaded, st.session_state.db_ready = vector_db_instance, current_sidebar_db_config_key, True
                    st.session_state.db_status_message = f"✅ DB 준비 완료: {st.session_state.embedding_model_name_loaded} (Chunk: {st.session_state.sidebar_chunk_size}/{st.session_state.sidebar_chunk_overlap})"
                    if st.session_state.sidebar_force_recreate_db: st.session_state.sidebar_force_recreate_db = False
                else:
                    st.session_state.vectorstore, st.session_state.db_ready = None, False
                    st.session_state.db_status_message = "❌ 벡터 DB 준비 실패. 로그 확인."
            else:
                st.session_state.db_status_message = "❌ DB 강제 재생성: 추가할 법률/Q&A 데이터 없음."
            st.rerun()

    with col2:
        st.subheader("💬 질의응답")
        answer_mode = st.radio("답변 방식:", ["RAG (문서 기반)", "Blended (RAG + LLM 개선)", "LLM 직접 답변"], index=1, key="answer_mode_main", horizontal=True,
                               help=("**RAG**: 문서 기반 답변.\n**Blended**: RAG 답변을 개선하고, 실패 시 전체 문서를 참고하여 재시도.\n**LLM 직접 답변**: LLM 자체 지식으로 답변."))
        
        llm_ready = bool(st.session_state.llm)
        db_compatible = st.session_state.db_ready and st.session_state.vectorstore is not None and (st.session_state.vectorstore_config_key_loaded == current_sidebar_db_config_key)
        can_ask = False

        if not llm_ready:
            st.warning("LLM이 준비되지 않았습니다. 사이드바 설정 확인 후 잠시 기다려주세요.")
        
        if answer_mode == "LLM 직접 답변":
            if llm_ready: can_ask = True
        else:  # RAG or Blended
            if not llm_ready: pass
            elif not st.session_state.db_ready: st.warning("DB가 로드되지 않았습니다. 왼쪽 'DB 준비' 버튼을 클릭해주세요.")
            elif not db_compatible: st.warning("로드된 DB가 현재 사이드바 설정과 다릅니다. DB를 현재 설정으로 다시 준비해주세요.")
            else: can_ask = True

        question = st.text_area("질문 입력:", height=100, key="user_question_main", placeholder="법률 관련 질문 입력...", disabled=not can_ask)
        
        if st.button("답변 생성 🚀", key="ask_btn_main", disabled=not can_ask or not question.strip(), use_container_width=True):
            with st.spinner("AI 답변 생성 중... 🤔"):
                final_result_data = None
                if answer_mode != "LLM 직접 답변" and not st.session_state.vectorstore: st.error("벡터스토어 미준비.")
                elif not st.session_state.llm: st.error("LLM 미준비.")
                else:
                    if answer_mode == "RAG (문서 기반)":
                        final_result_data = generate_answer_with_rag(question, st.session_state.vectorstore, st.session_state.llm)
                    elif answer_mode == "Blended (RAG + LLM 개선)":
                        full_texts_map = st.session_state.get("full_texts_map_cache", {})
                        final_result_data = generate_blended_answer(question, st.session_state.vectorstore, st.session_state.llm, full_texts_map)
                    elif answer_mode == "LLM 직접 답변":
                        final_result_data = generate_direct_llm_answer(question, st.session_state.llm)
                
                if final_result_data:
                    display_results(final_result_data, answer_mode)
        elif not question.strip() and can_ask:
            st.info("답변을 생성하려면 질문을 입력해주세요.")

if __name__ == "__main__":
    Path(DEFAULT_DOC_DIRECTORY).mkdir(parents=True, exist_ok=True)
    logging.info(f"Default document directory ensured: '{DEFAULT_DOC_DIRECTORY}'")
    main()