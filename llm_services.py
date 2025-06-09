# llm_services.py
import re
import time
import shutil
import logging
import streamlit as st
from pathlib import Path
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Import from our modules
from config import GOOGLE_API_KEY
from doc_processing import preprocess_text, split_text_by_article, extract_metadata_from_title

@st.cache_resource(show_spinner="Ollama 임베딩 모델 로드 중...")
def get_ollama_embeddings(model_name: str) -> Optional[OllamaEmbeddings]:
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        embeddings.embed_query("test")
        return embeddings
    except Exception as e:
        st.error(f"Ollama 임베딩 모델 로드 실패: {e}")
        return None

@st.cache_resource(show_spinner="Google AI 임베딩 모델 로드 중...")
def get_google_embeddings(model_name: str) -> Optional[GoogleGenerativeAIEmbeddings]:
    if not GOOGLE_API_KEY:
        st.error("Google API 키가 설정되지 않았습니다.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=GOOGLE_API_KEY)
        embeddings.embed_query("test")
        return embeddings
    except Exception as e:
        st.error(f"Google AI 임베딩 모델 로드 실패: {e}")
        return None

@st.cache_resource(show_spinner="Ollama LLM 로드 중...")
def get_ollama_llm(model_name: str, temperature: float = 0.1) -> Optional[ChatOllama]:
    try:
        llm = ChatOllama(model=model_name, temperature=temperature)
        llm.invoke("안녕")
        return llm
    except Exception as e:
        st.error(f"Ollama LLM 로드 실패: {e}")
        return None

@st.cache_resource(show_spinner="Gemini LLM 로드 중...")
def get_gemini_llm(model_name: str, temperature: float = 0.1) -> Optional[ChatGoogleGenerativeAI]:
    if not GOOGLE_API_KEY:
        st.error("Google API 키가 설정되지 않았습니다.")
        return None
    try:
        safety_settings = {category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory}
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=GOOGLE_API_KEY, safety_settings=safety_settings)
        llm.invoke("안녕, Gemini!")
        return llm
    except Exception as e:
        st.error(f"Gemini LLM 로드 실패: {e}")
        return None

def create_or_load_vectorstore(
    texts_with_sources: List, qa_documents: List, embeddings: Embeddings,
    emb_provider: str, emb_model: str, chunk_size: int, chunk_overlap: int, force_recreate: bool
) -> Optional[Chroma]:
    final_docs = []
    if texts_with_sources:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for name, text in texts_with_sources:
            processed_text = preprocess_text(text)
            articles = split_text_by_article(processed_text)
            for title, content in articles.items():
                metadata = extract_metadata_from_title(title)
                metadata.update({"source_file": name, "article_title_full": title, "doc_type": "legal_document_chunk"})
                article_doc = Document(page_content=content, metadata=metadata)
                chunks = splitter.split_documents([article_doc])
                final_docs.extend(chunks)

    if qa_documents: final_docs.extend(qa_documents)

    if not final_docs:
        st.error("DB를 생성할 문서가 없습니다."); return None

    safe_model_name = re.sub(r'[^a-zA-Z0-9_-]', '_', emb_model)
    db_path = Path(f"chroma_db_{emb_provider}_{safe_model_name}_cz{chunk_size}_co{chunk_overlap}")

    if force_recreate and db_path.exists():
        shutil.rmtree(db_path)

    if not force_recreate and db_path.exists():
        try: return Chroma(persist_directory=str(db_path), embedding_function=embeddings)
        except Exception as e: st.warning(f"기존 DB 로드 실패: {e}. 새로 생성합니다.")

    try: return Chroma.from_documents(documents=final_docs, embedding=embeddings, persist_directory=str(db_path))
    except Exception as e: st.error(f"새 DB 생성 실패: {e}"); return None