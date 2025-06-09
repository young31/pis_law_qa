# doc_processing.py
import os
import re
import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader

@st.cache_data(show_spinner="텍스트 파일 로드 중...")
def load_text_documents_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    return _load_documents(directory_path, "txt")

@st.cache_data(show_spinner="PDF 파일 로드 중...")
def load_pdf_documents_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    return _load_documents(directory_path, "pdf")

def _load_documents(directory_path: str, file_type: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(directory_path):
        st.error(f"디렉토리 '{directory_path}'를 찾을 수 없습니다.")
        return []
    glob_pattern = f"**/*.{file_type.lower()}"
    try:
        loader_cls = TextLoader if file_type == "txt" else PyPDFLoader
        loader_kwargs = {'encoding': 'utf-8', 'autodetect_encoding': True} if file_type == "txt" else {}
        loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs,
                                 recursive=True, show_progress=False, use_multithreading=True, silent_errors=True)
        docs = loader.load()
    except Exception as e:
        st.error(f"'{directory_path}'에서 문서 로드 중 오류: {e}")
        return []

    docs_by_source = {}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        docs_by_source.setdefault(source, []).append(doc.page_content or "")
    
    texts_with_sources = []
    for source, content_list in docs_by_source.items():
        full_text = "\n".join(content_list).strip()
        if full_text:
            texts_with_sources.append((Path(source).name, full_text))
            
    if not texts_with_sources: st.warning(f"'{directory_path}'에서 유효한 텍스트를 추출하지 못했습니다.")
    return texts_with_sources

@st.cache_data(show_spinner="이전 Q&A 데이터 로드 중...")
def load_qa_from_excel(uploaded_file) -> List[Document]:
    if uploaded_file is None: return []
    try:
        df = pd.read_excel(uploaded_file)
        possible_q = ["Question", "질문", "질의"]
        possible_a = ["Answer", "답변", "응답"]
        q_col = next((c for c in df.columns if c in possible_q), None)
        a_col = next((c for c in df.columns if c in possible_a), None)

        if not q_col or not a_col:
            st.error(f"Excel 파일에서 질문/답변 컬럼을 찾을 수 없습니다. ({possible_q} / {possible_a})")
            return []

        qa_docs = []
        for _, row in df.iterrows():
            q_text = str(row[q_col]) if pd.notna(row[q_col]) else ""
            a_text = str(row[a_col]) if pd.notna(row[a_col]) else ""
            if q_text.strip() and a_text.strip():
                content = f"이전 질문: {q_text}\n이전 답변: {a_text}"
                metadata = {"source_file": uploaded_file.name, "doc_type": "qa_pair", "original_question": q_text}
                qa_docs.append(Document(page_content=content, metadata=metadata))
        return qa_docs
    except Exception as e:
        st.error(f"Excel 파일 처리 중 오류: {e}")
        return []

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return ""
    lines = text.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if any(keyword in line for keyword in ["법제처 국가법령정보센터", "www.law.go.kr"]): continue
        if re.match(r'^\s*page\s*\d+\s*/\s*\d+\s*$', line.lower()): continue
        line = re.sub(r'<[^>]+>|\[.*?\]', '', line)
        if re.match(r'^\s*제\d+조(?:의\d+)?\s+<삭제>', line): continue
        line = re.sub(r'\s+', ' ', line).strip()
        if line: cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def split_text_by_article(processed_text: str) -> Dict[str, str]:
    articles = {}
    current_title, current_content = None, []
    pattern = re.compile(r'^(제\s*\d+조(?:의\d+)?\s*\([^)]+\))', re.IGNORECASE)

    for line in processed_text.strip().split('\n'):
        match = pattern.match(line)
        if match:
            if current_title and current_content:
                articles[current_title] = "\n".join(current_content).strip()
            current_title = match.group(1)
            current_content = [line]
        elif current_title:
            current_content.append(line)
    if current_title and current_content:
        articles[current_title] = "\n".join(current_content).strip()
        
    if not articles and processed_text:
        articles["전체 문서"] = processed_text
    return articles

def extract_metadata_from_title(article_title: str) -> Dict[str, str]:
    if article_title == "전체 문서": return {"조_번호": "N/A", "조_제목": "전체 문서"}
    match = re.match(r'제\s*(\d+(?:조)?(?:의\s*\d+)?)?(?:\s*조)?\s*\(([^)]+)\)', article_title, re.IGNORECASE)
    if match:
        num_part = re.sub(r'[조\s]', '', match.group(1) or "").strip()
        return {"조_번호": num_part or "번호 불명", "조_제목": match.group(2).strip() or "제목 없음"}
    return {"조_번호": "N/A", "조_제목": article_title}