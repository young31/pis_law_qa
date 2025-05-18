# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import logging
from pathlib import Path
import dotenv
from typing import List, Tuple, Dict, Optional, Any
import time
import pandas as pd

# LangChain components
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
else:
    logging.warning("GOOGLE_API_KEY environment variable not found.")

# --- Constants ---
DEFAULT_DOC_DIRECTORY = "./laws_data_streamlit_app"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "mxbai-embed-large" # or "nomic-embed-text"
DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
DEFAULT_OLLAMA_LLM_MODEL = "phi4-mini" #  Adjust if needed, e.g., "llama3", "qwen2"
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 150

DEFAULT_RAG_PROMPT_TEMPLATE = """당신은 법률 조항에 대해 답변하는 AI 어시스턴트입니다. 오직 주어진 문맥 정보를 바탕으로 사용자의 질문에 대해 명확하고 간결하게 답변해주세요.
주어진 문맥에는 관련 법률 문서의 일부와 함께, 참고할 만한 이전 질의응답(Q&A) 기록이 포함될 수 있습니다. 이러한 Q&A 기록은 질문의 의도를 파악하거나 답변의 방향을 잡는 데 도움이 될 수 있으니 적극적으로 참고해주세요.
답변은 반드시 제공된 법률 문서 및 Q&A 문맥에 근거해야 하며, 문맥에 없는 내용은 절대로 언급하거나 추측하지 마세요.
만약 문맥에서 답변을 찾을 수 없다면, "제공된 문맥 정보만으로는 질문에 답변할 수 없습니다."라고 명확히 답변해주세요.
가능하다면 관련 법 조항의 제목이나 번호를 자연스럽게 언급해주세요. Q&A 내용을 참고했다면, 그 사실을 직접적으로 언급할 필요는 없으나 답변 내용에 자연스럽게 반영해주세요.

문맥:
{context}

질문: {question}

답변 (한국어, 문맥 근거):"""

REFINEMENT_PROMPT_TEMPLATE = """당신은 AI 법률 어시스턴트입니다. 사용자의 질문에 대해 검색된 관련 문서(문맥)와 Q&A 기록을 바탕으로 생성된 초기 답변이 주어졌습니다. 이 초기 답변과 문맥 정보를 사용하여, 사용자의 원본 질문에 대해 더 정확하고, 포괄적이며, 자연스러운 최종 답변을 생성해 주세요.
최종 답변은 반드시 제공된 **문맥 정보**(법률 문서, Q&A 기록 포함)와 **초기 답변**에 근거해야 합니다. 문맥에 없는 내용은 추가하지 마세요.
답변의 흐름을 개선하고, 명확성을 높이세요. 관련된 법 조항을 자연스럽게 언급하면 좋습니다. 초기 답변이 "제공된 문맥 정보만으로는 질문에 답변할 수 없습니다."와 같이 답변을 찾지 못한 경우, 최종 답변도 동일하게 또는 유사한 의미로 답변해야 합니다. 최종 답변은 한국어로 작성해주세요.

**원본 질문:** {original_question}

**검색된 문맥 정보:**
{retrieved_context}

**초기 RAG 답변:**
{rag_answer}

**개선된 최종 답변 (한국어, 문맥 및 초기 답변 근거):**"""

# --- Helper Functions ---
@st.cache_data(show_spinner="텍스트 파일 로드 중...")
def load_text_documents_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    return _load_documents(directory_path, "txt")

@st.cache_data(show_spinner="PDF 파일 로드 중...")
def load_pdf_documents_from_directory(directory_path: str) -> List[Tuple[str, str]]:
    return _load_documents(directory_path, "pdf")

def _load_documents(directory_path: str, file_type: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(directory_path):
        logging.error(f"Directory not found: {directory_path}")
        st.error(f"디렉토리 '{directory_path}'를 찾을 수 없습니다. 올바른 경로인지 확인해주세요.")
        return []
    glob_pattern = f"**/*.{file_type.lower()}"
    try:
        common_loader_params = {"recursive": True, "show_progress": False, "use_multithreading": True, "silent_errors": True}
        if file_type == "pdf":
            loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=PyPDFLoader, **common_loader_params)
        elif file_type == "txt":
            loader = DirectoryLoader(directory_path, glob=glob_pattern, loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True}, **common_loader_params)
        else:
            logging.error(f"Unsupported file type: {file_type}"); st.error(f"지원하지 않는 파일 타입입니다: {file_type}"); return []
        logging.info(f"Loading {file_type} documents from '{directory_path}' using glob: '{glob_pattern}'")
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} document sections initially from DirectoryLoader.")
        if not docs:
            logging.warning(f"No documents loaded from '{directory_path}' with pattern '{glob_pattern}'."); st.warning(f"'{directory_path}'에서 '{glob_pattern}' 패턴으로 로드된 문서가 없습니다."); return []
    except Exception as e:
        logging.error(f"Error during document loading from {directory_path} (file_type: {file_type}): {e}", exc_info=True); st.error(f"디렉토리 '{directory_path}'에서 문서를 로드하는 중 심각한 오류 발생: {e}"); return []

    docs_by_source: Dict[str, List[str]] = {}
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown Source'); page_content = doc.page_content if doc.page_content else ""
        if source not in docs_by_source: docs_by_source[source] = []
        docs_by_source[source].append(page_content)

    texts_with_sources: List[Tuple[str, str]] = []
    for source, content_list in docs_by_source.items():
        full_text = "\n".join(content_list).strip()
        if not full_text:
            logging.warning(f"File '{Path(source).name}' resulted in empty text. Skipping.")
            continue
        texts_with_sources.append((Path(source).name, full_text))

    logging.info(f"Extracted text from {len(texts_with_sources)} source files.")
    if not texts_with_sources: st.warning("로드된 파일에서 유효한 텍스트를 추출하지 못했습니다.")
    return texts_with_sources

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        logging.warning(f"preprocess_text received non-string: {type(text)}")
        return ""
    lines = text.strip().split("\n"); cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        if "법제처 국가법령정보센터" in line or "www.law.go.kr" in line or "National Law Information Center" in line: continue
        if re.match(r'^\s*page\s*\d+\s*/\s*\d+\s*$', line.lower()): continue
        line = re.sub(r'<[^>]+>', '', line); line = re.sub(r'\[.*?\]', '', line) # Remove HTML tags and bracketed content
        if re.match(r'^\s*제\d+조(?:의\d+)?\s+(?:<삭제>|삭제)\s*(\(\s*\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*\))?$', line.strip()) or \
           re.match(r'^\s*\d+\.\s+(?:<삭제>|삭제)\s*$', line.strip()): continue
        line = re.sub(r'\b\d{2,4}-\d{3,4}-\d{4}\b', '', line) # Remove phone numbers
        line = re.sub(r'https?://\S+', '', line) # Remove URLs
        line = re.sub(r'\s+', ' ', line).strip() # Normalize whitespace
        if line: cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def split_text_by_article(processed_text: str) -> Dict[str, str]:
    articles: Dict[str, str] = {}
    current_article_title: Optional[str] = None
    current_article_content: List[str] = []
    article_start_pattern = re.compile(r'^(제\s*(\d+(?:조)?(?:의\s*\d+)?)(?:\s*조)?\s*\(([^)]+)\))\s*(.*)', re.IGNORECASE)
    chapter_pattern = re.compile(r'^제\s*\d+\s*장\s+.*', re.IGNORECASE)
    annex_pattern = re.compile(r'^(부\s*칙|별\s*표)', re.IGNORECASE)

    lines = processed_text.strip().split('\n')
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line: continue

        if annex_pattern.match(line):
            if current_article_title and current_article_content:
                article_text = "\n".join(current_article_content).strip()
                if article_text and article_text != current_article_title:
                    articles[current_article_title] = article_text
            current_article_title = None
            current_article_content = []
            logging.debug(f"Annex/Appendix found: '{line}'. Stopping article splitting for this section.")
            continue

        match = article_start_pattern.match(line)
        if match:
            if current_article_title and current_article_content:
                article_text = "\n".join(current_article_content).strip()
                if article_text and article_text != current_article_title:
                    articles[current_article_title] = article_text
            current_article_title = match.group(1).strip()
            initial_content_on_same_line = match.group(4).strip()
            current_article_content = [current_article_title]
            if initial_content_on_same_line:
                current_article_content.append(initial_content_on_same_line)
            logging.debug(f"Started new article: '{current_article_title}'")
        elif current_article_title:
            if chapter_pattern.match(line):
                logging.debug(f"Skipping chapter line '{line}' within article '{current_article_title}'")
                continue
            current_article_content.append(line)

    if current_article_title and current_article_content:
        article_text = "\n".join(current_article_content).strip()
        if article_text and article_text != current_article_title:
             articles[current_article_title] = article_text
    articles = {title: content for title, content in articles.items() if content.strip() and content != title}
    if not articles and processed_text.strip():
        logging.warning("Article splitting did not find specific articles. Using '전체 문서' as a single article.")
        articles["전체 문서"] = processed_text
    elif not articles:
        logging.info("No articles found after splitting.")
    else:
        logging.info(f"Split text into {len(articles)} articles.")
    return articles

def extract_metadata_from_title(article_title: str) -> Dict[str, str]:
    metadata = {"조_번호": "N/A", "조_제목": "내용 없음"}
    if article_title == "전체 문서":
        return {"조_번호": "N/A", "조_제목": "전체 문서"}
    match = re.match(r'제\s*(\d+(?:조)?(?:의\s*\d+)?)?(?:\s*조)?\s*\(([^)]+)\)', article_title, re.IGNORECASE)
    if match:
        raw_number_part = match.group(1)
        if raw_number_part: metadata["조_번호"] = re.sub(r'[조\s]', '', raw_number_part).strip()
        else: metadata["조_번호"] = "번호 불명"
        title_candidate = match.group(2).strip().rstrip(')')
        metadata["조_제목"] = title_candidate if title_candidate else "제목 없음"
    else:
        logging.warning(f"Could not parse article number/title from: '{article_title}'."); metadata["조_제목"] = article_title
    return metadata

@st.cache_data(show_spinner="이전 Q&A 데이터 로드 중...")
def load_qa_from_excel(uploaded_file) -> List[Document]:
    if uploaded_file is None:
        return []
    try:
        df = pd.read_excel(uploaded_file)
        qa_docs = []
        possible_q_cols = ["Question", "질문", "질의", "질문 내용"]
        possible_a_cols = ["Answer", "답변", "응답", "답변 내용"]
        question_col, answer_col = None, None

        for col in df.columns:
            if col in possible_q_cols: question_col = col; break
        for col in df.columns:
            if col in possible_a_cols: answer_col = col; break

        if not question_col or not answer_col:
            st.error(f"Excel 파일에서 질문 컬럼({', '.join(possible_q_cols)}) 또는 답변 컬럼({', '.join(possible_a_cols)})을 찾을 수 없습니다.")
            return []

        for index, row in df.iterrows():
            question = str(row[question_col]) if pd.notna(row[question_col]) else ""
            answer = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
            if not question.strip() or not answer.strip():
                logging.warning(f"Skipping empty Q or A at Excel row {index+2} in '{uploaded_file.name}'")
                continue
            page_content = f"이전 질문: {question}\n이전 답변: {answer}"
            metadata = {
                "source_file": uploaded_file.name,
                "doc_type": "qa_pair",
                "original_question": question,
                "row_number": index + 2
            }
            for col_name in df.columns:
                if col_name not in [question_col, answer_col] and pd.notna(row[col_name]):
                    metadata[f"excel_{col_name.lower().replace(' ', '_')}"] = str(row[col_name])
            qa_docs.append(Document(page_content=page_content, metadata=metadata))

        logging.info(f"Loaded {len(qa_docs)} Q&A pairs from '{uploaded_file.name}'")
        if qa_docs: pass
        elif df.empty: st.warning(f"'{uploaded_file.name}' 파일이 비어있거나 읽을 수 있는 데이터가 없습니다.")
        else: st.warning(f"'{uploaded_file.name}'에서 유효한 Q&A 데이터를 찾지 못했습니다. 질문/답변 내용 또는 컬럼명을 확인해주세요.")
        return qa_docs
    except Exception as e:
        logging.error(f"Error loading Q&A from Excel ('{uploaded_file.name}'): {e}", exc_info=True)
        st.error(f"Excel 파일 ('{uploaded_file.name}') 처리 중 오류 발생: {e}")
        return []

@st.cache_resource(show_spinner="Ollama 임베딩 모델 로드 중...")
def get_ollama_embeddings(model_name: str) -> Optional[OllamaEmbeddings]:
    try:
        embeddings = OllamaEmbeddings(model=model_name)
        embeddings.embed_query("test")
        logging.info(f"Ollama Embeddings '{model_name}' initialized successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Ollama Embeddings '{model_name}' initialization FAIL: {e}", exc_info=True)
        st.error(f"Ollama 임베딩 모델 ('{model_name}') 로드 실패: {e}. Ollama 서버 및 모델 상태를 확인하세요.")
        return None

@st.cache_resource(show_spinner="Google AI 임베딩 모델 로드 중...")
def get_google_embeddings(model_name: str) -> Optional[GoogleGenerativeAIEmbeddings]:
    if not GOOGLE_API_KEY:
        st.error("Google API 키가 설정되지 않았습니다. .env 파일 또는 환경 변수를 확인해주세요.")
        logging.error("Google API key not found when attempting to initialize Google Embeddings.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        embeddings.embed_query("test")
        logging.info(f"Google AI Embeddings '{model_name}' initialized successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Google AI Embeddings '{model_name}' initialization FAIL: {e}", exc_info=True)
        st.error(f"Google AI 임베딩 모델 ('{model_name}') 로드 실패: {e}. API 키와 모델명을 확인하세요.")
        return None

@st.cache_resource(show_spinner="Ollama LLM 로드 중...")
def get_ollama_llm(model_name: str, temperature: float = 0.1) -> Optional[ChatOllama]:
    try:
        llm = ChatOllama(model=model_name, temperature=temperature)
        llm.invoke("안녕")
        logging.info(f"Ollama LLM '{model_name}' initialized successfully.")
        return llm
    except Exception as e:
        logging.error(f"Ollama LLM '{model_name}' initialization FAIL: {e}", exc_info=True)
        st.error(f"Ollama LLM ('{model_name}') 로드 실패: {e}. Ollama 서버 및 모델 상태를 확인하세요.")
        return None

# --- Vector Store Creation/Loading ---
def create_or_load_vectorstore(
    texts_with_sources: List[Tuple[str, str]],
    qa_documents: List[Document],
    embeddings: Embeddings,
    embedding_provider_name: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    force_recreate: bool = False
) -> Optional[Chroma]:
    overall_start_time = time.time()
    final_documents_for_db: List[Document] = []

    doc_processing_start_time = time.time()
    if texts_with_sources:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", "", "제\d+조", "제\d+항"]
        )
        logging.info("Starting document processing for primary vector store sources...")
        primary_doc_chunks_count = 0
        for source_idx, (source_name, raw_text) in enumerate(texts_with_sources):
            if not raw_text.strip(): logging.warning(f"Source file '{source_name}' (idx {source_idx}) is empty. Skipping."); continue
            processed_text = preprocess_text(raw_text)
            if not processed_text.strip(): logging.warning(f"Preprocessing resulted in empty text for '{source_name}'. Skipping."); continue
            article_sections = split_text_by_article(processed_text)
            if not article_sections: logging.warning(f"No articles found in '{source_name}'.")
            for article_title, article_content in article_sections.items():
                if not article_content.strip(): logging.debug(f"Skipping empty article content for title '{article_title}' in '{source_name}'."); continue
                base_metadata = extract_metadata_from_title(article_title)
                base_metadata.update({"source_file": source_name, "article_title_full": article_title, "doc_type": "legal_document_chunk"})
                article_doc = Document(page_content=article_content, metadata=base_metadata)
                split_article_chunks = text_splitter.split_documents([article_doc])
                for i, chunk_doc in enumerate(split_article_chunks):
                    chunk_doc.metadata["chunk_index_in_article"] = i
                    final_documents_for_db.append(chunk_doc)
                    primary_doc_chunks_count += 1
        logging.info(f"Total legal document chunks created: {primary_doc_chunks_count}")
    doc_processing_end_time = time.time()
    if texts_with_sources:
        logging.info(f"Primary document processing took: {doc_processing_end_time - doc_processing_start_time:.2f}s")

    if qa_documents:
        final_documents_for_db.extend(qa_documents)
        logging.info(f"Added {len(qa_documents)} Q&A documents to the list for vector store.")
    logging.info(f"Total documents prepared for vector store: {len(final_documents_for_db)}")

    if not final_documents_for_db:
        st.error("문서 또는 Q&A 데이터 처리 후 생성된 내용이 없어 DB를 만들거나 로드할 수 없습니다.")
        logging.error("No documents (neither primary nor QA) to process for DB.")
        return None

    safe_model_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', embedding_model_name).replace('/', '_').replace(':', '_')
    persist_directory_name = f"chroma_db_{embedding_provider_name.lower()}_{safe_model_name}_cz{chunk_size}_co{chunk_overlap}"
    persist_directory = Path(persist_directory_name)
    logging.info(f"Target ChromaDB persistence directory: {persist_directory.resolve()}")
    vector_db: Optional[Chroma] = None

    if force_recreate and persist_directory.exists():
        logging.info(f"Force Recreate: Deleting existing ChromaDB at {persist_directory}")
        import shutil
        try: shutil.rmtree(persist_directory); logging.info(f"Deleted DB: {persist_directory}")
        except Exception as e: logging.error(f"Error deleting DB: {e}", exc_info=True); st.error(f"DB 삭제 오류: {e}"); return None

    if not force_recreate and persist_directory.exists() and any(persist_directory.iterdir()):
        load_db_start_time = time.time()
        try:
            logging.info(f"Attempting to load existing ChromaDB from {persist_directory}...")
            vector_db = Chroma(persist_directory=str(persist_directory), embedding_function=embeddings)
            if vector_db._collection.count() == 0: logging.warning(f"Existing DB at {persist_directory} is empty."); vector_db = None
            else: logging.info(f"Existing ChromaDB loaded ({vector_db._collection.count()} items). Took {time.time() - load_db_start_time:.2f}s.")
        except Exception as e:
            logging.error(f"Error loading ChromaDB: {e}", exc_info=True); st.warning(f"DB 로딩 오류: {e}. 새 DB 생성을 시도합니다."); vector_db = None

    if vector_db is None:
        if not final_documents_for_db: logging.warning("No documents for new DB."); return None
        create_db_start_time = time.time()
        try:
            logging.info(f"Creating new ChromaDB ({len(final_documents_for_db)} chunks)...")
            persist_directory.mkdir(parents=True, exist_ok=True)
            vector_db = Chroma.from_documents(documents=final_documents_for_db, embedding=embeddings, persist_directory=str(persist_directory))
            logging.info(f"New ChromaDB created ({vector_db._collection.count()} items). Took {time.time() - create_db_start_time:.2f}s.")
        except Exception as e:
            logging.error(f"Fatal error creating new ChromaDB: {e}", exc_info=True); st.error(f"새 DB 생성 실패: {e}")
            if persist_directory.exists(): import shutil; shutil.rmtree(persist_directory)
            return None
    logging.info(f"Total create_or_load_vectorstore took: {time.time() - overall_start_time:.2f}s.")
    return vector_db

# --- RAG & LLM Interaction Functions ---
def format_docs_for_context(docs: List[Document]) -> str:
    if not docs: return "제공된 문맥 정보가 없습니다."
    formatted_docs_list = []
    for i, doc in enumerate(docs):
        doc_type = doc.metadata.get('doc_type', 'unknown')
        header, content_preview = "", ""
        if doc_type == "qa_pair":
            original_q = doc.metadata.get('original_question', '질문 정보 없음')
            excel_filename = doc.metadata.get('source_file', 'Excel')
            header = f"참고 Q&A {i+1} (출처: {excel_filename}, 원본 질문: \"{original_q[:50].strip()}...\")"
            content_preview = doc.page_content
        elif doc_type == "legal_document_chunk":
            source_file = doc.metadata.get('source_file', '출처 파일 불명')
            article_title = doc.metadata.get('article_title_full', '전체 문서의 일부')
            header = f"법률 문서 {i+1}: '{source_file}'"
            if article_title not in ["전체 문서의 일부", "전체 문서", "내용 없음", "N/A", "조항 정보 없음"]:
                header += f" (관련 조항 추정: {article_title})"
            content_preview = doc.page_content.strip()
        else:
            source_file = doc.metadata.get('source_file', '출처 불명')
            header = f"기타 문서 {i+1} (출처: {source_file})"
            content_preview = doc.page_content.strip()
        max_content_length = 700 # Increased slightly
        if len(content_preview) > max_content_length: content_preview = content_preview[:max_content_length] + "..."
        formatted_docs_list.append(f"{header}\n추출 내용:\n{content_preview}")
    return "\n\n---\n\n".join(formatted_docs_list)

def generate_answer_with_rag(question: str, vectorstore: Chroma, llm: ChatOllama) -> Dict[str, Any]:
    result_data = {"query": question, "result": "RAG 답변 생성 중 오류 발생", "source_documents": []}
    try:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6})
        prompt = PromptTemplate(template=DEFAULT_RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
        rag_chain = (
            {"context": retriever | format_docs_for_context, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        logging.info(f"Invoking RAG chain for question: '{question[:70]}...'")
        answer = rag_chain.invoke(question)
        retrieved_docs_for_display = retriever.invoke(question) # For display
        if not retrieved_docs_for_display: logging.warning("RAG retriever found no relevant documents (for display).")
        result_data.update({"result": answer, "source_documents": retrieved_docs_for_display})
        return result_data
    except Exception as e:
        logging.error(f"Error during RAG generation: {e}", exc_info=True)
        st.error(f"RAG 답변 생성 중 오류: {e}")
        result_data["result"] = "RAG 답변 생성 과정에서 시스템 오류가 발생했습니다. 로그를 확인해주세요."
        return result_data

def generate_direct_llm_answer(question: str, llm: ChatOllama) -> Dict[str, Any]:
    result_data = {"query": question, "result": "LLM 직접 답변 생성 중 오류 발생", "source_documents": None}
    try:
        direct_prompt_text = f"""당신은 법률 분야 AI 어시스턴트입니다. 다음 질문에 대해 당신의 일반 지식을 바탕으로 한국어로 답변해주세요. (외부 문서/실시간 정보 검색 기반 아님)\n\n질문: {question}\n\n답변:"""
        logging.info(f"Invoking direct LLM for question: '{question[:70]}...'")
        response = llm.invoke(direct_prompt_text)
        answer = response.content if isinstance(response, BaseMessage) else str(response)
        result_data["result"] = answer
        return result_data
    except Exception as e:
        logging.error(f"Error during direct LLM call: {e}", exc_info=True)
        st.error(f"LLM 직접 답변 생성 중 오류: {e}")
        result_data["result"] = "LLM 직접 답변 과정에서 시스템 오류 발생."
        return result_data

def generate_blended_answer(question: str, vectorstore: Chroma, llm: ChatOllama) -> Dict[str, Any]:
    logging.info(f"Blended Answer Step 1: RAG for: '{question[:70]}...'")
    rag_result_data = generate_answer_with_rag(question, vectorstore, llm)
    initial_rag_answer, source_documents = rag_result_data.get("result"), rag_result_data.get("source_documents")
    rag_had_error = "오류 발생" in (initial_rag_answer or "") or "시스템 오류" in (initial_rag_answer or "")
    if rag_had_error or not initial_rag_answer:
        logging.warning(f"RAG step failed/produced error: '{initial_rag_answer}'."); return rag_result_data
    no_answer_phrases = ["제공된 문맥 정보만으로는 질문에 답변할 수 없습니다.", "문맥에서 답변을 찾을 수 없습니다."]
    if any(phrase in initial_rag_answer for phrase in no_answer_phrases):
        logging.info("RAG indicated no answer in context. Skipping refinement."); return rag_result_data
    logging.info("Blended Answer Step 2: LLM refinement.")
    retrieved_context_str = format_docs_for_context(source_documents if source_documents else [])
    refinement_prompt = PromptTemplate(template=REFINEMENT_PROMPT_TEMPLATE, input_variables=["original_question", "retrieved_context", "rag_answer"])
    refinement_chain = refinement_prompt | llm | StrOutputParser()
    try:
        final_answer = refinement_chain.invoke({"original_question": question, "retrieved_context": retrieved_context_str, "rag_answer": initial_rag_answer})
        return {"query": question, "result": final_answer, "source_documents": source_documents}
    except Exception as e:
        logging.error(f"Error during Blended Refinement: {e}", exc_info=True); st.error(f"답변 개선 중 오류: {e}")
        st.warning("답변 개선 실패. 초기 RAG 답변 표시."); return rag_result_data

# --- Display Function ---
def display_results(result_data: Optional[Dict[str, Any]], answer_mode: str):
    if result_data is None: st.error("결과 데이터 없음 (None)."); logging.error("display_results called with None"); return
    if not isinstance(result_data, dict): st.error(f"결과 데이터 형식 오류 (type: {type(result_data)})."); logging.error(f"Invalid result_data: {result_data}"); return
    st.markdown(f"##### 🤖 AI 답변 ({answer_mode})")
    answer = result_data.get("result", "답변 생성 실패/결과 없음.")
    if not answer or answer.strip() == "": st.warning("AI 생성 답변 내용 비어있음."); answer = "AI가 답변을 생성하지 못했습니다."
    st.markdown(answer); st.markdown("---")
    source_documents = result_data.get("source_documents")
    if source_documents and isinstance(source_documents, list):
        st.markdown("##### 📚 참고 문헌 (검색된 문서)")
        for i, doc in enumerate(source_documents):
            if not isinstance(doc, Document) or not hasattr(doc, 'metadata'): logging.warning(f"Source doc {i} invalid: {doc}"); st.warning(f"출처 {i+1} 형식 오류."); continue
            doc_type = doc.metadata.get('doc_type', 'unknown')
            expander_title = ""
            if doc_type == "qa_pair":
                original_q = doc.metadata.get('original_question', "질문 없음")
                excel_filename = doc.metadata.get('source_file', 'Excel')
                expander_title = f"참고 Q&A {i+1}: \"{original_q[:40].strip()}...\" (출처: {excel_filename})"
            else: # legal_document_chunk or unknown
                source_file = doc.metadata.get('source_file', '출처 불명')
                article_title_full = doc.metadata.get('article_title_full', '조항 정보 없음')
                expander_title = f"법률 문서 {i+1}: {source_file}"
                if article_title_full not in ['조항 정보 없음', 'N/A', '내용 없음', '전체 문서', '전체 문서의 일부']:
                    expander_title += f" - (관련 조항 추정: {article_title_full})"
            with st.expander(expander_title):
                st.markdown(f"**추출된 내용:**"); st.text(doc.page_content if doc.page_content else "내용 없음")
                display_meta = {k:v for k,v in doc.metadata.items() if k not in ['source_file', 'article_title_full', 'start_index', 'doc_type', 'original_question']}
                if display_meta: st.markdown(f"**메타데이터:**"); st.json(display_meta)
    elif answer_mode in ["RAG (문서 기반)", "Blended (RAG + LLM 개선)"]:
        st.info("이 답변 모드에서는 근거 조항이 제공될 수 있으나, 현재 검색된 관련 문서가 없습니다.")

# --- Main Streamlit Application ---
def main():
    st.set_page_config(page_title="AI 법률 질의응답 시스템", layout="wide", initial_sidebar_state="expanded")
    st.title("⚖️ AI 법률 질의응답 시스템")

    session_defaults = {
        "llm": None, "llm_model_name": DEFAULT_OLLAMA_LLM_MODEL,
        "embeddings": None, "embedding_provider_name_loaded": "Ollama", "embedding_model_name_loaded": DEFAULT_OLLAMA_EMBEDDING_MODEL,
        "vectorstore": None, "vectorstore_config_key_loaded": None,
        "db_status_message": "DB가 로드되지 않았습니다. 설정을 확인하고 'DB 준비' 버튼을 클릭하세요.", "db_ready": False,
        "texts_with_sources_cache": None, "last_loaded_doc_dir": DEFAULT_DOC_DIRECTORY, "last_loaded_file_type": "txt",
        "uploaded_qa_file_name": None, "qa_documents_cache": None,
        "sidebar_doc_dir": DEFAULT_DOC_DIRECTORY, "sidebar_file_type": "txt", "sidebar_emb_provider": "Ollama",
        "sidebar_ollama_emb_model": DEFAULT_OLLAMA_EMBEDDING_MODEL, "sidebar_google_emb_model": DEFAULT_GOOGLE_EMBEDDING_MODEL,
        "sidebar_ollama_llm_model": DEFAULT_OLLAMA_LLM_MODEL, "sidebar_chunk_size": CHUNK_SIZE_DEFAULT,
        "sidebar_chunk_overlap": CHUNK_OVERLAP_DEFAULT, "sidebar_force_recreate_db": False,
    }
    for key, value in session_defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    
    st.caption(f"LLM ({st.session_state.llm_model_name if st.session_state.llm else '로드 안됨'}), "
               f"임베딩 ({st.session_state.embedding_provider_name_loaded}: "
               f"{st.session_state.embedding_model_name_loaded if st.session_state.embeddings else '로드 안됨'})")

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
        st.session_state.sidebar_ollama_llm_model = st.text_input("Ollama LLM 모델명", value=st.session_state.sidebar_ollama_llm_model, key="ollama_llm_sb", help=f"예: {DEFAULT_OLLAMA_LLM_MODEL}")

        st.subheader("4. 벡터 DB 설정")
        st.session_state.sidebar_chunk_size = st.slider("Chunk Size", 200, 4000, st.session_state.sidebar_chunk_size, 50, key="chunk_size_sb")
        st.session_state.sidebar_chunk_overlap = st.slider("Chunk Overlap", 0, 1000, st.session_state.sidebar_chunk_overlap, 50, key="chunk_overlap_sb")
        
        st.subheader("5. 이전 Q&A 데이터 (선택)")
        uploaded_qa_file_ui = st.file_uploader("Q&A 참고자료 Excel (.xlsx, .xls)", type=["xlsx", "xls"], key="qa_file_uploader_sb")
        
        st.session_state.sidebar_force_recreate_db = st.checkbox("DB 강제 재생성", value=st.session_state.sidebar_force_recreate_db, key="force_recreate_sb")


    if st.session_state.llm is None or st.session_state.llm_model_name != st.session_state.sidebar_ollama_llm_model:
        with st.spinner(f"Ollama LLM ({st.session_state.sidebar_ollama_llm_model}) 로드 중..."):
            llm_instance = get_ollama_llm(st.session_state.sidebar_ollama_llm_model)
            if llm_instance: st.session_state.llm, st.session_state.llm_model_name = llm_instance, st.session_state.sidebar_ollama_llm_model
            else:
                if st.session_state.llm: st.sidebar.error(f"새 LLM 로드 실패. 이전 LLM({st.session_state.llm_model_name}) 사용.")
                else: st.sidebar.error(f"LLM({st.session_state.sidebar_ollama_llm_model}) 로드 실패.")
    if st.session_state.llm: st.sidebar.success(f"LLM ({st.session_state.llm_model_name}) 준비 완료")
    else: st.sidebar.error(f"LLM ({st.session_state.sidebar_ollama_llm_model}) 로드 실패 또는 안됨.")

    col1, col2 = st.columns([1.2, 2]) # Adjusted column ratio
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
            st.session_state.db_ready = False; st.session_state.db_status_message = "DB 처리 시작..."
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
                st.session_state.db_status_message = "❌ 임베딩 모델 초기화 실패."; st.session_state.embeddings = None; st.rerun()

            st.session_state.embeddings = embeddings_instance
            st.session_state.embedding_provider_name_loaded = st.session_state.sidebar_emb_provider
            st.session_state.embedding_model_name_loaded = current_embedding_model_to_use
            logging.info(f"Embeddings set: {st.session_state.embedding_provider_name_loaded} - {st.session_state.embedding_model_name_loaded}")

            # Load Q&A documents
            current_qa_docs = []
            if uploaded_qa_file_ui is not None:
                if (st.session_state.uploaded_qa_file_name != uploaded_qa_file_ui.name) or \
                   st.session_state.qa_documents_cache is None or st.session_state.sidebar_force_recreate_db:
                    with st.spinner(f"'{uploaded_qa_file_ui.name}' Q&A 데이터 로드..."):
                        loaded_qa_docs = load_qa_from_excel(uploaded_qa_file_ui)
                    if loaded_qa_docs: st.session_state.qa_documents_cache, st.session_state.uploaded_qa_file_name = loaded_qa_docs, uploaded_qa_file_ui.name; st.success(f"'{uploaded_qa_file_ui.name}'에서 {len(loaded_qa_docs)}개 Q&A 로드 완료.")
                    else: st.session_state.qa_documents_cache, st.session_state.uploaded_qa_file_name = [], uploaded_qa_file_ui.name # Error handled in load_qa_from_excel
                current_qa_docs = st.session_state.qa_documents_cache
                if current_qa_docs and not ((st.session_state.uploaded_qa_file_name != uploaded_qa_file_ui.name) or st.session_state.qa_documents_cache is None or st.session_state.sidebar_force_recreate_db):
                     if uploaded_qa_file_ui.name == st.session_state.uploaded_qa_file_name : st.info(f"캐시된 Q&A {len(current_qa_docs)}개 사용: '{st.session_state.uploaded_qa_file_name}'")

            else: # No QA file uploaded this time
                if st.session_state.uploaded_qa_file_name is not None: st.info("Q&A Excel 파일 제거됨.") # If there was one before
                st.session_state.qa_documents_cache, st.session_state.uploaded_qa_file_name = [], None
            current_qa_docs = st.session_state.qa_documents_cache if st.session_state.qa_documents_cache else []


            # Load primary documents
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
                    st.session_state.texts_with_sources_cache, st.session_state.last_loaded_doc_dir, st.session_state.last_loaded_file_type = texts_sources_for_db, st.session_state.sidebar_doc_dir, st.session_state.sidebar_file_type
                else: st.warning(f"'{st.session_state.sidebar_doc_dir}' ({st.session_state.sidebar_file_type})에서 문서 로드 실패."); st.session_state.texts_with_sources_cache = None; st.rerun()
            else: texts_sources_for_db = st.session_state.texts_with_sources_cache; st.info(f"캐시된 법률 문서 {len(texts_sources_for_db)}개 사용.")

            # Create/Load Vectorstore
            if (texts_sources_for_db or current_qa_docs) or not st.session_state.sidebar_force_recreate_db:
                with st.spinner("벡터 DB 구성 중... (시간 소요될 수 있음)"):
                    vector_db_instance = create_or_load_vectorstore(texts_sources_for_db, current_qa_docs, st.session_state.embeddings,
                                                                    st.session_state.embedding_provider_name_loaded, st.session_state.embedding_model_name_loaded,
                                                                    st.session_state.sidebar_chunk_size, st.session_state.sidebar_chunk_overlap,
                                                                    st.session_state.sidebar_force_recreate_db)
                if vector_db_instance:
                    st.session_state.vectorstore, st.session_state.vectorstore_config_key_loaded, st.session_state.db_ready = vector_db_instance, current_sidebar_db_config_key, True
                    st.session_state.db_status_message = f"✅ DB 준비 완료: {st.session_state.embedding_model_name_loaded} (Chunk: {st.session_state.sidebar_chunk_size}/{st.session_state.sidebar_chunk_overlap})"
                    if st.session_state.sidebar_force_recreate_db: st.session_state.sidebar_force_recreate_db = False
                else: st.session_state.vectorstore, st.session_state.db_ready = None, False; st.session_state.db_status_message = "❌ 벡터 DB 준비 실패. 로그 확인."
            else: st.session_state.db_status_message = "❌ DB 강제 재생성: 추가할 법률/Q&A 데이터 없음."
            st.rerun()

    with col2:
        st.subheader("💬 질의응답")
        answer_mode = st.radio("답변 방식:", ["RAG (문서 기반)", "Blended (RAG + LLM 개선)", "LLM 직접 답변"], index=0, key="answer_mode_main", horizontal=True,
                               help=("**RAG**: 문서 기반 답변.\n**Blended**: RAG 답변을 LLM이 개선.\n**LLM 직접 답변**: LLM 자체 지식으로 답변."))
        llm_ready = bool(st.session_state.llm)
        db_compatible = st.session_state.db_ready and st.session_state.vectorstore is not None and (st.session_state.vectorstore_config_key_loaded == current_sidebar_db_config_key)
        can_ask = False
        if not llm_ready: st.warning("LLM 미준비. 사이드바 설정 확인.")
        if answer_mode == "LLM 직접 답변":
            if llm_ready: can_ask = True
        else: # RAG or Blended
            if not llm_ready: pass
            elif not st.session_state.db_ready: st.warning("DB 미로드. 왼쪽 'DB 준비' 클릭.")
            elif not db_compatible: st.warning("로드된 DB가 현재 설정과 다름. DB 재준비 필요.")
            else: can_ask = True

        question = st.text_area("질문 입력:", height=100, key="user_question_main", placeholder="법률 관련 질문 입력...", disabled=not can_ask)
        if st.button("답변 생성 🚀", key="ask_btn_main", disabled=not can_ask or not question.strip(), use_container_width=True):
            with st.spinner("AI 답변 생성 중... 🤔"):
                final_result_data = None
                if answer_mode != "LLM 직접 답변" and not st.session_state.vectorstore: st.error("벡터스토어 미준비.")
                elif not st.session_state.llm: st.error("LLM 미준비.")
                else:
                    if answer_mode == "RAG (문서 기반)": final_result_data = generate_answer_with_rag(question, st.session_state.vectorstore, st.session_state.llm)
                    elif answer_mode == "Blended (RAG + LLM 개선)": final_result_data = generate_blended_answer(question, st.session_state.vectorstore, st.session_state.llm)
                    elif answer_mode == "LLM 직접 답변": final_result_data = generate_direct_llm_answer(question, st.session_state.llm)
                if final_result_data: display_results(final_result_data, answer_mode)
        elif not question.strip() and can_ask: st.info("답변 생성하려면 질문 입력.")

if __name__ == "__main__":
    Path(DEFAULT_DOC_DIRECTORY).mkdir(parents=True, exist_ok=True)
    logging.info(f"Default document directory ensured: '{DEFAULT_DOC_DIRECTORY}'")
    main()