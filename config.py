# config.py
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Constants ---
DEFAULT_DOC_DIRECTORY = "./laws_data_streamlit_app"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "bge-m3" # or "nomic-embed-text" "mxbai-embed-large"
DEFAULT_GOOGLE_EMBEDDING_MODEL = "models/embedding-001"
DEFAULT_OLLAMA_LLM_MODEL = "phi4-mini" #  Adjust if needed, e.g., "llama3", "qwen2"
DEFAULT_GEMINI_LLM_MODEL = "gemini-2.0-flash-lite-001" # Adjust if needed, e.g., "gemini-2.0-flash-001"
CHUNK_SIZE_DEFAULT = 1500
CHUNK_OVERLAP_DEFAULT = 300

# --- Prompt Templates ---
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

FULL_DOC_FALLBACK_PROMPT_TEMPLATE = """당신은 법률 AI 어시스턴트입니다. 제공된 전체 법률 문서 내용을 바탕으로 사용자의 질문에 답변해주세요.
문서의 내용이 방대할 수 있으니, 질문과 가장 관련 있는 부분을 찾아 명확하고 간결하게 답변해야 합니다.
만약 제공된 전체 문서 내용에서 답변을 찾을 수 없다면, "제공된 전체 문서 내용만으로는 질문에 답변할 수 없습니다."라고 명확히 답변해주세요.

제공된 전체 문서 (출처: {source_description}):
{context}

질문: {question}

답변 (한국어, 전체 문서 내용 근거):"""