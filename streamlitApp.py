## BACKEND IMPORTS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from llama_index.core import PromptTemplate
from llama_index.core.settings import Settings
from llama_index.llms.llama_cpp import LlamaCPP # type: ignore
## FRONTEND IMPORTS
import streamlit as st
import re
pattern = r"<\|.*\|>"

st.set_page_config(page_title="ADP RAG Engine bêta", page_icon="./favicon.png")
st.title("PoC RAG ADP")

def save_feedback():
    pass


class RagEngine:
    def __init__(self, dir="./data", embedding_model_name = 'dangvantuan/sentence-camembert-base', model_path="./models/Meta-Llama-3.1-8B-Instruct.gguf"):
        self.dir = dir
        self.embedding_model_name = embedding_model_name
        # DISPLAY CURRENT DATABASE TO USER
        current_datastore = ""
        for file_name in os.listdir(self.dir):
            current_datastore += f"\t{file_name}\n"
        st.info(f"L'assistant a connaissance des documents suivants : {current_datastore}", icon="ℹ️")
        print("## LOADING EMBEDDINGS MODEL")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs = {"normalize_embeddings": False}
        )
        print("## LOADING LLM MODEL")
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            #max_new_tokens=256,
            generate_kwargs={},
            model_kwargs={
                "low_cpu_mem_usage": True,
            },
            messages_to_prompt=self.messages_to_prompt,
            completion_to_prompt=self.completion_to_prompt,
            verbose=False,
        )
        print("DEFINING SETTINGS")
        self.chunk_size = 500
        Settings.llm = self.llm
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = self.chunk_size
        print("SETTING QA TEMPLATE")
        self.text_qa_template_str = (
            "<|system|>: Vous êtes un assistant IA qui répond à la question posée à la fin en utilisant le contexte suivant. "
            "Construisez une réponse élégante pour répondre à la question en reformulant le contexte. Si vous ne connaissez pas "
            "la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse. Veuillez répondre "
            "exclusivement en français.\n"
            "<|user|>: {context_str}\n"
            "Question: {query_str}\n"
            "<|assistant|>:"
        )
        self.text_qa_template = PromptTemplate(self.text_qa_template_str)

    def ingestDocuments(self, userDocs=[]):
        print("LOADING DOCUMENTS TO VECTOR STORE")
        self.documents = SimpleDirectoryReader(
            input_files=[
                f"{self.dir}/{file_name}" for file_name in os.listdir(self.dir)
            ] + 
            [
                f"/tmp/{file_name}" for file_name in userDocs
            ]
        ).load_data()
        self.vectorstore_index = VectorStoreIndex.from_documents(
            documents = self.documents,
            show_progress=True
        )
        self.vectorstore_index.storage_context.persist(persist_dir='llama_index')
        self.query_engine = self.vectorstore_index.as_query_engine(
            text_qa_template=self.text_qa_template,
            similarity_top_k=2,
            streaming=True
        )
        print("## ENGINE READY")
    
    def query(self, userQuestion):
        response = self.query_engine.query(userQuestion)
        buffer = ""
        for text in response.response_gen: # type: ignore
            buffer += text
            if re.fullmatch(pattern, buffer):
                print(f"Token indésirable détecté: {buffer}")  # Affiche le token indésirable
                return
            yield text


    def messages_to_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message.role == 'system':
                prompt += f"<|system|>\n{message.content}\n"
            elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}\n"
            elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}\n"
        # Add end-of-text token at the end of the prompt
        prompt += "<|eot_id|>\n"
        return prompt
    
    def completion_to_prompt(self, completion):
        return f"<|system|>\n\n<|user|>\n{completion}\n</s>\n<|assistant|>\n"


@st.cache_resource
def start_engine():
    engine = RagEngine()
    engine.ingestDocuments()
    return engine

engine = start_engine()


## FRONTEND
if "engine" not in st.session_state:
    st.session_state["engine"] = engine
    #st.session_state["engine"].ingestDocuments()
    st.session_state.get_feedback = True
if "messages" not in st.session_state:
    st.session_state.messages = []


if "uploadedDocs" not in st.session_state:
    st.session_state["uploadedDocs"] = []
    print("USER UPLOADS EMPTY")

# DISPLAY HISTORY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# UPLOAD SIDEBAR
st.sidebar.header("Ajouter des documents") # Add in temp dir and reload ingestor
selected_files = st.sidebar.file_uploader('Uploadez votre document', type="pdf", accept_multiple_files=True)
if selected_files:
    new_files = []
    for file in selected_files:
        st.session_state["uploadedDocs"].append(file.file_id)
        # CREATE FILE

        with open(f"/tmp/{file.file_id}", 'wb') as f: 
            f.write(file.getvalue())
    print("USER UPLOADED DOCS\nRELOADING SESSION ENGINE INGESTOR")
    # RE-INGEST
    st.session_state["engine"].ingestDocuments(userDocs=st.session_state["uploadedDocs"])
    print("USER ENGINE RE-INGESTED")

# USER PROMPT AND ASSISTANT RESPONSES
if prompt := st.chat_input("Quelle est votre question ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state["engine"].query(prompt))
        st.feedback("thumbs", key="feedback", on_change=save_feedback)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})