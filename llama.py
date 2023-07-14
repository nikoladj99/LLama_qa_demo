#importovanje
from llama_index import VectorStoreIndex, SimpleDirectoryReader,ServiceContext, LLMPredictor
from llama_index import StorageContext, load_index_from_storage
from langchain import HuggingFaceHub
from llama_index import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate
from llama_index import set_global_service_context
import streamlit as st
import pypdf
from streamlit_chat import message

def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

st.title("Sova demo app")
uploaded_files = st.file_uploader("Choose file(s)", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        filename = './docs_database/' + uploaded_file.name
        if '.pdf' in filename:
            pdf_converted_to_text = read_pdf(uploaded_file)
            pdf_converted_to_text = pdf_converted_to_text.encode("ascii", "replace")
            pdf_converted_to_text = pdf_converted_to_text.decode(encoding="utf-8", errors="ignore")
            pdf_converted_to_text = pdf_converted_to_text.replace("?", " ")
            filename = filename.replace('.pdf', '.txt')
            with open(filename, 'wt', encoding="utf-8") as f:
                f.write(pdf_converted_to_text)
        else:
            file = uploaded_file.getvalue()
            with open(filename, 'wb') as f:
                f.write(file)
        st.write("File "+uploaded_file.name+" loaded")

#temp slider 
temperature_value = st.slider('Please select the model temperature:', 0.0, 1.0, 0.5)
st.write('Current temperature:', temperature_value)

#odabir modela 
access_token = "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"
repo_id = "google/flan-t5-small"
llm_predictor = LLMPredictor(llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs= {"temperature": temperature_value, "max_length": 64},
    huggingfacehub_api_token= access_token
))
embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
service_context = ServiceContext.from_defaults(llm_predictor= llm_predictor, embed_model=embed_model)
set_global_service_context(service_context)
#za nalazenje source-a odakle je response 
file_metadata = lambda x : {"filename": x}
def find_source(response):
    max_score = 0
    source = response.source_nodes[0]
    for node in response.source_nodes:
        if node.score > max_score:
            max_score = node.score
            source = node
        if source.score> 0.3:
            return source.node.metadata.get('filename')
        else:
            return "General Knowledge. Cannot verify source."
        
#kreiranje indeksa pri prvom pozivu i updatovanje kasnije
def load_index(dir_path):
    documents = SimpleDirectoryReader(dir_path, filename_as_id=True, file_metadata=file_metadata).load_data()
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        print("Index loaded from storage")
    except FileNotFoundError:
        #pravimo indeks od nule samo pri prvom pozivu
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.set_index_id("sova_vector_index")
        index.storage_context.persist()
        print("New index created")

    refreshed_docs = index.refresh_ref_docs(documents, update_kwargs={"delete_kwargs": {'delete_from_docstore': True}})
    print('Number of newly inserted/refreshed docs: ', sum(refreshed_docs))
    index.storage_context.persist()
    return index 

index = load_index('./docs_database')
query_engine = index.as_query_engine()

# nizovi poruka
if 'generated' not in st.session_state:
    st.session_state['generated'] = [ ]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = [ ]

chat_prompt = st.chat_input("Ask a question")
#pravljenje template-a za odgovor
if chat_prompt is not None:
    response = query_engine.query(str(chat_prompt))
    file_source = find_source(response)
    template = """{answer}
    \nFrom source: {file_source}
    """
    model_prompt = PromptTemplate(input_variables = ['answer', 'file_source'], template=template)
   
    st.session_state.past.append(chat_prompt)
    st.session_state.generated.append(model_prompt.format(answer = response, file_source = file_source))
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))




    