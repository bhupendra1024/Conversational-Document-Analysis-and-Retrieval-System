import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter     #divide texts into chunks 
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS# database that stores all of the numerical representaion of the text chunks
                          # and it runs locally
from langchain.llms import openai
import os

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from htmltemplate import css, bot_template, user_template


os.environ['OPENAI_API_KEY'] = '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'

# Your code here

def get_pdf_text(pdf_docs):
  text = ""  # variable that contains all of the raw texts concatinated 
  for pdf in pdf_docs:   # looping each pdf 
    pdf_reader = PdfReader(pdf)  #initialised pdf object for each pdf 
    for page in pdf_reader.pages:   #loop thru all of the pages of each pdf
      text += page.extract_text()   #extract the texts from the page and concatinate it
  return text                        # in text variable 



# splitting the texts into chunks of 1000 characters 
# Chunk_overlap - to make sure in case the charaters cuts off in the middle of a word then to overlap with few
# cahravters ahead for the texts to make sense later when its embedded and fed into LLM model  
def get_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=1000,  #1000 characters
    chunk_overlap=200,
    length_function=len
  )

  chunks = text_splitter.split_text(raw_text)
  return chunks 


def get_vectorstore(text_chunks):

  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_userinput():
  response = st.session_state.conversation({'question', user_question})
  st.session_state.chat_history = response['chat_history']

  for i, message in enumerate(st.session_state.chat_history):
    if i & 2 == 0:
      st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
    else:
      st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)



def main():

  load_dotenv()
  st.set_page_config(page_title="Chat with multiple doc/pdf files", page_icon=":books:")

  st.write(css, unsafe_allow_html=True)

  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.header("Chat with multiple doc/pdf files")
  user_question = st.text_input("Ask questions about your document")
  if user_question:
    handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your Pdfs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
      with st.spinner("Processing"):
        # get the pdf text
        raw_text = get_pdf_text(pdf_docs)
        
        # get the text chunks 
        text_chunks = get_text_chunks(raw_text)
        # st.write(text_chunks)

         # create vector store with the embedding 
        vectorstore = get_vectorstore(text_chunks)

        # conversation Chain - have memory allowing us to ask follow-up questions 
        st.session_state.conversation = get_conversation_chain(vectorstore) 
        # Streamlit reloads everytime some event takes place and reinitialises the variables
        # Used session_state for keeping the variables as it is after the reloads 

  st.write(user_template.replace("{{MSG}}","Hello Bot"), unsafe_allow_html=True)
  st.write(bot_template.replace("{{MSG}}","Hello HUman"), unsafe_allow_html=True)


if __name__ == '__main__':
  main()