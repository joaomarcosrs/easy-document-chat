import os
import tomllib
import importlib
import subprocess
from time import sleep
from pathlib import Path
from langchain_community import embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import settings


class EmbeddingModel:
    def __init__(self, embedding=None):
        self._embedding = embedding
    
    def list_embeddings(self):
        return embeddings.__all__
    
    def __embed_configuration(self, embed=None, embeddings=None, **kwargs):
        module_name = embeddings._module_lookup[embed]
        module = importlib.import_module(module_name)
        module_class = getattr(module, embed)
        embed_model = module_class(**kwargs)
        
        return embed_model
    
    def embedding(self, **kwargs):
        if not kwargs:
            raise Exception('Embed cannot be empty.')
        
        for embedding_ in embeddings.__all__:
            if str(self._embedding).lower() in embedding_.lower():
                embed = self.__embed_configuration(embed=embedding_, embeddings=embeddings, **kwargs)
                
                return embed

class pdfChat:
    def __init__(self, prompt = None, path = None, llm = None, model = None, embedding = None):
        self._prompt = prompt
        self._path = path
        self._llm = llm
        self._model = model
        self._embedding = embedding
        
    def __db(self):
        vector_db = Chroma(
            persist_directory=settings.CHROMA_PATH, embedding_function=self._embedding
        )
        
        return vector_db
    
    def __pdf_loader(self):
        documents = PyPDFDirectoryLoader(self._path).load()
        
        return documents
    
    def __text_splitter(self, documents, **kwargs):
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
        chunks = text_splitter.split_documents(documents)
        
        return chunks
    
    def __calc_chunk_ids(self, chunks):
        last_page_id = None
        chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get('source')
            page = chunk.metadata.get('page')
            page_id = str(source) + ':' + str(page)

            if page_id != last_page_id:
                chunk_index = 0
            else:
                chunk_index += 1

            chunk_id = str(page_id) + ':' + str(chunk_index)
            last_page_id = page_id
            chunk.metadata["id"] = chunk_id

        return chunks
    
    def run(self):
        try:
            data = tomllib.loads(Path(os.path.dirname(os.path.abspath(__file__))+'/metadata.toml').read_text(encoding='utf 8'))['llm']
        
        except Exception as error:
            print(f'Error reading metadata. \n Error: {error}')
        
        for llm in data:
            if self._llm in llm:
                try:
                    subprocess.run(data[llm]['stop'], shell=True, check=True)
                except subprocess.CalledProcessError as err:
                    try:
                        subprocess.run(data[llm]['install'], shell=True, check=True)
                    except subprocess.CalledProcessError as err:
                        print(f'Ollama install error: {err}')
                
                try:
                    subprocess.run(data[llm]['version'], shell=True, check=True)
                except subprocess.CalledProcessError as err:
                    print(f'Ollama does not return version: {err}')
                
                try:
                    print('Starting Ollama server...')
                    subprocess.Popen(f"{data[llm]['start']}", shell=True)
                    sleep(7)
                    print('Ollama server started.')
                except subprocess.CalledProcessError as err:
                    print(f'Ollama server start error: {err}')
                
                try:
                    print(f'Downloading {self._model}...')
                    subprocess.run(f"{data[llm]['pull']} {self._model}", shell=True)
                    print(f'{self._model} downloaded.')
                    
                    if self._embedding:
                        subprocess.run(f"{data[llm]['pull']} {self._embedding.model}", shell=True)
                except subprocess.CalledProcessError as err:
                    print(f"Ollama model pull error: {err}")

    def response(self, chunk_size = None, chunk_overlap  = None, length_function = None, is_separator_regex = False, **kwargs):
        try:
            kwargs['k']
        except KeyError:
            kwargs['k'] = 4
        
        if not self._prompt:
            raise Exception('Prompt cannot be None.')
        
        if not self._path:
            raise Exception('Inform the files path.')
            
        vector_db = self.__db()
        
        documents = self.__pdf_loader()
        
        chunks = self.__text_splitter(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=length_function, is_separator_regex=is_separator_regex)
        chunks_with_ids = self.__calc_chunk_ids(chunks)
        
        _items = vector_db.get(include=[])
        existing_ids = set(_items['ids'])

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            vector_db.add_documents(new_chunks, ids=new_chunk_ids)
            vector_db.persist()
        
        results = vector_db.similarity_search_with_score(self._prompt, k=kwargs['k'])
        doc = [doc.page_content for doc, _score in results]
        context_text = "\n---\n".join(doc)
        
        prompt_template = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=self._prompt)

        model = Ollama(model=self._model)
        response_text = model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = {
            'response': response_text,
            'sources': sources
        }
        
        return formatted_response
