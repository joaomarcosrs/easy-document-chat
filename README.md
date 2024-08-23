# Easy Document Chat

### Installation:
#### Clone the repository
```bash
git clone https://github.com/joaomarcosrs/easy-document-chat.git
```
#### After creating your virtual environment and activating it, install the requirements
```bash
pip install -r requirements.txt
```
### Usage:
```python3
from documentChat import EmbeddingModel, PDFChat


files_path = 'path/to/directory/with/pdfs/'

query_text = '<question string about the document>'

embedding_model = EmbeddingModel(embedding='ollama')
embedding = embedding_model.embedding(model='<ollama embedding model>')

model = PDFChat(
    prompt=query_text,
    path=files_path,
    llm='ollama',
    model=<ollama model>,
    embedding=embedding,
)
model.run()

result = model.response(chunk_size=<int>, chunk_overlap=<int>, length_function=len, is_separator_regex=False, k=<int>)

print(result.get('response'))
print(result.get('sources'))
```


