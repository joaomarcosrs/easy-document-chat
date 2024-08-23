import importlib
from langchain_community import embeddings


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