import numpy as np

from ucall.posix import Server
# from ukv.umem import DataBase
# from usearch.index import Index


server = Server()

@server
def find(request: str) -> str:
    print('Received request:', request)
    return 'Hi there from AWS ' + request


# store = DataBase()
# store_metadata = store['metadata']
# store_images = store['images']
# store_vectors = Index()

server.run()    
