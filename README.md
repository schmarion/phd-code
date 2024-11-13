# Towards efficient Knowledge Graph-based Retrieval Augmented Generation for conversational agents

This repository contains the code associated with the thesis manuscript. 

We are building a conversational agent based on KG-RAG on the theme of disability management in a French town. 

To use the project code:
```
git clone https://github.com/schmarion/phd-code.git
cd phd-code
```  

## KG construction

As we are using a KG-RAG architecture, we need a knowledge graph. 
We are interested in automatic KG construction tools and ontology learning.
The folder `kg_construction` contains the data used to build the KGs, the construction scripts and the KGs produced.

- The `pipeline.ipynb` notebook creates a KG from an input text and an ontology learning pipeline.
- The folders `asnieres_sur_seine`, `puteaux` and `villeneuve_la_garenne` contain information about the cities. 
    - `kr.json` files contain the knowledge representations built using the [OLAF](https://github.com/wikit-ai/olaf) framework. 
    - `text.md` files contain the text used to construct the knowledge representations. 
    - `onto_owl.ttl` files contain the OWL ontologies built using the [OLAF](https://github.com/wikit-ai/olaf) framework. 

To use the notebook: 
```
cd kg_construction
python3 -m venv ./kg_construction_env
source kg_construction_env/bin/activate
pip install -r requirements.txt
```

## RAG

The `rag` folder contains the code used to implement the RAG architecture. 2 types of RAG are available: RAG on a graph (KG-RAG) and RAG on text (Text-RAG). It includes:

- The folder `retriever` contains all the methods available to retrieve relevant context to answer a user utterance.
    - `el_retriever.py` implements entity linking and query patterns to extract context from the graph.
    - `llm_retriever.py` implements LLM-based query generation to extract context from the graph.
    - `hybrid_retriever.py` uses first `el_retriever` and then `llm_retriever` if the context is empty.
    - `chunks.json` contains the chunks obtained using the [Chunk Norris](https://github.com/wikit-ai/chunknorris) library on the disability management data for Villeneuve-la-Garenne
    - `text_similarity_retriever.py` implements embeddings similarity to extract the relevant chunks. 
- The folder `generator` contains the LLM generator method for generating an answer based on a user utterance and the extracted context.
- The folder `kg_rag` contains the implementation of the KG-RAG architecture.
- The folder `text_rag` contains the implementation of the Text-RAG architecture.

The code is available as a FastAPI service and can be used with Docker. 

To install the dependencies:
```
cd rag
python3 -m venv ./rag_env
source rag_env/bin/activate
pip install -r requirements.txt
```

To launch the application with FastAPI:
```
uvicorn main:app --reload
```
The swagger is then available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

To launch the application with Docker: 
```
docker build -t city-rag:1.0 .
docker run -p 8080:8080 -e GRAPH_URI="" -e GRAPH_USER="" -e GRAPH_PASSWORD="" -e GRAPH_DATABASE="" -e EMBEDDING_MODEL="" -e MODEL_NAME="" -e MISTRAL_API_KEY="" city-rag:1.0
```
The swagger is then available at [http://0.0.0.0:8080/docs](http://0.0.0.0:8000/docs).