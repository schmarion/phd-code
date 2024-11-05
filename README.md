# Towards efficient Knowledge Graph-based Retrieval Augmented Generation for conversational agents

This repository contains the code associated with the thesis manuscript. 

We are building a conversational agent based on KG-RAG on the theme of disability management in a French town. 

To use the project code:
```
git clone https://github.com/schmarion/phd-code.git
cd phd-code
python3 -m venv phd_env
source phd_env/bin/activate
pip install -r requirements.txt
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