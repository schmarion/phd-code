import os

import litellm
from ecologits import EcoLogits
from py2neo import Graph


class LLMRetriever:
    """KG retriever based on LLM for the RAG task."""

    def __init__(self) -> None:
        self.model_name = os.environ.get("GENERATION_MODEL")
        self.graph = Graph(
            uri=os.environ.get("GRAPH_URI"),
            user=os.environ.get("GRAPH_USER"),
            password=os.environ.get("GRAPH_PASSWORD"),
            name=os.environ.get("GRAPH_DATABASE"),
        )

    def _get_kg_schema(self) -> str:
        """Extract KG schema from Neo4j.
        The query can be uncoment to be executed.
        The result of the query is directly assigned to a variable to save time."""
        # kg_schema = ""
        # nodes = self.graph.run("MATCH (n) RETURN DISTINCT LABELS(n), KEYS(n)").data()
        # if len(nodes) > 0:
        #     kg_schema += "Labels des noeuds et propriétés :\n"
        # for node in nodes:
        #     kg_schema += f"Le type de noeud {node['LABELS(n)'][0]} a pour propriétés {', '.join(node['KEYS(n)'])}\n"
        # relations = self.graph.run("MATCH (a)-[r]->(b) RETURN DISTINCT LABELS(a),TYPE(r),LABELS(b)").data()
        # print(relations)
        # if len(relations) > 0:
        #     kg_schema += "Relations :\n"
        # for relation in relations:
        #     kg_schema += f"{relation['LABELS(a)'][0]} - {relation['TYPE(r)']} -> {relation['LABELS(b)'][0]}\n"
        kg_schema = """Labels des noeuds et propriétés :
        Le type de noeud ServiceMunicipal a pour propriétés telephone, courriel, adresse, infos, nom
        Le type de noeud DossierAdministratif a pour propriétés infos, nom
        Le type de noeud Mission a pour propriétés infos, nom
        Le type de noeud ServiceMunicipal a pour propriétés infos, nom
        Le type de noeud ServiceMunicipal a pour propriétés courriel, adresse, infos, nom
        Le type de noeud Activite a pour propriétés infos, nom, public_cible, type
        Le type de noeud Mission a pour propriétés nom
        Le type de noeud ServiceMunicipal a pour propriétés telephone, infos, nom, adresse, courriel
        Le type de noeud Association a pour propriétés infos, nom
        Le type de noeud MDPH a pour propriétés telephone, courriel, nom, adresse
        Le type de noeud Activite a pour propriétés public_cible, nom
        Le type de noeud Activite a pour propriétés nom
        Le type de noeud TravailleurSocial a pour propriétés nom, infos
        Le type de noeud Association a pour propriétés nom, infos
        Le type de noeud Activite a pour propriétés nom, public_cible
        Le type de noeud Mission a pour propriétés nom, infos
        Le type de noeud Activite a pour propriétés nom, type
        Le type de noeud ProfessionnelSante a pour propriétés nom, infos
        Le type de noeud Activite a pour propriétés nom, infos, public_cible, type
        Le type de noeud Activite a pour propriétés nom, public_cible, infos
        Le type de noeud ProfessionnelSante a pour propriétés nom
        Le type de noeud Aide a pour propriétés nom, domaine, infos, public_cible
        Le type de noeud Aide a pour propriétés nom, public_cible, infos
        Le type de noeud ServiceMunicipal a pour propriétés nom
        Le type de noeud Aide a pour propriétés infos, nom
        Relations :
        ServiceMunicipal - COORDONNE -> ServiceMunicipal
        ServiceMunicipal - PROPOSE -> Aide
        ServiceMunicipal - PROPOSE -> Activite
        ServiceMunicipal - A_POUR_MISSION -> Mission
        ServiceMunicipal - PROPOSE -> TravailleurSocial
        ServiceMunicipal - PROPOSE -> ProfessionnelSante
        Activite - A_POUR_MISSION -> Mission
        Association - PROPOSE -> Activite
        Association - ANIME -> ServiceMunicipal
        Association - A_POUR_MISSION -> Mission
        MDPH - TRAITE -> DossierAdministratif
        TravailleurSocial - A_POUR_MISSION -> Mission
        ProfessionnelSante - A_POUR_MISSION -> Mission
        ProfessionnelSante - ANIME -> Activite"""
        return kg_schema

    def _generate_prompt(self, question: str) -> list[dict[str, str]]:
        """Generate contextualised prompt for query generation."""
        prompt = [
            {
                "role": "system",
                "content": """Tu es un assistant qui aide à formuler des requêtes Cypher.""",
            },
            {
                "role": "user",
                "content": """Ecrit une requête Cypher pour répondre à la question de l'utilisateur en te basant sur le schéma du graphe Neo4j donné.  
                Utilise uniquement les labels, les propriétés et les types de relation fournis dans le schéma.
                N'ajoute aucun texte, tu dois uniquement répondre avec la requête Cypher générée.
                N'ajoute pas de majuscules aux entités extraites de la question.""",
            },
            {
                "role": "user",
                "content": f"""Schéma du graphe: {self._get_kg_schema()}""",
            },
            {
                "role": "user",
                "content": f"""Question: {question.lower()}""",
            },
        ]
        return prompt

    def _generate_query(self, prompt: list[dict[str, str]]) -> dict[str, any]:
        """Generate query with a LLM based on a given question and a graph schema."""
        EcoLogits.init(providers="litellm", electricity_mix_zone="SWE")
        response = litellm.completion(
            model=self.model_name, messages=prompt, temperature=0.0
        )
        output = {
            "query": response["choices"][0]["message"]["content"],
            "env_impacts": response.impacts,
        }
        return output

    def _execute_query(self, query: str) -> list[dict[str, any]]:
        """Execute query on the graph."""
        facts = self.graph.run(query).data()
        return facts

    def _format_facts(self, facts: list[dict[str, any]]) -> str:
        """Convert facts from Neo4j to text for the prompt content."""
        context = ""
        if len(facts) == 0:
            context = "[]"
        else:
            for fact in facts:
                context += f"{str(fact)} \n"
        return context

    def run(self, question: str) -> dict[str, any]:
        """Retrieve context for a question based on a graph.."""
        prompt = self._generate_prompt(question)
        query_output = self._generate_query(prompt)
        graph_context = self._execute_query(query_output["query"])
        output = {
            "query": query_output["query"],
            "context": self._format_facts(graph_context),
            "env_impacts": query_output["env_impacts"],
        }
        return output
