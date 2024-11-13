import itertools
import os

import spacy
from buzz_el.entity_matcher import EntityMatcher
from buzz_el.graph import KnowledgeGraph
from py2neo import Graph


class ELRetriever:
    """Retriever based on entity linking and query patterns."""

    def __init__(self) -> None:
        self.graph = Graph(
            uri=os.environ.get("GRAPH_URI"),
            user=os.environ.get("GRAPH_USER"),
            password=os.environ.get("GRAPH_PASSWORD"),
            name=os.environ.get("GRAPH_DATABASE"),
        )
        kg = self._cypher_graph_loader()
        self.spacy_model = spacy.load("fr_core_news_lg")
        self.em = EntityMatcher(
            knowledge_graph=kg, spacy_model=self.spacy_model, use_fuzzy=True
        )
        self.qp = QueryPattern()

    def _cypher_graph_loader(self) -> KnowledgeGraph:
        """Create a KG with entity patterns created from Neo4j graph content."""
        patterns = []
        node_types = self.graph.run("MATCH(n) RETURN DISTINCT LABELS(n)").data()
        for node_type in node_types:
            patterns.append(
                {
                    "label": "KG_ENT",
                    "pattern": node_type["LABELS(n)"][0],
                    "id": f"{node_type['LABELS(n)'][0]}",
                }
            )
        nodes = self.graph.run("MATCH(n) RETURN DISTINCT LABELS(n), n.nom").data()
        for node in nodes:
            patterns.append(
                {
                    "label": node["LABELS(n)"][0],
                    "pattern": node["n.nom"],
                    "id": f"{node['LABELS(n)'][0]}:{node['n.nom']}",
                }
            )

        def get_context(entity: str) -> str:
            return entity

        kg = KnowledgeGraph(
            kg=self.graph, entity_patterns=patterns, get_entity_context=get_context
        )
        return kg

    def _extract_entities(self, question: str) -> list[str]:
        """Extract KG entites from a user utterance using spaCy matcher."""
        spacy_question = self.spacy_model(question)
        spacy_question = self.em(spacy_question)
        entities = []
        for entity in spacy_question.spans["fuzzy"]:
            entities.append(entity.id_)
        return entities

    def _execute_queries(self, queries: list[str]):
        """Execute queries on the graph."""
        facts = []
        for query in queries:
            facts += self.graph.run(query).data()
        return facts

    def run(self, question: str) -> dict[str, any]:
        """Retrieve context for a question based on a graph."""
        context = ""
        question_entities = self._extract_entities(question)
        queries = self.qp.run(question_entities)
        facts = self._execute_queries(queries)
        if len(facts) == 0:
            queries = [
                self.qp.create_one_entity_query(entity) for entity in question_entities
            ]
            facts = self._execute_queries(queries)
        for fact in facts:
            context += str(fact)
            context += "\n"
        if len(context) == 0:
            context = "[]"
        output = {
            "question_entities": question_entities,
            "queries": queries,
            "context": context,
            "llm_query": "",
        }
        return output


class QueryPattern:
    """Query patterns construction for graph context extraction."""

    def create_one_entity_query(self, entity: str) -> str:
        """Create cypher query from an entity."""
        if ":" in entity:
            sep_pos = entity.find(":")
            query = f"""MATCH (e:{entity[:sep_pos]}) WHERE (e.nom="{entity[sep_pos+1:]}") RETURN e"""
        else:
            query = f"""MATCH (e:{entity}) RETURN e"""
        return query

    def _create_multiple_entities_query(self, entity_group: list[str]) -> str:
        """Create cypher queries from entity combinations."""
        query_begin = "MATCH "
        query_middle = " WHERE "
        query_end = " RETURN "
        and_sep = False
        for i, entity in enumerate(entity_group):
            if ":" in entity:
                sep_pos = entity.find(":")
                entity_type = entity[:sep_pos]
                entity_name = entity[sep_pos + 1 :]
                if and_sep:
                    query_middle += " AND "
                    and_sep = False
                query_middle += f'(e{i}.nom="{entity_name}")'
                and_sep = True
            else:
                entity_type = entity
            if i == 0:
                query_begin += f"""(e{i}:{entity_type}) -[r{i}]-> """
                query_end += f"""e{i} as entity_{i}, type(r{i}) as relation_{i}, """
            elif i == len(entity_group) - 1:
                query_begin += f"""(e{i}:{entity_type})"""
                query_end += f"""e{i} as entity_{i}"""
            else:
                query_begin += f"""(e{i}:{entity_type}) -[r{i}]-> """
                query_end += f"""e{i} as entity_{i}, type(r{i}) as relation_{i}, """
        return query_begin + query_middle + query_end

    def run(self, entities: list[str]) -> list[str]:
        """Create cypher queries based on patterns and entities found in the question."""
        queries = []
        if len(entities) == 1:
            queries.append(self.create_one_entity_query(entities[0]))
        elif len(entities) > 1:
            entity_groups = itertools.permutations(entities, len(entities))
            for entity_group in entity_groups:
                queries.append(self._create_multiple_entities_query(entity_group))
        return queries
