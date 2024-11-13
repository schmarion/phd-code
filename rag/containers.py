from dependency_injector import containers, providers
from kg_rag.service import KGRAG
from text_rag.service import TextRAG


class ApplicationContainer(containers.DeclarativeContainer):
    """The ApplicationContainer provides integration of services,
    databases and other main components into the application.
    It's an easy way to configure the wiring and dependencies of the application.
    It provides features such as service registration and access, configuration,
    database access and integration of third-party components."""

    wiring_config = containers.WiringConfiguration(
        modules=["kg_rag.router", "text_rag.router"]
    )
    configuration = providers.Configuration()

    # Singleton without dependency injection
    kg_rag_service = providers.Singleton(KGRAG)
    text_rag_service = providers.Singleton(TextRAG)
