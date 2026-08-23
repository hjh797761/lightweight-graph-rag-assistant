class GraphRAGError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(GraphRAGError):
    pass


class ModelUnavailableError(GraphRAGError):
    pass


class DocumentParseError(GraphRAGError):
    pass


class StorageError(GraphRAGError):
    pass


class MigrationError(GraphRAGError):
    pass
