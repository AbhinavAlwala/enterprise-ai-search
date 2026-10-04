import socket

import pytest


@pytest.fixture(autouse=True)
def prevent_http_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must mock outbound network I/O")

    # Block HTTP connection creation while preserving Windows' internal event-loop socket pairs.
    monkeypatch.setattr(socket, "create_connection", forbidden)
