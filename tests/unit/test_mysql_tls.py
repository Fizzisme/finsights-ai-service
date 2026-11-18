import ssl

import pytest

from app.core.settings import Settings
from app.storage.mysql import engine as mod

NAME = "MySQL_Endpoint_Server"


@pytest.mark.parametrize("node", ["mysql_source_tls_ca", "mysql_chat_tls_ca"])
def test_context_verifies_chain_and_name(node: str) -> None:
    ca = getattr(Settings(), node)
    assert ca.is_file()
    context = mod.tls_context(ca, NAME)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2


def test_handshake_name_is_the_endpoint_not_the_dialled_ip() -> None:
    context = mod.tls_context(Settings().mysql_chat_tls_ca, NAME)
    sslobj = context.wrap_bio(
        ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="161.33.181.173"
    )
    assert sslobj.server_hostname == NAME


def test_tls_can_be_disabled_for_a_local_mysql(monkeypatch) -> None:
    settings = Settings(mysql_tls=False)
    monkeypatch.setattr(mod, "get_settings", lambda: settings)
    args = mod._connect_args(settings.mysql_chat_tls_ca, init_command="X")
    assert "ssl" not in args and args["init_command"] == "X"


def test_tls_is_on_by_default(monkeypatch) -> None:
    settings = Settings()
    monkeypatch.setattr(mod, "get_settings", lambda: settings)
    assert isinstance(
        mod._connect_args(settings.mysql_source_tls_ca)["ssl"], ssl.SSLContext
    )
