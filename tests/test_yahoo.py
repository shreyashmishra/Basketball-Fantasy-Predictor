from fantasy.yahoo import sync_league


def test_yahoo_absent_falls_back(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("YAHOO_CLIENT_ID", raising=False)
    monkeypatch.delenv("YAHOO_CLIENT_SECRET", raising=False)
    result = sync_league("test.league", env_path=tmp_path / ".missing.env", token_path=tmp_path / "oauth2.json")
    assert result.enabled is False
    assert "config.yaml" in result.message

