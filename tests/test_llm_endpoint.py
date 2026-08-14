"""Checks that the Bedrock endpoint stays shut.

The failure this suite exists to catch is the endpoint being reachable without
a token, or reachable with one but willing to invoke an arbitrary model. Both
would cost money on someone else's account.

No AWS call is made anywhere in here — every test is refused before it would
reach Bedrock, which is the point.
"""

from __future__ import annotations

import importlib

import pytest

import app as app_module
import auth
import bedrock
import config

TOKEN = "test-token-do-not-use-in-production"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_limiter():
    """Each test gets a fresh limiter, or ordering decides who gets 429s."""
    auth.limiter = auth.RateLimiter()
    app_module.limiter = auth.limiter
    yield


@pytest.fixture
def enabled(monkeypatch):
    """Endpoint switched on with one allowed model."""
    monkeypatch.setattr(config, "LLM_API_TOKEN", TOKEN)
    monkeypatch.setattr(config, "BEDROCK_ALLOWED_MODELS", ("test.model-v1",))
    return {"Authorization": f"Bearer {TOKEN}"}


def post(client, headers=None, **body):
    payload = {"model_id": "test.model-v1", "prompt": "hello"}
    payload.update(body)
    return client.post("/api/llm/generate", json=payload, headers=headers or {})


class TestClosedByDefault:
    def test_no_token_configured_hides_the_route(self, client, monkeypatch):
        """With LLM_API_TOKEN unset the endpoint must not exist at all."""
        monkeypatch.setattr(config, "LLM_API_TOKEN", "")
        assert post(client).status_code == 404

    def test_no_token_configured_hides_it_even_from_a_guesser(
            self, client, monkeypatch):
        monkeypatch.setattr(config, "LLM_API_TOKEN", "")
        response = post(client, headers={"Authorization": "Bearer anything"})
        assert response.status_code == 404

    def test_missing_header_is_indistinguishable_from_a_missing_route(
            self, client, enabled):
        response = post(client)
        assert response.status_code == 404
        assert response.get_json()["error"] == "Not found."

    def test_wrong_token_is_indistinguishable_from_a_missing_route(
            self, client, enabled):
        response = post(client, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 404
        assert response.get_json()["error"] == "Not found."

    def test_token_as_query_param_is_not_accepted(self, client, enabled):
        """Query strings land in access logs and browser history."""
        response = client.post(f"/api/llm/generate?token={TOKEN}",
                               json={"model_id": "test.model-v1", "prompt": "x"})
        assert response.status_code == 404

    def test_models_route_is_hidden_too(self, client, monkeypatch):
        monkeypatch.setattr(config, "LLM_API_TOKEN", "")
        assert client.get("/api/llm/models").status_code == 404


class TestModelAllowlist:
    def test_empty_allowlist_refuses_everything(self, client, monkeypatch):
        monkeypatch.setattr(config, "LLM_API_TOKEN", TOKEN)
        monkeypatch.setattr(config, "BEDROCK_ALLOWED_MODELS", ())
        response = post(client, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 503

    def test_model_outside_the_allowlist_is_refused(self, client, enabled):
        response = post(client, headers=enabled,
                        model_id="anthropic.something-expensive")
        assert response.status_code == 400
        assert "allowed list" in response.get_json()["error"]

    def test_allowlist_is_exact_not_a_prefix_match(self, client, enabled):
        response = post(client, headers=enabled, model_id="test.model-v1-plus")
        assert response.status_code == 400

    def test_check_model_raises_rather_than_returning_false(self):
        with pytest.raises(bedrock.BedrockError):
            bedrock.check_model("not-allowed")


class TestInputLimits:
    def test_missing_prompt_is_rejected(self, client, enabled):
        response = post(client, headers=enabled, prompt="")
        assert response.status_code == 400

    def test_missing_model_id_is_rejected(self, client, enabled):
        response = post(client, headers=enabled, model_id="")
        assert response.status_code == 400

    def test_oversized_prompt_is_rejected_before_any_aws_call(
            self, client, enabled, monkeypatch):
        monkeypatch.setattr(config, "BEDROCK_MAX_PROMPT_CHARS", 100)
        response = post(client, headers=enabled, prompt="x" * 101)
        assert response.status_code == 413


class TestRateLimiting:
    def test_per_minute_cap_applies(self, client, enabled, monkeypatch):
        monkeypatch.setattr(config, "LLM_RATE_PER_MINUTE", 3)
        monkeypatch.setattr(config, "LLM_RATE_PER_DAY", 1000)
        # Requests are refused at the allowlist stage, which is after the
        # limiter — so they still consume quota, which is what we're checking.
        codes = [post(client, headers=enabled, model_id="nope").status_code
                 for _ in range(5)]
        assert codes.count(429) == 2, codes

    def test_daily_cap_applies(self, client, enabled, monkeypatch):
        monkeypatch.setattr(config, "LLM_RATE_PER_MINUTE", 1000)
        monkeypatch.setattr(config, "LLM_RATE_PER_DAY", 2)
        codes = [post(client, headers=enabled, model_id="nope").status_code
                 for _ in range(4)]
        assert codes.count(429) == 2, codes

    def test_rate_limit_is_checked_before_the_model_allowlist(
            self, client, enabled, monkeypatch):
        """A caller with a bad model can't probe the allowlist without limit."""
        monkeypatch.setattr(config, "LLM_RATE_PER_MINUTE", 1)
        first = post(client, headers=enabled, model_id="nope")
        second = post(client, headers=enabled, model_id="nope")
        assert first.status_code == 400 and second.status_code == 429

    def test_unauthenticated_requests_do_not_consume_quota(
            self, client, enabled, monkeypatch):
        """Otherwise anyone could exhaust the real caller's quota unauthenticated."""
        monkeypatch.setattr(config, "LLM_RATE_PER_MINUTE", 2)
        for _ in range(10):
            post(client, headers={"Authorization": "Bearer wrong"})
        assert post(client, headers=enabled, model_id="nope").status_code == 400


class TestErrorTranslation:
    def test_aws_detail_is_kept_off_the_wire(self, client, enabled, monkeypatch):
        """AWS messages name account IDs and ARNs; callers must not see them."""
        secret = "arn:aws:iam::123456789012:role/SuperSecretRole"

        def boom(*args, **kwargs):
            raise bedrock.BedrockError(
                "This server isn't allowed to use that model.",
                status=403, detail=secret)

        monkeypatch.setattr(bedrock, "converse", boom)
        response = post(client, headers=enabled)
        assert response.status_code == 403
        assert secret not in response.get_data(as_text=True)

    def test_models_route_reports_what_is_enabled(self, client, enabled):
        body = client.get("/api/llm/models", headers=enabled).get_json()
        assert body["models"] == ["test.model-v1"]
        assert body["limits"]["daily_limit"] == config.LLM_RATE_PER_DAY
