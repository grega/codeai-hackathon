# Bedrock prompt endpoint

`POST /api/llm/generate` runs one prompt against one Bedrock model. It exists so
the pose and rigging teams can try prompts against the same credentials the
deployed app uses, without each building their own harness.

It spends real money on a real AWS account, so it is closed by default and every
layer fails closed.

## Enabling it

Both are required; either one missing leaves the endpoint unusable.

```bash
LLM_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
BEDROCK_ALLOWED_MODELS="<model-id>,<model-id>"
AWS_DEFAULT_REGION=eu-west-1
```

Plus AWS credentials in the environment — `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` for temporary credentials.
boto3 reads them itself; no code here touches them.

Model IDs are account- and region-specific, and many newer models are only
invocable through a cross-region inference profile whose ID carries a geography
prefix. List yours:

```bash
aws bedrock list-inference-profiles --region "$AWS_DEFAULT_REGION"
aws bedrock list-foundation-models  --region "$AWS_DEFAULT_REGION"
```

On Heroku: `heroku config:set LLM_API_TOKEN=... BEDROCK_ALLOWED_MODELS=...`

## Using it

```bash
curl -s https://YOUR_APP/api/llm/generate \
  -H "Authorization: Bearer $LLM_API_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "model_id": "<model-id>",
        "prompt": "Describe a stick figure waving.",
        "system": "You reply in one sentence.",
        "max_tokens": 200
      }'
```

```json
{
  "model_id": "<model-id>",
  "text": "...",
  "stop_reason": "end_turn",
  "usage": { "input_tokens": 21, "output_tokens": 48, "total_tokens": 69 }
}
```

| Field | Required | Notes |
|---|---|---|
| `model_id` | yes | Must be in `BEDROCK_ALLOWED_MODELS` |
| `prompt` | yes | Up to `BEDROCK_MAX_PROMPT_CHARS` (20k default) |
| `system` | no | System prompt |
| `max_tokens` | no | Default 1024, capped at `BEDROCK_MAX_TOKENS` |
| `temperature` | no | **Omitted unless you send it** — newer Anthropic models reject sampling parameters, so a default would break them |

`GET /api/llm/models` returns the allowlist, the region, and current rate-limit
usage. Both routes take the same token.

## How it is protected

| Layer | Behaviour |
|---|---|
| No `LLM_API_TOKEN` set | Both routes return **404** |
| Missing or wrong token | **404**, identical to a route that doesn't exist |
| Token comparison | `secrets.compare_digest`, on every path including the missing-header one |
| Empty `BEDROCK_ALLOWED_MODELS` | Every model refused |
| Model not on the allowlist | Refused before any AWS call — exact match, not a prefix |
| Per-caller rate limit | `LLM_RATE_PER_MINUTE` (10), bucketed by token **and** client address |
| Whole-deployment cap | `LLM_RATE_PER_DAY` (500) — this is the one that bounds the bill |
| Prompt size | Rejected before any AWS call |
| AWS errors | Translated to generic messages; the original goes to the server log |

**404 rather than 401** is deliberate: an unauthenticated caller cannot tell the
endpoint from a typo'd URL, so a scanner finds nothing to come back to. The
tradeoff is that a developer with a stale token also sees a 404 — the server log
distinguishes the two (`[llm] refused: bad or missing token from ...`).

**Unauthenticated requests consume no quota.** The token check runs before the
rate limiter, so nobody can exhaust a real caller's allowance without the token.

**Rate limiting is per-process and in memory.** It resets on restart and does not
coordinate across dynos. That is sound while the app runs on one dyno (which the
in-memory store already requires), and would need Redis if that ever changes.

## The token is not a browser secret

The frontend does not call this endpoint and must not. Anything shipped to the
browser is public — put this token in JavaScript and you have published an open
LLM proxy. Callers are developers with curl, or server-side code.

If a browser feature ever needs model output, the right shape is a
purpose-specific endpoint (like `/api/avatars/<id>/poses`) that takes a
constrained input and calls the model server-side, not a general prompt pipe.

## Why the Converse API

Converse is model-agnostic — Anthropic, Meta, Mistral and Amazon Nova all take
the same request and return the same response — which is what lets the caller
name the model per request without `bedrock.py` knowing anything about model
families.

The tradeoff is that Converse exposes a lowest common denominator. For
Anthropic-specific features — the full Messages API surface, adaptive thinking,
prompt caching — use the `AnthropicBedrockMantle` client from the `anthropic`
SDK, which takes `anthropic.`-prefixed model IDs. That is a second provider
alongside this one, not a change to it.

## Rotating the token

```bash
heroku config:set LLM_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

The dyno restarts and every old token stops working immediately. There is no
revocation list because there is one token — if you need per-person revocation,
that is the point at which this should become real credentials rather than a
shared secret.

## Troubleshooting

| Symptom | Cause |
|---|---|
| 404 with a token you believe is right | Token mismatch or `LLM_API_TOKEN` unset — check the server log |
| 503 "No models are enabled" | `BEDROCK_ALLOWED_MODELS` is empty |
| 503 "isn't configured" | No `AWS_DEFAULT_REGION` / `BEDROCK_REGION` |
| 400 "wasn't valid for this model" | Usually a model ID that doesn't exist in that region, or `temperature` sent to a model that rejects it |
| 403 "isn't allowed to use that model" | The IAM principal lacks `bedrock:InvokeModel`, or model access isn't granted in the Bedrock console |
| 503 "credentials have expired" | `AWS_SESSION_TOKEN` expired — temporary credentials are short-lived |
