# 使い方

実行

```shell
uv run python -m app.app --log-level info --retrievers bm25,vec
```

グラフ作成

```shell
uv run python -m app.app --export-graph
```

インデックス作成

```shell
uv run python -m app.indexer.indexer --log-level info
```

`--log-level` は `debug|info|warning|error|critical` を指定できます。
`--retrievers` は `bm25,vec` のようにカンマ区切りで指定できます。
CLI未指定時は `RETRIEVERS` 環境変数を参照し、未設定なら `bm25,vec` を使用します。

## Web search environment variables

- `WEB_SEARCH_TIMEOUT_S` (default: `45.0`)
  - Per-request timeout in seconds for OpenAI web search calls.
  - Values below `1.0` are treated as `1.0`.
- `WEB_SEARCH_MAX_RETRIES` (default: `1`)
  - Max retry attempts for OpenAI web search calls.
  - Clamped to `0..5`.

When web search fails (for example timeout or connection errors), the bot continues
answer generation using local FAQ evidence only.
