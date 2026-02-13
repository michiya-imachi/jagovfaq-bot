# 使い方

実行

```shell
uv run python -m app.app --log-level info
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
