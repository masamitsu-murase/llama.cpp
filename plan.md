# JSON-RPC 2.0 over stdio ゲートウェイ — 実装計画

## 1. 概要

llama.cpp の推論エンジンに対して、**JSON-RPC 2.0 over stdio** プロトコルでアクセスするゲートウェイプロセス (`llama-server-jsonrpc`) の実装計画である。既存の HTTP サーバー (`tools/server/`) が提供する OpenAI 互換エンドポイント群を、stdio (stdin/stdout) 上の JSON-RPC 2.0 メッセージとして公開する。

### 現状

`tools/server-jsonrpc/` に初期実装が既に存在し、以下の機能が動作する：

- JSON-RPC 2.0 プロトコルの基本パース (`jsonrpc.cpp/h`)
- `models` メソッド（モデル情報取得）
- `chat/completions` メソッド（チャット補完、ストリーミング対応）
- stdio 上の行区切り JSON による入出力
- `server_context` を直接利用した推論実行
- シグナルハンドリングとグレースフルシャットダウン

### 目標

初期実装を拡張し、HTTP サーバーとの機能パリティを達成する。

---

## 2. アーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│  クライアントプロセス (IDE / エディタ / スクリプト)         │
│  stdin ← JSON-RPC Response/Notification (1 行 = 1 JSON) │
│  stdout → JSON-RPC Request (1 行 = 1 JSON)              │
└──────────────┬───────────────────────────┬───────────────┘
               │ stdin                     │ stdout
┌──────────────▼───────────────────────────▼───────────────┐
│  llama-server-jsonrpc                                     │
│                                                           │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────┐ │
│  │ stdio reader │──▶│ JSON-RPC     │──▶│ method        │ │
│  │ (main loop)  │   │ dispatcher   │   │ handlers      │ │
│  └─────────────┘   └──────────────┘   └───────┬───────┘ │
│                                                │         │
│  ┌─────────────┐   ┌──────────────────────────▼───────┐ │
│  │ stdout       │◀──│ server_context                   │ │
│  │ writer       │   │ (推論エンジン / server-context)    │ │
│  │ (mutex保護)  │   └──────────────────────────────────┘ │
│  └─────────────┘                                         │
│                     ログ出力 → stderr                     │
└──────────────────────────────────────────────────────────┘
```

### 設計原則

1. **トランスポート層の分離**: JSON-RPC プロトコル処理と推論ロジックを明確に分離する
2. **server_context の再利用**: HTTP サーバーと同じ `server_context` / `server_task` / `server_response_reader` を活用し、推論コードの重複を避ける
3. **行区切りプロトコル**: 1 行 = 1 JSON メッセージ（改行区切り）。パースが容易で、既存の行指向ツールとの相性がよい
4. **ログは stderr**: stdout は JSON-RPC メッセージ専用。すべてのログ出力は stderr に限定する
5. **並行リクエスト対応**: リクエストごとにスレッドを起動し、複数リクエストを同時処理する

---

## 3. フェーズ別実装計画

### フェーズ 1: 基盤強化（既存コードの改善）

#### 1.1 Content-Length ベースのフレーミング（オプション対応）

現在は行区切り (newline-delimited JSON) だが、LSP (Language Server Protocol) 互換の `Content-Length` ヘッダー方式もオプションで対応する。

- **目的**: LSP クライアントや MCP (Model Context Protocol) との統合を容易にする
- **実装**:
  - コマンドラインオプション `--jsonrpc-framing {newline|content-length}` を追加
  - `content-length` モード時は `Content-Length: N\r\n\r\n{...}` 形式で読み書き
  - デフォルトは `newline` (後方互換性維持)

#### 1.2 バッチリクエスト対応

JSON-RPC 2.0 仕様ではリクエストの配列（バッチ）をサポートする。

- 入力が JSON 配列の場合、各要素を個別のリクエストとして処理
- すべての結果を JSON 配列としてまとめて返却
- 通知のみのバッチでは応答なし（仕様準拠）

#### 1.3 キャンセル機構

進行中のリクエストをキャンセルする仕組みを追加する。

- `$/cancelRequest` 通知を受信した場合、対応する `id` のリクエストを中断
- `server_context` の既存キャンセル機構 (`SERVER_TASK_TYPE_CANCEL`) を活用

---

### フェーズ 2: メソッド拡充

既存の HTTP サーバーが提供するエンドポイントを JSON-RPC メソッドとして公開する。

| JSON-RPC メソッド | HTTP エンドポイント相当 | 説明 |
|---|---|---|
| `models` | `GET /v1/models` | ✅ 実装済み |
| `chat/completions` | `POST /v1/chat/completions` | ✅ 実装済み（ストリーミング対応） |
| `completions` | `POST /v1/completions` | テキスト補完 |
| `embeddings` | `POST /v1/embeddings` | 埋め込みベクトル生成 |
| `rerank` | `POST /v1/reranking` | リランキング |
| `infill` | `POST /infill` | Fill-in-the-middle 補完 |
| `tokenize` | `POST /tokenize` | テキスト→トークン変換 |
| `detokenize` | `POST /detokenize` | トークン→テキスト変換 |
| `health` | `GET /health` | ヘルスチェック |
| `props` | `GET /props` | サーバープロパティ取得 |
| `metrics` | `GET /metrics` | メトリクス取得 |
| `slots` | `GET /slots` | スロット状態取得 |
| `slots/save` | `POST /slots/{id}?action=save` | スロット保存 |
| `slots/restore` | `POST /slots/{id}?action=restore` | スロット復元 |
| `slots/erase` | `POST /slots/{id}?action=erase` | スロット消去 |
| `lora/list` | `GET /lora-adapters` | LoRA アダプタ一覧 |
| `lora/apply` | `POST /lora-adapters` | LoRA アダプタ適用 |

#### 実装方針

- 各メソッドハンドラは `server_routes` の既存ロジックを可能な限り再利用する
- `server_http_req` 相当の構造を JSON-RPC の `params` から構築し、`server_routes` のハンドラに渡す
- レスポンスは `server_res_generator` から取得し、JSON-RPC レスポンスに変換する

---

### フェーズ 3: ストリーミングプロトコルの整備

#### 3.1 ストリーミング通知の標準化

現在のストリーミング実装を整理し、以下の通知フォーマットを標準化する：

```jsonc
// 中間チャンク (server → client 通知)
{
  "jsonrpc": "2.0",
  "method": "chat/completions.chunk",
  "params": {
    "request_id": 1,        // 対応するリクエストの id
    "data": { /* OpenAI SSE chunk 相当 */ }
  }
}

// 最終レスポンス (server → client)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { /* 最終結果 */ }
}
```

#### 3.2 completions メソッドのストリーミング

`chat/completions` と同様に `completions` メソッドでもストリーミングに対応する。通知メソッド名は `completions.chunk` とする。

---

### フェーズ 4: 堅牢性・運用性の向上

#### 4.1 接続ライフサイクル管理

- `initialize` / `initialized` ハンドシェイク（オプション）
  - クライアントがサーバーの capabilities を事前に問い合わせるための仕組み
  - サポートするメソッド一覧やプロトコルバージョンを返す
- `shutdown` / `exit` メソッド
  - クライアントからの明示的なシャットダウン要求に対応

#### 4.2 エラーハンドリングの強化

- アプリケーション固有のエラーコード定義（-32000 〜 -32099 の範囲を使用）

```cpp
enum jsonrpc_app_error_code {
    JSONRPC_MODEL_NOT_LOADED  = -32001,
    JSONRPC_SLOT_UNAVAILABLE  = -32002,
    JSONRPC_CONTEXT_OVERFLOW  = -32003,
    JSONRPC_CANCELLED         = -32004,
    JSONRPC_RATE_LIMITED      = -32005,
};
```

#### 4.3 リクエストタイムアウト

- `params` 内の `timeout_ms` フィールド（オプション）でリクエスト単位のタイムアウトを指定可能にする
- タイムアウト時は `JSONRPC_INTERNAL_ERROR` を返却

#### 4.4 スレッド管理の改善

- デタッチスレッドの代わりにスレッドプールを使用し、リソース管理を改善
- 最大同時リクエスト数の制限（`--jsonrpc-max-concurrent` オプション）
- シャットダウン時に実行中リクエストの完了を待機

---

### フェーズ 5: テスト・ドキュメント

#### 5.1 テスト

- **ユニットテスト**: `jsonrpc.cpp` のパース・シリアライズロジック
- **統合テスト**: Python スクリプトによる E2E テスト
  - プロセス起動 → stdin にリクエスト送信 → stdout からレスポンス読み取り
  - 各メソッドの正常系・異常系
  - ストリーミングの検証
  - バッチリクエストの検証
  - 同時リクエストの検証
  - キャンセルの検証
- **既存テストフレームワーク** (`tools/server/tests/`) のパターンに合わせて `tools/server-jsonrpc/tests/` に配置

#### 5.2 ドキュメント

- `tools/server-jsonrpc/README.md` — 使用方法、対応メソッド一覧、プロトコル仕様
- メソッドごとのリクエスト/レスポンス例
- ストリーミングプロトコルの説明
- クライアント実装ガイド（Python 例）

---

## 4. ファイル構成（最終形）

```
tools/server-jsonrpc/
├── CMakeLists.txt          # ビルド定義
├── README.md               # ドキュメント
├── main.cpp                # エントリポイント、stdin/stdout ループ
├── jsonrpc.cpp             # JSON-RPC 2.0 プロトコル処理
├── jsonrpc.h               # JSON-RPC 2.0 型定義・ユーティリティ
├── jsonrpc-dispatch.cpp    # メソッドディスパッチャ（新規）
├── jsonrpc-dispatch.h      # ディスパッチャヘッダ（新規）
├── jsonrpc-methods.cpp     # 各メソッドハンドラ実装（新規）
├── jsonrpc-methods.h       # メソッドハンドラ宣言（新規）
└── tests/
    ├── test_jsonrpc.py     # 統合テスト
    └── CMakeLists.txt      # テストビルド定義
```

---

## 5. 依存関係

- **内部ライブラリ**: `server-context`, `llama-common`, `llama` (既存の依存と同一)
- **外部ライブラリ**: なし（nlohmann/json は既にプロジェクトに含まれる）
- **ビルドシステム**: 既存の CMake 構成に統合済み

---

## 6. 実装優先順位

| 優先度 | 項目 | フェーズ |
|---|---|---|
| **P0** | `completions` / `embeddings` メソッド追加 | 2 |
| **P0** | ユニットテスト・統合テスト | 5 |
| **P1** | `tokenize` / `detokenize` / `health` メソッド追加 | 2 |
| **P1** | キャンセル機構 (`$/cancelRequest`) | 1 |
| **P1** | ドキュメント (README.md) | 5 |
| **P2** | スレッドプール化・同時実行数制限 | 4 |
| **P2** | アプリケーション固有エラーコード | 4 |
| **P2** | `rerank` / `infill` メソッド追加 | 2 |
| **P3** | Content-Length フレーミング対応 | 1 |
| **P3** | バッチリクエスト対応 | 1 |
| **P3** | `initialize` / `shutdown` ハンドシェイク | 4 |
| **P3** | スロット管理・LoRA メソッド | 2 |

---

## 7. セキュリティ考慮事項

- **入力バリデーション**: すべての JSON-RPC パラメータを厳密に検証する（既存の `jsonrpc_parse_request` を拡張）
- **リソース制限**: 最大リクエストサイズの制限（巨大な JSON による DoS 防止）
- **プロセス分離**: stdio 通信であるため、ネットワーク経由のアクセスは不可。ローカルプロセス間通信に限定される
- **ログ漏洩防止**: ユーザー入力（プロンプト内容）をログに含めない設定を提供

---

## 8. 使用例

### 基本的なリクエスト・レスポンス

```bash
# サーバー起動
$ llama-server-jsonrpc -m model.gguf 2>/dev/null

# stdin に送信
{"jsonrpc":"2.0","id":1,"method":"models"}

# stdout に返却
{"jsonrpc":"2.0","id":1,"result":{"object":"list","data":[...]}}
```

### チャット補完（非ストリーミング）

```bash
{"jsonrpc":"2.0","id":2,"method":"chat/completions","params":{"messages":[{"role":"user","content":"Hello!"}],"stream":false}}
```

### チャット補完（ストリーミング）

```bash
# リクエスト
{"jsonrpc":"2.0","id":3,"method":"chat/completions","params":{"messages":[{"role":"user","content":"Tell me a joke"}],"stream":true}}

# サーバーからの出力（複数行）
{"jsonrpc":"2.0","method":"chat/completions.chunk","params":{"request_id":3,"data":{...}}}
{"jsonrpc":"2.0","method":"chat/completions.chunk","params":{"request_id":3,"data":{...}}}
{"jsonrpc":"2.0","id":3,"result":{...}}
```

### Python クライアント例

```python
import subprocess, json

proc = subprocess.Popen(
    ["llama-server-jsonrpc", "-m", "model.gguf"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1
)

def rpc_call(method, params=None, req_id=1):
    req = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        req["params"] = params
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())

# モデル情報取得
print(rpc_call("models"))

# チャット補完
print(rpc_call("chat/completions", {
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": False
}, req_id=2))
```
