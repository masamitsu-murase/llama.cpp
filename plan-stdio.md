# JSON-RPC 2.0 over stdio ゲートウェイの実装計画

## 概要

llama.cpp の既存サーバーアーキテクチャ（server-context ライブラリ）を再利用し、JSON-RPC 2.0 プロトコルで stdio（標準入出力）経由の通信を行う新しいツール llama-server-jsonrpc を作成する。エディタ/IDE連携やパイプライン呼び出しを想定。


## アーキテクチャ設計

### トランスポート層

* 入力: stdin から1行ずつ JSON-RPC 2.0 リクエストを読み取る（改行区切り）
* 出力: stdout に JSON-RPC 2.0 レスポンスを1行ずつ書き出す
* ログ: stderr に出力（stdout とレスポンスが混ざらないようにする）

### JSON-RPC 2.0 マッピング

リクエスト形式:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "chat/completions",
  "params": {
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.8,
    "stream": false
  }
}
```

レスポンス形式（非ストリーミング）:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [...]
  }
}
```

エラー形式:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32600,
    "message": "Invalid Request",
    "data": { ... }
  }
}
```

### 対応メソッド

|JSON-RPC method|対応する既存処理|説明|
|----------------|----------------|----|
|chat/completions|post_chat_completions → handle_completions_impl|チャット推論|
|models|get_models|モデル情報取得|

### ストリーミング（案A: JSON-RPC Notification 方式）

ストリーミングリクエスト時、中間結果を JSON-RPC Notification（id なし）として送信し、最終結果を通常のレスポンス（id あり）として送信する。


中間チャンク（Notification）:

```json
{
  "jsonrpc": "2.0",
  "method": "chat/completions.chunk",
  "params": {
    "request_id": 1,
    "choices": [{"delta": {"content": "Hello"}, "index": 0}]
  }
}
```


最終レスポンス:


```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "choices": [...],
    "usage": {...}
  }
}
```

### 並行リクエスト

* stdin からのリクエストはメインスレッドで読み取り、各リクエストを別スレッドで処理する
* 複数のリクエストが並列に来ても、既存の server_context のスロット/キュー機構がスケジューリングを行う
* stdout への書き込みは mutex で排他制御する
* バッチリクエスト（JSON-RPC の配列形式）は非対応とし、受信時は -32600 Invalid Request を返す

### 実装ファイル構成

新規作成するファイル:

* tools/server-jsonrpc/ — 新しいツールディレクトリ
* tools/server-jsonrpc/CMakeLists.txt — ビルド定義
* tools/server-jsonrpc/main.cpp — エントリポイント（引数解析、モデルロード、stdinリーダーループ）
* tools/server-jsonrpc/jsonrpc.h — JSON-RPC 2.0 プロトコル処理（パース、バリデーション、レスポンス生成）
* tools/server-jsonrpc/jsonrpc.cpp — JSON-RPC 2.0 実装


変更するファイル:

* tools/CMakeLists.txt — add_subdirectory(server-jsonrpc) を追加


### 各コンポーネントの詳細

1. jsonrpc.h / jsonrpc.cpp — JSON-RPC 2.0 プロトコル層
   * JSON-RPC 2.0 リクエストのパース・バリデーション
     * jsonrpc フィールドが "2.0" であることの検証
     * method が文字列であることの検証
     * id の存在確認（Notification でない場合）
     * 配列（バッチ）リクエストの拒否
   * 標準エラーコード定義
     * -32700 Parse error
     * -32600 Invalid Request
     * -32601 Method not found
     * -32602 Invalid params
     * -32603 Internal error
   * レスポンス/エラー/Notification の JSON 生成ヘルパー関数
2. main.cpp — メインプログラム
   * コマンドライン引数の解析（既存の common_params_parse を再利用）
   * llama バックエンドの初期化とモデルの事前ロード（server_context::load_model）
   * server_context::start_loop() をバックグラウンドスレッドで起動
   * メインスレッドで stdin からの行読み取りループ
     * 各行を JSON パースし、JSON-RPC バリデーション
     * メソッドに応じたハンドラにディスパッチ
     * ハンドラは server_context の既存インフラ（server_queue, server_response_reader）を使用
     * 各リクエストの処理は別スレッドで行い、並行リクエストに対応
   * stdout への出力は mutex で保護
   * SIGINT/SIGTERM でグレースフルシャットダウン
3. メソッドハンドラの実装方針
   * 既存の server_routes の handler 実装を参考に、server_context が公開する以下の API を直接使用する:
     * server_response_reader — タスクの投入と結果の読み取り
     * server_queue — タスクキュー
     * oaicompat_chat_params_parse — OpenAI 形式のパラメータ変換
     * server_context_meta — モデルメタ情報
   * server_http_req / server_http_res の HTTP 層は使用せず、JSON-RPC 層で直接 server_context とやりとりする。
4. ビルド設定
   * server-context 静的ライブラリにリンクし、HTTP 関連（cpp-httplib, server-http）には依存しない構成とする。

### エラーマッピング

|状況|JSON-RPC エラーコード|
|----------------|----------------|
|JSON パースエラー| -32700 Parse error|
|JSON-RPC 形式不正| -32600 Invalid Request|
|未知のメソッド| -32601 Method not found|
|パラメータ不正| -32602 Invalid params|
|推論エラー| -32603 Internal error|
|バッチリクエスト| -32600 Invalid Request|


## 作業ステップ

1. JSON-RPC 2.0 プロトコル処理モジュールの作成（jsonrpc.h/cpp）
2. メインエントリポイントの作成（main.cpp）— 引数解析、モデルロード、stdin ループ
3. models メソッドハンドラの実装
4. chat/completions メソッドハンドラの実装（非ストリーミング）
5. ストリーミング対応の実装（Notification 方式）
6. 並行リクエスト対応（スレッド化 + mutex）
7. CMakeLists.txt の作成と親 CMakeLists.txt への登録
8. ビルド確認
