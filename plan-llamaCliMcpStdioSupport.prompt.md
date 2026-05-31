## Plan: CLI MCP Stdio Support

llama-cli に stdio 限定の MCP client 機能を追加し、JSON config で複数サーバーを起動・初期化して、モデルの tool call を自動実行する。既存の server/chat 基盤は tool-call 生成とパースに再利用し、CLI 側で不足しているのは 1) MCP サーバー設定の読み込み、2) stdio transport と JSON-RPC client、3) tool registry / conflict 検証、4) tool loop 実行である。UI の MCP 実装は構造上の参考にとどめ、C++ 側では最小スコープで tools のみを扱い、prompts/resources/HTTP transports は対象外とする。

**Steps**
1. Phase 1: CLI の既存制御点を MCP 対応の受け皿にする。`common_params` に MCP config ファイル引数を追加し、`llama-cli` 初期化時に config を読み込めるようにする。引数名・ help 生成・デフォルト挙動を CLI 向けに定義する。CLI 専用 JSON schema は `servers[].name`, `command`, `args`, `env`, `cwd`, `timeout_seconds` を最小必須/任意項目として固定する。
2. Phase 1: MCP config パーサーと内部表現を新設する。JSON 構文検証、必須フィールド検証、重複 server 名検証、同名 tool 検出時の起動失敗ポリシーをここで扱う。実装は CLI ローカル型として閉じ、既存 UI 設定型との互換は目指さない。
3. Phase 2: stdio transport と MCP client を新設する。各 server ごとに subprocess を spawn し、stdin/stdout で JSON-RPC 2.0 を送受信する。最低限 `initialize`, `tools/list`, `tools/call`, `shutdown` を実装し、stderr は診断ログ用に別扱いにする。1 server = 1 client の所有関係にして、CLI 終了時と異常時に必ず cleanup する。Phase 1 に依存。
4. Phase 2: MCP tools を `common_chat_tool` へ変換する registry 層を追加する。`tools/list` の結果から name/description/inputSchema を CLI 用定義へ落とし込み、tool 名の一意性を全 server 横断で検証する。ここで `server_name -> client` と `tool_name -> server` のマップを構築する。Step 3 に依存。
5. Phase 3: `llama-cli` の prompt 生成に tool 定義を流し込む。現在 `inputs.tools = {};` と `tool_choice = NONE` になっている経路を、registry が空でなければ tool 定義と `AUTO` を設定する経路へ置き換える。chat template / parser の既存基盤はそのまま再利用する。Step 4 に依存。
6. Phase 3: CLI の generation 後処理を tool loop 対応に拡張する。assistant 応答から `tool_calls` を取得し、tool call がある間は MCP 実行結果を `tool` role message として履歴へ追加し、再度 completion を回す。最終 assistant 応答が返るか、最大 loop 回数・致命エラーに達するまで継続する。単なるテキスト応答は現行動作を維持する。Step 5 に依存。
7. Phase 3: error semantics を定義して CLI に反映する。MCP 起動失敗、初期化失敗、tool 実行失敗、JSON 引数不正は、ユーザーに見える診断とモデルに返す tool result の両方で扱う。推奨は `tool` role message にエラー内容を返してモデルに自己修復させ、recover 不可能な初期化/設定エラーだけ即時終了にする。Step 6 に依存。
8. Phase 4: UX と安全策を詰める。推奨項目は `--mcp-config` 未指定時は完全無効、verbose 時に server 起動/登録 tool 数/実行 tool を表示、tool loop の最大反復回数と 1 call timeout を config or CLI で制御、Ctrl+C 時に subprocess を確実に停止する。Step 6 と parallel で一部可能。
9. Phase 4: documentation と help を更新する。CLI README に config 例、stdio 限定であること、同名 tool で起動失敗すること、未サポート範囲を明記する。実装完了後に help 自動生成物がずれないよう更新手順も合わせて反映する。前段実装完了後。

**Relevant files**
- `c:/work/git_repos/llama_cpp/tools/cli/cli.cpp` — `llama_cli()`, `cli_context`, `generate_completion()`, `format_chat()` を MCP config 初期化・tool 定義注入・tool loop 実行に拡張する主戦場。
- `c:/work/git_repos/llama_cpp/tools/cli/main.cpp` — entry は薄いが、新規初期化エラーの exit code 方針確認に使う。
- `c:/work/git_repos/llama_cpp/common/arg.h` — `common_params` に MCP config 引数と必要なら loop/timeout 設定を追加する。
- `c:/work/git_repos/llama_cpp/common/arg.cpp` — CLI help 生成、引数パース、既存カテゴリへの組み込みを行う。
- `c:/work/git_repos/llama_cpp/common/chat.h` — `common_chat_tool`, `common_chat_msg`, `common_chat_templates_inputs`, `common_chat_parser_params` の既存 tool-call モデルを再利用する参照点。
- `c:/work/git_repos/llama_cpp/tools/server/server-context.h` — CLI が内部利用する server 実行基盤の責務境界を確認する参照先。
- `c:/work/git_repos/llama_cpp/tools/server/server-tools.h` — `get_definition()/invoke()` の tool abstraction を CLI 側 registry / execution interface 設計の参考にする。
- `c:/work/git_repos/llama_cpp/tools/server/server-chat.cpp` — server 側の tool-call 抽出と streaming 差分処理の既存パターン参照先。
- `c:/work/git_repos/llama_cpp/tools/ui/src/lib/services/mcp.service.ts` — `initialize`, `tools/list`, `callTool` の責務分割と transport 抽象の参考実装。
- `c:/work/git_repos/llama_cpp/tools/ui/src/lib/stores/mcp.svelte.ts` — multi-server coordination, conflict 検出, tool-to-server routing の参照先。
- `c:/work/git_repos/llama_cpp/tools/ui/src/lib/types/mcp.d.ts` — MCP config / connection / tool call 型の参考。C++ 実装に直接流用せず、必要項目だけ写経する。
- `c:/work/git_repos/llama_cpp/tools/cli/README.md` — 新オプション、設定例、制約事項を追加する。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-config.h` — 新規。CLI 専用の config 型と parse/validate API。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-config.cpp` — 新規。JSON ファイル読込と validation 実装。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-client.h` — 新規。stdio subprocess client と request/response API。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-client.cpp` — 新規。process lifecycle, JSON-RPC, tools/list, tools/call, shutdown 実装。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-registry.h` — 新規。server 群から tool registry を構築し、tool 名衝突を検出する層。
- `c:/work/git_repos/llama_cpp/tools/cli/mcp-registry.cpp` — 新規。`common_chat_tool` 変換と routing 実装。
- `c:/work/git_repos/llama_cpp/tools/cli/CMakeLists.txt` — 新規 CLI MCP 実装ファイルをビルドへ追加する。

**Verification**
1. `llama-cli --help` で `--mcp-config` と関連説明が出ることを確認する。
2. 無効 config, 必須項目欠落, 重複 server 名, 同名 tool 衝突で、CLI が起動前に明確なエラーを返すことを確認する。
3. 最小 stdio MCP server の fixture を使い、`initialize -> tools/list -> tools/call -> shutdown` が通る単体テストを追加する。
4. tool call を 1 回返すモデル出力 fixture で、CLI が tool 実行結果を message history に戻して 2 周目 generation へ進むことを slice test で確認する。
5. tool 実行失敗時に、recoverable error は tool result message としてモデルに返り、process-level fatal error だけが終了コード非 0 になることを確認する。
6. Windows を含む対象環境で subprocess cleanup が機能し、Ctrl+C や通常終了で MCP child process が残らないことを手動確認する。
7. 既存の MCP 未使用パスで CLI の通常会話モードが回帰しないことを既存/narrow test または手動 smoke で確認する。

**Decisions**
- 自動 tool loop を回し、最終 assistant 応答が出るまで継続する。
- config は CLI 専用 JSON schema を採用し、UI 設定互換はスコープ外とする。
- サポート transport は stdio のみ。HTTP, SSE, WebSocket は対象外。
- 同名 tool は自動 rename せず、起動時エラーにする。
- 初期スコープは MCP tools のみ。MCP prompts/resources, UI proxy, built-in tools との統合拡張は後続とする。

**Further Considerations**
1. subprocess 実装は Windows と POSIX で分岐が必要になるため、まず最低限の同期 I/O で成立させ、非同期最適化は後回しにするのが妥当。
2. loop 暴走防止のため、固定上限（例: 8 回）を初期実装から入れるべき。
3. 将来 `llama-server` と共通化したくなっても、初手から共通ライブラリ化せず CLI ローカル実装で閉じる方が変更面を小さく保てる。