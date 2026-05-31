#include "jsonrpc.h"

#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <atomic>
#include <clocale>
#include <iostream>
#include <mutex>
#include <signal.h>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

static std::atomic<bool> g_is_terminated{false};
static std::function<void(int)> g_shutdown_handler;

static void signal_handler(int sig) {
    g_is_terminated.store(true);
    if (g_shutdown_handler) {
        g_shutdown_handler(sig);
    }
}

// mutex to protect stdout writes from concurrent request handlers
static std::mutex g_stdout_mutex;

static void write_response(const json & response) {
    std::string line = response.dump(-1, ' ', false, json::error_handler_t::replace);
    std::lock_guard<std::mutex> lock(g_stdout_mutex);
    std::cout << line << std::endl; // endl flushes
}

// Handle "models" method
static json handle_models(const server_context_meta & meta) {
    json model_info = {
        {"id",       meta.model_name},
        {"object",   "model"},
        {"created",  std::time(nullptr)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
        }},
    };

    return json{
        {"object", "list"},
        {"data",   json::array({model_info})},
    };
}

// Handle "chat/completions" method
static void handle_chat_completions(
        const jsonrpc_request & rpc_req,
        server_context & ctx_server,
        const common_params & params) {
    auto meta = ctx_server.get_meta();

    // Parse OpenAI-compatible chat params
    json body = rpc_req.params;
    std::vector<raw_buffer> files;
    json data;
    try {
        data = oaicompat_chat_params_parse(body, meta.chat_params, files);
    } catch (const std::exception & e) {
        write_response(jsonrpc_error(rpc_req.id, JSONRPC_INVALID_PARAMS, e.what()));
        return;
    }

    bool stream = json_value(data, "stream", false);

    // Create response reader
    server_response_reader rd = ctx_server.get_response_reader();
    auto completion_id = gen_chatcmplid();

    try {
        std::vector<server_task> tasks;
        const auto & prompt = data.at("prompt");

        std::vector<server_tokens> inputs =
            tokenize_input_prompts(ctx_server.impl->vocab, ctx_server.impl->mctx, prompt, true, true);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task(SERVER_TASK_TYPE_COMPLETION);
            task.id = rd.get_new_id();
            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                ctx_server.impl->vocab,
                params,
                meta.slot_n_ctx,
                meta.logit_bias_eog,
                data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta.model_name;

            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        write_response(jsonrpc_error(rpc_req.id, JSONRPC_INVALID_PARAMS, e.what()));
        return;
    }

    auto should_stop = []() { return g_is_terminated.load(); };

    if (!stream) {
        // Non-streaming: collect all results and send as single response
        auto all_results = rd.wait_for_all(should_stop);
        if (all_results.is_terminated) {
            return; // shutting down
        }
        if (all_results.error) {
            json err_json = all_results.error->to_json();
            std::string err_msg = "Inference error";
            if (err_json.contains("message") && err_json["message"].is_string()) {
                err_msg = err_json["message"].get<std::string>();
            }
            write_response(jsonrpc_error(rpc_req.id, JSONRPC_INTERNAL_ERROR, err_msg, err_json));
            return;
        }

        json arr = json::array();
        for (auto & res : all_results.results) {
            arr.push_back(res->to_json());
        }

        if (arr.empty()) {
            write_response(jsonrpc_error(rpc_req.id, JSONRPC_INTERNAL_ERROR, "Empty results"));
            return;
        }

        // If single result, return directly; if multiple, merge choices
        json result_json;
        if (arr.size() == 1) {
            result_json = arr[0];
        } else {
            json & choices = arr[0]["choices"];
            for (size_t i = 1; i < arr.size(); i++) {
                choices.push_back(std::move(arr[i]["choices"][0]));
            }
            result_json = arr[0];
        }

        write_response(jsonrpc_result(rpc_req.id, result_json));
    } else {
        // Streaming: send intermediate results as notifications, final as response
        bool first = true;
        json final_result;

        while (true) {
            if (should_stop()) {
                return;
            }

            auto result = rd.next(should_stop);
            if (!result) {
                return; // terminated
            }

            if (result->is_error()) {
                json err_json = result->to_json();
                std::string err_msg = "Inference error";
                if (err_json.contains("message") && err_json["message"].is_string()) {
                    err_msg = err_json["message"].get<std::string>();
                }
                write_response(jsonrpc_error(rpc_req.id, JSONRPC_INTERNAL_ERROR, err_msg, err_json));
                return;
            }

            auto * res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                // Send intermediate chunk as notification
                json chunk = result->to_json();
                write_response(jsonrpc_notification("chat/completions.chunk", {
                    {"request_id", rpc_req.id},
                    {"data",       chunk},
                }));
                continue;
            }

            auto * res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                // Final result — send as response with id
                final_result = result->to_json();
                break;
            }
        }

        write_response(jsonrpc_result(rpc_req.id, final_result));
    }
}

static void handle_request(
        const jsonrpc_request & rpc_req,
        server_context & ctx_server,
        const common_params & params) {
    if (rpc_req.method == "models") {
        auto meta = ctx_server.get_meta();
        json result = handle_models(meta);
        write_response(jsonrpc_result(rpc_req.id, result));
    } else if (rpc_req.method == "chat/completions") {
        handle_chat_completions(rpc_req, ctx_server, params);
    } else {
        write_response(jsonrpc_error(rpc_req.id, JSONRPC_METHOD_NOT_FOUND,
            "Method not found: " + rpc_req.method));
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

#if defined(_WIN32)
    // Set stdin/stdout to binary mode to avoid \r\n issues
    _setmode(_fileno(stdin),  _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    // Redirect logs to stderr (default for LOG_* macros)
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("JSON-RPC 2.0 over stdio gateway starting...\n");

    // Load model
    server_context ctx_server;

    if (params.n_parallel < 0) {
        LOG_INF("n_parallel is set to auto, using n_parallel = 4\n");
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    if (!ctx_server.load_model(params)) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    LOG_INF("Model loaded successfully\n");

    // Start inference loop in background thread
    std::thread inference_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    // Setup signal handlers
    g_shutdown_handler = [&ctx_server](int) {
        ctx_server.terminate();
    };

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sa;
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#elif defined(_WIN32)
    auto ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), TRUE) : FALSE;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(ctrl_handler), TRUE);
#endif

    // Signal readiness
    LOG_INF("Ready. Waiting for JSON-RPC requests on stdin...\n");

    // Main stdin read loop
    std::string line;
    while (!g_is_terminated.load() && std::getline(std::cin, line)) {
        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        // Parse JSON
        json j;
        try {
            j = json::parse(line);
        } catch (const std::exception &) {
            write_response(jsonrpc_error(nullptr, JSONRPC_PARSE_ERROR,
                "Parse error: invalid JSON"));
            continue;
        }

        // Reject batch requests (arrays)
        if (j.is_array()) {
            write_response(jsonrpc_error(nullptr, JSONRPC_INVALID_REQUEST,
                "Invalid Request: batch requests are not supported"));
            continue;
        }

        // Parse JSON-RPC request
        jsonrpc_request rpc_req;
        json parse_error;
        if (!jsonrpc_parse_request(j, rpc_req, parse_error)) {
            write_response(parse_error);
            continue;
        }

        // Notifications (no id) are fire-and-forget; we don't currently handle any
        if (rpc_req.is_notification) {
            // silently ignore notifications
            continue;
        }

        // Dispatch request in a separate thread for concurrency
        std::thread([rpc_req = std::move(rpc_req), &ctx_server, &params]() {
            try {
                handle_request(rpc_req, ctx_server, params);
            } catch (const std::exception & e) {
                write_response(jsonrpc_error(rpc_req.id, JSONRPC_INTERNAL_ERROR, e.what()));
            } catch (...) {
                write_response(jsonrpc_error(rpc_req.id, JSONRPC_INTERNAL_ERROR, "Unknown error"));
            }
        }).detach();
    }

    // Cleanup
    LOG_INF("Shutting down...\n");
    g_is_terminated.store(true);
    ctx_server.terminate();
    inference_thread.join();

    llama_backend_free();
    LOG_INF("Goodbye.\n");

    return 0;
}
