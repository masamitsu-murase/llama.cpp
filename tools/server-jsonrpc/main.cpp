#include "jsonrpc.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include <atomic>
#include <clocale>
#include <ctime>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <signal.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

struct app_state {
    common_params params;
    server_context ctx_server;
    std::unique_ptr<server_context_meta> meta;
    const llama_vocab * vocab = nullptr;
    std::mutex stdout_mutex;
};

static std::atomic<bool> g_is_terminating = false;
static std::atomic_flag g_signal_seen = ATOMIC_FLAG_INIT;
static server_context * g_ctx_server = nullptr;

static bool should_stop() {
    return g_is_terminating.load();
}

static void request_shutdown() {
    if (g_is_terminating.exchange(true)) {
        return;
    }

    if (g_ctx_server != nullptr) {
        g_ctx_server->terminate();
    }
}

static void signal_handler(int) {
    if (g_signal_seen.test_and_set()) {
        std::exit(1);
    }

    request_shutdown();
}

static void write_stdout(app_state & state, const json & payload) {
    std::lock_guard<std::mutex> lock(state.stdout_mutex);
    std::cout << safe_json_to_str(payload) << '\n';
    std::cout.flush();
}

static json build_model_info(const server_context_meta & meta) {
    return json {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(nullptr)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
        }},
    };
}

static json build_models_result(const server_context_meta & meta) {
    return json {
        {"models", {
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""},
                {"type", "model"},
                {"description", ""},
                {"tags", {""}},
                {"capabilities", meta.has_mtmd ? json({"completion", "multimodal"}) : json({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", {""}},
                    {"parameter_size", ""},
                    {"quantization_level", ""},
                }},
            },
        }},
        {"object", "list"},
        {"data", {
            build_model_info(meta),
        }},
    };
}

static json merge_chat_results(std::vector<json> results) {
    std::vector<json> compact;
    compact.reserve(results.size());

    for (auto & result : results) {
        if (!result.is_null()) {
            compact.push_back(std::move(result));
        }
    }

    GGML_ASSERT(!compact.empty());

    if (compact.size() == 1) {
        return std::move(compact.front());
    }

    json merged = std::move(compact.front());
    json & choices = merged["choices"];

    for (size_t i = 1; i < compact.size(); ++i) {
        if (compact[i].contains("choices") && compact[i]["choices"].is_array() && !compact[i]["choices"].empty()) {
            choices.push_back(std::move(compact[i]["choices"][0]));
        }
    }

    return merged;
}

static jsonrpc::rpc_error map_server_error(const json & err) {
    const std::string type = json_value(err, "type", std::string());
    const int code = (type == "invalid_request_error" || type == "exceed_context_size_error")
        ? jsonrpc::INVALID_PARAMS
        : jsonrpc::INTERNAL_ERROR;

    return jsonrpc::rpc_error(code, jsonrpc::default_message(code), err);
}

static void emit_chunk_notifications(app_state & state, const std::optional<json> & request_id, json chunks) {
    if (!chunks.is_array()) {
        chunks = json::array({std::move(chunks)});
    }

    for (auto & chunk : chunks) {
        json params = chunk.is_object()
            ? std::move(chunk)
            : json {{"chunk", std::move(chunk)}};

        params["request_id"] = request_id ? *request_id : json(nullptr);
        write_stdout(state, jsonrpc::make_notification("chat/completions.chunk", std::move(params)));
    }
}

static std::vector<server_task> build_chat_tasks(
        app_state & state,
        server_response_reader & rd,
        const json & data,
        const std::vector<raw_buffer> & files,
        const std::string & completion_id) {
    if (!data.contains("prompt") || !data.at("prompt").is_string()) {
        throw jsonrpc::rpc_error(jsonrpc::INVALID_PARAMS, jsonrpc::default_message(jsonrpc::INVALID_PARAMS), json {
            {"detail", "\"prompt\" must be a string"},
        });
    }

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id         = rd.get_new_id();
    task.id_slot    = json_value(data, "id_slot", -1);
    task.cli        = true;
    task.cli_prompt = data.at("prompt").get<std::string>();
    task.cli_files  = files;
    task.params     = server_task::params_from_json_cmpl(
        state.vocab,
        state.params,
        state.meta->slot_n_ctx,
        state.meta->logit_bias_eog,
        data);
    task.params.res_type          = TASK_RESPONSE_TYPE_OAI_CHAT;
    task.params.oaicompat_cmpl_id = completion_id;
    task.params.oaicompat_model   = state.meta->model_name;

    if (task.params.n_cmpl > 1) {
        for (int i = 0; i < task.params.n_cmpl - 1; ++i) {
            task.add_child(task.id, rd.get_new_id());
        }
    }

    std::vector<server_task> tasks;
    tasks.push_back(std::move(task));
    return tasks;
}

static json handle_chat_completions(const jsonrpc::request & req, app_state & state) {
    if (!req.params.is_null() && !req.params.is_object()) {
        throw jsonrpc::rpc_error(jsonrpc::INVALID_PARAMS, jsonrpc::default_message(jsonrpc::INVALID_PARAMS), json {
            {"detail", "\"params\" must be a JSON object"},
        });
    }

    json body = req.params.is_null() ? json::object() : req.params;
    std::vector<raw_buffer> files;
    json data;

    try {
        data = oaicompat_chat_params_parse(body, state.meta->chat_params, files);
    } catch (const std::exception & e) {
        throw jsonrpc::rpc_error(jsonrpc::INVALID_PARAMS, jsonrpc::default_message(jsonrpc::INVALID_PARAMS), json {
            {"detail", e.what()},
        });
    }

    server_response_reader rd = state.ctx_server.get_response_reader();
    const std::string completion_id = gen_chatcmplid();

    try {
        rd.post_tasks(build_chat_tasks(state, rd, data, files, completion_id));
    } catch (const jsonrpc::rpc_error &) {
        throw;
    } catch (const std::exception & e) {
        throw jsonrpc::rpc_error(jsonrpc::INVALID_PARAMS, jsonrpc::default_message(jsonrpc::INVALID_PARAMS), json {
            {"detail", e.what()},
        });
    }

    const bool stream = json_value(data, "stream", false);

    if (!stream) {
        auto all_results = rd.wait_for_all(should_stop);

        if (all_results.is_terminated) {
            throw jsonrpc::rpc_error(jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), json {
                {"detail", "request interrupted"},
            });
        }

        if (all_results.error) {
            throw map_server_error(all_results.error->to_json());
        }

        std::vector<json> final_results;
        final_results.reserve(all_results.results.size());

        for (auto & result : all_results.results) {
            auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (final == nullptr) {
                throw jsonrpc::rpc_error(jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), json {
                    {"detail", "unexpected non-final completion result"},
                });
            }

            final_results.push_back(result->to_json());
        }

        return merge_chat_results(std::move(final_results));
    }

    std::vector<json> final_results(rd.id_tasks.size(), nullptr);

    while (rd.has_next()) {
        server_task_result_ptr result = rd.next(should_stop);
        if (!result) {
            throw jsonrpc::rpc_error(jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), json {
                {"detail", "request interrupted"},
            });
        }

        if (result->is_error()) {
            throw map_server_error(result->to_json());
        }

        if (dynamic_cast<server_task_result_cmpl_partial *>(result.get()) != nullptr) {
            emit_chunk_notifications(state, req.id, result->to_json());
            continue;
        }

        auto * final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        if (final == nullptr) {
            throw jsonrpc::rpc_error(jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), json {
                {"detail", "unexpected completion result type"},
            });
        }

        final->stream = false;
        final_results[result->index] = final->to_json();
    }

    return merge_chat_results(std::move(final_results));
}

static json handle_models(const jsonrpc::request & req, app_state & state) {
    if (!req.params.is_null() && !req.params.is_object()) {
        throw jsonrpc::rpc_error(jsonrpc::INVALID_PARAMS, jsonrpc::default_message(jsonrpc::INVALID_PARAMS), json {
            {"detail", "\"params\" must be a JSON object"},
        });
    }

    return build_models_result(*state.meta);
}

static json handle_request(const jsonrpc::request & req, app_state & state) {
    if (req.method == "chat/completions") {
        return handle_chat_completions(req, state);
    }

    if (req.method == "models") {
        return handle_models(req, state);
    }

    throw jsonrpc::rpc_error(jsonrpc::METHOD_NOT_FOUND, jsonrpc::default_message(jsonrpc::METHOD_NOT_FOUND), json {
        {"method", req.method},
    });
}

static void process_request(std::shared_ptr<app_state> state, jsonrpc::request req) {
    try {
        json result = handle_request(req, *state);
        if (!req.is_notification()) {
            write_stdout(*state, jsonrpc::make_success(*req.id, std::move(result)));
        }
    } catch (const jsonrpc::rpc_error & e) {
        if (!req.is_notification()) {
            write_stdout(*state, jsonrpc::make_error(*req.id, e.code, e.message, e.data));
        } else {
            SRV_WRN("dropping notification error for method %s: %s\n", req.method.c_str(), e.what());
        }
    } catch (const std::exception & e) {
        if (!req.is_notification()) {
            write_stdout(*state, jsonrpc::make_error(*req.id, jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), json {
                {"detail", e.what()},
            }));
        } else {
            SRV_ERR("notification handler failed for method %s: %s\n", req.method.c_str(), e.what());
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    auto state = std::make_shared<app_state>();

    common_init();

    if (!common_params_parse(argc, argv, state->params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    if (state->params.n_parallel < 0) {
        state->params.n_parallel = 4;
        state->params.kv_unified = true;
    }

    if (state->params.model_alias.empty() && !state->params.model.name.empty()) {
        state->params.model_alias.insert(state->params.model.name);
    }

    llama_backend_init();
    llama_numa_init(state->params.numa);
    common_params_print_info(state->params);

    if (!state->ctx_server.load_model(state->params)) {
        llama_backend_free();
        return 1;
    }

    state->meta = std::make_unique<server_context_meta>(state->ctx_server.get_meta());

    llama_context * ll_ctx = state->ctx_server.get_llama_context();
    if (ll_ctx == nullptr) {
        SRV_ERR("%s", "failed to get llama context after model load\n");
        llama_backend_free();
        return 1;
    }

    state->vocab = llama_model_get_vocab(llama_get_model(ll_ctx));
    if (state->vocab == nullptr) {
        SRV_ERR("%s", "failed to get model vocab\n");
        llama_backend_free();
        return 1;
    }

    g_ctx_server = &state->ctx_server;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, nullptr);
    sigaction(SIGTERM, &sigint_action, nullptr);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT || ctrl_type == CTRL_CLOSE_EVENT) {
            signal_handler(SIGINT);
            return true;
        }
        return false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    std::thread loop_thread([state]() {
        state->ctx_server.start_loop();
    });

    std::vector<std::thread> request_threads;
    std::string line;

    while (!should_stop() && std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        try {
            jsonrpc::request req = jsonrpc::parse_request_line(line);
            request_threads.emplace_back(process_request, state, std::move(req));
        } catch (const jsonrpc::rpc_error & e) {
            write_stdout(*state, jsonrpc::make_error(e.id, e.code, e.message, e.data));
        }
    }

    for (auto & thread : request_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    request_shutdown();

    if (loop_thread.joinable()) {
        loop_thread.join();
    }

    llama_backend_free();
    return 0;
}
