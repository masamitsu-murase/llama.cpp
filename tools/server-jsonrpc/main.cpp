#include "jsonrpc.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <csignal>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_is_terminating = false;
std::mutex g_stdout_mutex;

bool should_stop() {
    return g_is_terminating.load();
}

void signal_handler(int) {
    g_is_terminating.store(true);
}

void write_json_line(const json & value) {
    std::lock_guard<std::mutex> lock(g_stdout_mutex);
    std::cout << safe_json_to_str(value) << '\n';
    std::cout.flush();
}

json make_models_result(const server_context_meta & meta) {
    json model_info = {
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

    return {
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
        {"data", { std::move(model_info) }},
    };
}

int map_error_code(const json & err) {
    const int http_code = json_value(err, "code", 500);
    const std::string type = json_value(err, "type", std::string());

    if (http_code == 400 || type == "invalid_request_error" || type == "exceed_context_size_error") {
        return jsonrpc::INVALID_PARAMS;
    }

    return jsonrpc::INTERNAL_ERROR;
}

json aggregate_results(std::vector<server_task_result_ptr> & results, task_response_type res_type) {
    json arr = json::array();
    for (auto & result : results) {
        auto * final_result = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        GGML_ASSERT(final_result != nullptr);
        arr.push_back(final_result->to_json());
    }

    GGML_ASSERT(!arr.empty());
    if (arr.size() == 1) {
        return arr[0];
    }

    if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
        json merged = arr[0];
        json & choices = merged["choices"];
        for (size_t i = 1; i < arr.size(); ++i) {
            choices.push_back(std::move(arr[i]["choices"][0]));
        }
        return merged;
    }

    return arr;
}

std::vector<server_task> build_chat_tasks(
        server_context & ctx_server,
        const common_params & params,
        const server_context_meta & meta,
        const llama_vocab * vocab,
        server_response_reader & rd,
        const json & body) {
    std::vector<server_task> tasks;
    const auto & prompt = body.at("prompt");
    const auto inputs = tokenize_input_prompts(vocab, nullptr, prompt, true, true);

    for (size_t i = 0; i < inputs.size(); ++i) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = rd.get_new_id();
        task.tokens = inputs[i].clone();
        task.params = server_task::params_from_json_cmpl(
            vocab,
            params,
            meta.slot_n_ctx,
            meta.logit_bias_eog,
            body);

        const auto message_spans = json_value(body, "message_spans", json::array());
        if (prompt.is_string() && message_spans.is_array()) {
            task.params.n_before_user = prompt_get_n_before_user(
                message_spans,
                prompt.get<std::string>(),
                {},
                vocab,
                nullptr);
        }

        task.id_slot = json_value(body, "id_slot", -1);
        task.params.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
        task.params.oaicompat_cmpl_id = gen_chatcmplid();
        task.params.oaicompat_model = meta.model_name;

        if (task.params.n_cmpl > 1) {
            const int n_children = task.params.n_cmpl - 1;
            for (int j = 0; j < n_children; ++j) {
                task.add_child(task.id, rd.get_new_id());
            }
        }

        tasks.push_back(std::move(task));
    }

    return tasks;
}

json handle_chat_completions(
        server_context & ctx_server,
        const common_params & params,
        const server_context_meta & meta,
        const llama_vocab * vocab,
        const jsonrpc::request & req) {
    jsonrpc::validate_params_object(req);

    json body = req.params;
    std::vector<raw_buffer> files;
    json parsed = oaicompat_chat_params_parse(body, meta.chat_params, files);

    if (!files.empty()) {
        throw jsonrpc::rpc_error(
            jsonrpc::INVALID_PARAMS,
            jsonrpc::default_message(jsonrpc::INVALID_PARAMS),
            {{"detail", "stdio JSON-RPC does not support binary file attachments"}},
            req.id.value_or(nullptr));
    }

    server_response_reader rd = ctx_server.get_response_reader();
    std::vector<server_task> tasks;

    try {
        tasks = build_chat_tasks(ctx_server, params, meta, vocab, rd, parsed);
        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        throw jsonrpc::rpc_error(
            jsonrpc::INVALID_PARAMS,
            jsonrpc::default_message(jsonrpc::INVALID_PARAMS),
            {{"detail", e.what()}},
            req.id.value_or(nullptr));
    }

    const bool stream = json_value(parsed, "stream", false);

    if (!stream) {
        auto all_results = rd.wait_for_all(should_stop);
        if (all_results.is_terminated) {
            throw jsonrpc::rpc_error(
                jsonrpc::INTERNAL_ERROR,
                jsonrpc::default_message(jsonrpc::INTERNAL_ERROR),
                {{"detail", "request terminated"}},
                req.id.value_or(nullptr));
        }
        if (all_results.error) {
            const json err = all_results.error->to_json();
            throw jsonrpc::rpc_error(
                map_error_code(err),
                json_value(err, "message", std::string(jsonrpc::default_message(jsonrpc::INTERNAL_ERROR))),
                err,
                req.id.value_or(nullptr));
        }
        return aggregate_results(all_results.results, TASK_RESPONSE_TYPE_OAI_CHAT);
    }

    while (rd.has_next()) {
        auto result = rd.next(should_stop);
        if (!result) {
            throw jsonrpc::rpc_error(
                jsonrpc::INTERNAL_ERROR,
                jsonrpc::default_message(jsonrpc::INTERNAL_ERROR),
                {{"detail", "request terminated"}},
                req.id.value_or(nullptr));
        }

        if (result->is_error()) {
            const json err = result->to_json();
            throw jsonrpc::rpc_error(
                map_error_code(err),
                json_value(err, "message", std::string(jsonrpc::default_message(jsonrpc::INTERNAL_ERROR))),
                err,
                req.id.value_or(nullptr));
        }

        if (auto * partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get())) {
            if (req.id.has_value()) {
                json chunk = partial->to_json_oaicompat_chat();
                if (chunk.is_array()) {
                    for (auto & item : chunk) {
                        write_json_line(jsonrpc::make_notification("chat/completions.chunk", {
                            {"request_id", *req.id},
                            {"chunk", std::move(item)},
                        }));
                    }
                } else {
                    write_json_line(jsonrpc::make_notification("chat/completions.chunk", {
                        {"request_id", *req.id},
                        {"chunk", std::move(chunk)},
                    }));
                }
            }
            continue;
        }

        auto * final_result = dynamic_cast<server_task_result_cmpl_final *>(result.get());
        GGML_ASSERT(final_result != nullptr);
        return final_result->to_json_oaicompat_chat();
    }

    throw jsonrpc::rpc_error(
        jsonrpc::INTERNAL_ERROR,
        jsonrpc::default_message(jsonrpc::INTERNAL_ERROR),
        {{"detail", "stream ended without a final result"}},
        req.id.value_or(nullptr));
}

void process_request(
        server_context & ctx_server,
        const common_params & params,
        const server_context_meta & meta,
        const llama_vocab * vocab,
        const std::string & line) {
    bool suppress_response = false;

    try {
        const jsonrpc::request req = jsonrpc::parse_request_line(line);
        suppress_response = req.is_notification();

        json result;
        if (req.method == "models") {
            result = make_models_result(meta);
        } else if (req.method == "chat/completions") {
            result = handle_chat_completions(ctx_server, params, meta, vocab, req);
        } else {
            throw jsonrpc::rpc_error(
                jsonrpc::METHOD_NOT_FOUND,
                jsonrpc::default_message(jsonrpc::METHOD_NOT_FOUND),
                {{"detail", "unsupported method: " + req.method}},
                req.id.value_or(nullptr));
        }

        if (!suppress_response) {
            write_json_line(jsonrpc::make_success(*req.id, std::move(result)));
        }
    } catch (const jsonrpc::rpc_error & e) {
        if (!suppress_response || e.code == jsonrpc::PARSE_ERROR || e.code == jsonrpc::INVALID_REQUEST) {
            write_json_line(jsonrpc::make_error(e.id, e.code, e.message, e.data));
        }
    } catch (const std::exception & e) {
        if (!suppress_response) {
            write_json_line(jsonrpc::make_error(nullptr, jsonrpc::INTERNAL_ERROR, jsonrpc::default_message(jsonrpc::INTERNAL_ERROR), {
                {"detail", e.what()},
            }));
        }
    }
}

} // namespace

int llama_server_jsonrpc(int argc, char ** argv);

int llama_server_jsonrpc(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    if (params.model.path.empty()) {
        LOG_ERR("%s", "a model path is required for llama-server-jsonrpc\n");
        return 1;
    }

    if (params.n_parallel < 0) {
        params.n_parallel = 4;
        params.kv_unified = true;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    common_params_print_info(params, true);

    server_context ctx_server;
    if (!ctx_server.load_model(params)) {
        LOG_ERR("%s", "failed to load model\n");
        llama_backend_free();
        return 1;
    }

    const server_context_meta meta = ctx_server.get_meta();
    const llama_context * ctx = ctx_server.get_llama_context();
    GGML_ASSERT(ctx != nullptr);
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    std::signal(SIGINT, signal_handler);
#if defined(SIGTERM)
    std::signal(SIGTERM, signal_handler);
#endif

    std::thread inference_thread([&ctx_server]() {
        ctx_server.start_loop();
    });

    std::vector<std::thread> workers;
    std::string line;
    while (!should_stop() && std::getline(std::cin, line)) {
        workers.emplace_back([&ctx_server, &params, &meta, vocab, line]() {
            process_request(ctx_server, params, meta, vocab, line);
        });
    }

    g_is_terminating.store(true);
    ctx_server.terminate();

    for (auto & worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    if (inference_thread.joinable()) {
        inference_thread.join();
    }

    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    return llama_server_jsonrpc(argc, argv);
}#include "jsonrpc.h"

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
