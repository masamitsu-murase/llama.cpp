#include "mcp-client.h"

#include "common.h"

#include <sheredom/subprocess.h>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <thread>

using json = nlohmann::ordered_json;

static std::vector<const char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<const char *> out;
    out.reserve(v.size() + 1);
    for (const auto & s : v) {
        out.push_back(s.c_str());
    }
    out.push_back(nullptr);
    return out;
}

static std::vector<std::string> to_env_vec(const std::map<std::string, std::string> & env) {
    std::vector<std::string> out;
    out.reserve(env.size());
    for (const auto & kv : env) {
        out.push_back(kv.first + "=" + kv.second);
    }
    return out;
}

static std::string trim_copy(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

cli_mcp_client::cli_mcp_client(cli_mcp_server_config cfg) : cfg_(std::move(cfg)) {}

cli_mcp_client::~cli_mcp_client() {
    stop();
}

const std::string & cli_mcp_client::server_name() const {
    return cfg_.name;
}

bool cli_mcp_client::start(std::string & err) {
    if (started_) {
        return true;
    }

    proc_ = std::make_unique<subprocess_s>();

    std::vector<std::string> argv;
    argv.push_back(cfg_.command);
    argv.insert(argv.end(), cfg_.args.begin(), cfg_.args.end());
    const auto cargv = to_cstr_vec(argv);

    int options = subprocess_option_no_window
                | subprocess_option_enable_async
                | subprocess_option_search_user_path;

    const bool inherit_env = cfg_.env.empty();
    std::vector<std::string> env;
    std::vector<const char *> cenv;
    if (inherit_env) {
        options |= subprocess_option_inherit_environment;
    } else {
        env = to_env_vec(cfg_.env);
        cenv = to_cstr_vec(env);
    }

    if (subprocess_create_ex(cargv.data(), options, inherit_env ? nullptr : cenv.data(), proc_.get()) != 0) {
        err = string_format("failed to spawn MCP server '%s'", cfg_.name.c_str());
        proc_.reset();
        return false;
    }

    started_ = true;
    next_request_id_ = 1;
    stdout_buffer_.clear();
    stderr_buffer_.clear();

    if (!send_initialize(err)) {
        stop();
        return false;
    }

    return true;
}

void cli_mcp_client::stop() {
    if (!started_ || !proc_) {
        return;
    }

    send_shutdown();

    if (subprocess_alive(proc_.get())) {
        subprocess_terminate(proc_.get());
    }

    int code = 0;
    subprocess_join(proc_.get(), &code);
    subprocess_destroy(proc_.get());

    proc_.reset();
    started_ = false;
}

bool cli_mcp_client::send_initialize(std::string & err) {
    json init_result;
    const json init_params = {
        {"protocolVersion", "2024-11-05"},
        {"capabilities", json::object()},
        {"clientInfo", {
            {"name", "llama-cli"},
            {"version", "0.0"},
        }},
    };
    if (!send_request("initialize", init_params, cfg_.timeout_seconds, init_result, err)) {
        return false;
    }

    json initialized = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"},
        {"params", json::object()},
    };

    if (!send_json_message(initialized, err)) {
        return false;
    }

    return true;
}

bool cli_mcp_client::send_shutdown() {
    std::string err;
    json result;
    (void) send_request("shutdown", json::object(), 2, result, err);

    json exit_notif = {
        {"jsonrpc", "2.0"},
        {"method", "exit"},
        {"params", json::object()},
    };
    (void) send_json_message(exit_notif, err);
    return true;
}

bool cli_mcp_client::send_json_message(const json & msg, std::string & err) {
    if (!started_ || !proc_) {
        err = "MCP client is not started";
        return false;
    }

    FILE * in = subprocess_stdin(proc_.get());
    if (!in) {
        err = "failed to open MCP stdin pipe";
        return false;
    }

    const std::string body = msg.dump();
    const std::string header = string_format("Content-Length: %zu\r\n\r\n", body.size());

    if (fwrite(header.data(), 1, header.size(), in) != header.size()) {
        err = "failed to write MCP header";
        return false;
    }
    if (fwrite(body.data(), 1, body.size(), in) != body.size()) {
        err = "failed to write MCP body";
        return false;
    }
    fflush(in);

    return true;
}

bool cli_mcp_client::read_json_message(json & out_msg, int timeout_ms, std::string & err) {
    if (!started_ || !proc_) {
        err = "MCP client is not started";
        return false;
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    for (;;) {
        size_t header_end = stdout_buffer_.find("\r\n\r\n");
        size_t delimiter_len = 4;
        if (header_end == std::string::npos) {
            header_end = stdout_buffer_.find("\n\n");
            delimiter_len = 2;
        }

        if (header_end != std::string::npos) {
            const std::string header_blob = stdout_buffer_.substr(0, header_end);
            size_t content_length = 0;
            bool has_content_length = false;

            std::vector<std::string> header_lines = string_split(header_blob, "\n");
            for (auto & line : header_lines) {
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                const std::string key = "Content-Length:";
                if (line.size() >= key.size() && std::equal(key.begin(), key.end(), line.begin(),
                    [](char a, char b) { return std::tolower((unsigned char) a) == std::tolower((unsigned char) b); })) {
                    std::string value = string_strip(line.substr(key.size()));
                    try {
                        content_length = (size_t) std::stoul(value);
                    } catch (const std::exception &) {
                        err = "invalid MCP message: bad Content-Length value";
                        return false;
                    }
                    has_content_length = true;
                    break;
                }
            }

            if (!has_content_length) {
                err = "invalid MCP message: missing Content-Length header";
                return false;
            }

            const size_t body_start = header_end + delimiter_len;
            if (stdout_buffer_.size() >= body_start + content_length) {
                std::string body = stdout_buffer_.substr(body_start, content_length);
                stdout_buffer_.erase(0, body_start + content_length);
                try {
                    out_msg = json::parse(body);
                } catch (const std::exception & e) {
                    err = string_format("failed to parse MCP JSON message: %s", e.what());
                    return false;
                }
                return true;
            }
        }

        char out_buf[4096];
        unsigned n_out = subprocess_read_stdout(proc_.get(), out_buf, sizeof(out_buf));
        if (n_out > 0) {
            stdout_buffer_.append(out_buf, n_out);
        }

        char err_buf[1024];
        unsigned n_err = subprocess_read_stderr(proc_.get(), err_buf, sizeof(err_buf));
        if (n_err > 0) {
            stderr_buffer_.append(err_buf, n_err);
            if (stderr_buffer_.size() > 8192) {
                stderr_buffer_.erase(0, stderr_buffer_.size() - 8192);
            }
        }

        if (n_out == 0 && n_err == 0) {
            if (!subprocess_alive(proc_.get())) {
                err = string_format("MCP server '%s' exited unexpectedly", cfg_.name.c_str());
                const std::string trimmed = trim_copy(stderr_buffer_);
                if (!trimmed.empty()) {
                    err += "\nstderr:\n" + trimmed;
                }
                return false;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                err = string_format("timeout waiting for MCP response from '%s'", cfg_.name.c_str());
                const std::string trimmed = trim_copy(stderr_buffer_);
                if (!trimmed.empty()) {
                    err += "\nstderr:\n" + trimmed;
                }
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

bool cli_mcp_client::send_request(const std::string & method, const json & params, int timeout_seconds, json & out_result, std::string & err) {
    const int req_id = next_request_id_++;

    json req = {
        {"jsonrpc", "2.0"},
        {"id", req_id},
        {"method", method},
        {"params", params},
    };

    if (!send_json_message(req, err)) {
        return false;
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    for (;;) {
        int timeout_ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        if (timeout_ms <= 0) {
            err = string_format("timeout waiting for MCP method '%s'", method.c_str());
            return false;
        }

        json msg;
        if (!read_json_message(msg, timeout_ms, err)) {
            return false;
        }

        if (!msg.is_object()) {
            continue;
        }

        if (!msg.contains("id")) {
            continue;
        }

        const auto & msg_id = msg["id"];
        bool id_match = false;
        if (msg_id.is_number_integer()) {
            id_match = msg_id.get<int>() == req_id;
        } else if (msg_id.is_string()) {
            id_match = msg_id.get<std::string>() == std::to_string(req_id);
        }

        if (!id_match) {
            continue;
        }

        if (msg.contains("error")) {
            err = string_format("MCP method '%s' failed: %s", method.c_str(), msg["error"].dump().c_str());
            return false;
        }

        if (!msg.contains("result")) {
            err = string_format("MCP method '%s' returned response without result", method.c_str());
            return false;
        }

        out_result = msg["result"];
        return true;
    }
}

bool cli_mcp_client::list_tools(std::vector<cli_mcp_tool_def> & out_tools, std::string & err) {
    json result;
    if (!send_request("tools/list", json::object(), cfg_.timeout_seconds, result, err)) {
        return false;
    }

    if (!result.is_object() || !result.contains("tools") || !result["tools"].is_array()) {
        err = string_format("invalid tools/list response from '%s'", cfg_.name.c_str());
        return false;
    }

    out_tools.clear();
    for (const auto & t : result["tools"]) {
        if (!t.is_object() || !t.contains("name") || !t["name"].is_string()) {
            err = string_format("invalid tool entry from '%s'", cfg_.name.c_str());
            return false;
        }

        cli_mcp_tool_def def;
        def.name = t["name"].get<std::string>();
        def.description = t.contains("description") && t["description"].is_string()
            ? t["description"].get<std::string>()
            : "";

        if (t.contains("inputSchema") && t["inputSchema"].is_object()) {
            def.input_schema = t["inputSchema"];
        } else {
            def.input_schema = json::object({{"type", "object"}, {"properties", json::object()}});
        }

        out_tools.push_back(std::move(def));
    }

    return true;
}

bool cli_mcp_client::call_tool(const std::string & tool_name, const json & arguments, int timeout_seconds, json & out_result, std::string & err) {
    json params = {
        {"name", tool_name},
        {"arguments", arguments},
    };
    return send_request("tools/call", params, timeout_seconds, out_result, err);
}
