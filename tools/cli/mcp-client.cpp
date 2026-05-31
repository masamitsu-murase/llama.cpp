#include "mcp-client.h"

#include "common.h"

#include <sheredom/subprocess.h>

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <chrono>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

constexpr const char * MCP_PROTOCOL_VERSION = "2024-11-05";
constexpr size_t MCP_STDERR_LOG_LIMIT = 64 * 1024;

static std::vector<const char *> to_cstr_vec(const std::vector<std::string> & items) {
    std::vector<const char *> result;
    result.reserve(items.size() + 1);
    for (const auto & item : items) {
        result.push_back(item.c_str());
    }
    result.push_back(nullptr);
    return result;
}

static std::vector<std::string> build_command(const cli_mcp_server_config & config) {
    std::vector<std::string> command;
    command.reserve(config.args.size() + 1);
    command.push_back(config.command);
    command.insert(command.end(), config.args.begin(), config.args.end());
    return command;
}

static std::vector<std::string> build_environment(const std::map<std::string, std::string> & overrides) {
    std::map<std::string, std::string> merged;
#if defined(_WIN32)
    char ** envp = _environ;
#else
    extern char ** environ;
    char ** envp = environ;
#endif
    for (char ** entry = envp; entry && *entry; ++entry) {
        std::string value = *entry;
        const size_t pos = value.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        merged[value.substr(0, pos)] = value.substr(pos + 1);
    }
    for (const auto & [key, value] : overrides) {
        merged[key] = value;
    }

    std::vector<std::string> env;
    env.reserve(merged.size());
    for (const auto & [key, value] : merged) {
        env.push_back(key + "=" + value);
    }
    return env;
}

static std::string to_lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return value;
}

static std::optional<size_t> try_parse_frame(std::string & buffer, json & out) {
    size_t header_end = buffer.find("\r\n\r\n");
    size_t delimiter_len = 4;
    if (header_end == std::string::npos) {
        header_end = buffer.find("\n\n");
        delimiter_len = 2;
    }
    if (header_end == std::string::npos) {
        return std::nullopt;
    }

    size_t content_length = 0;
    std::string headers = buffer.substr(0, header_end);
    std::stringstream ss(headers);
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string key = to_lower_ascii(string_strip(line.substr(0, colon)));
        std::string value = string_strip(line.substr(colon + 1));
        if (key == "content-length") {
            content_length = (size_t) std::stoul(value);
        }
    }
    if (content_length == 0) {
        throw std::runtime_error("MCP stdio frame missing valid Content-Length header");
    }

    const size_t body_pos = header_end + delimiter_len;
    if (buffer.size() < body_pos + content_length) {
        return std::nullopt;
    }

    out = json::parse(buffer.substr(body_pos, content_length));
    buffer.erase(0, body_pos + content_length);
    return content_length;
}

struct current_path_guard {
    current_path_guard(const std::string & cwd, std::mutex & mutex) : lock(mutex) {
        if (cwd.empty()) {
            return;
        }
        std::error_code ec;
        old = fs::current_path(ec);
        if (ec) {
            throw std::runtime_error("failed to get current working directory before spawning MCP server");
        }
        fs::current_path(cwd, ec);
        if (ec) {
            throw std::runtime_error(string_format("failed to change working directory to '%s': %s", cwd.c_str(), ec.message().c_str()));
        }
        changed = true;
    }

    ~current_path_guard() {
        if (!changed) {
            return;
        }
        std::error_code ec;
        fs::current_path(old, ec);
    }

    std::unique_lock<std::mutex> lock;
    fs::path old;
    bool changed = false;
};

static std::mutex g_spawn_mutex;

} // namespace

struct cli_mcp_client::impl {
    explicit impl(cli_mcp_server_config cfg) : config(std::move(cfg)) {}

    cli_mcp_server_config config;
    subprocess_s process{};
    bool started = false;
    bool joined = false;
    int next_id = 1;
    std::string stdout_buffer;
    std::string stderr_buffer;

    void append_stderr(const char * data, size_t len) {
        if (len == 0) {
            return;
        }
        stderr_buffer.append(data, len);
        if (stderr_buffer.size() > MCP_STDERR_LOG_LIMIT) {
            stderr_buffer.erase(0, stderr_buffer.size() - MCP_STDERR_LOG_LIMIT);
        }
    }

    void drain_stderr() {
        char buffer[1024];
        while (true) {
            const unsigned count = subprocess_read_stderr(&process, buffer, sizeof(buffer));
            if (count == 0) {
                break;
            }
            append_stderr(buffer, count);
            if (count < sizeof(buffer)) {
                break;
            }
        }
    }

    void spawn() {
        if (started) {
            return;
        }

        const auto command = build_command(config);
        const auto argv = to_cstr_vec(command);
        const auto env_storage = build_environment(config.env);
        const auto envp = to_cstr_vec(env_storage);

        int options = subprocess_option_enable_async
                    | subprocess_option_no_window
                    | subprocess_option_search_user_path;
        if (config.env.empty()) {
            options |= subprocess_option_inherit_environment;
        }

        current_path_guard cwd_guard(config.cwd, g_spawn_mutex);
        const char * const * env_ptr = config.env.empty() ? nullptr : envp.data();

        if (subprocess_create_ex(argv.data(), options, env_ptr, &process) != 0) {
            throw std::runtime_error(string_format("failed to spawn MCP server '%s' (%s)", config.name.c_str(), config.command.c_str()));
        }
        started = true;
    }

    void write_message(const json & message) {
        FILE * input = subprocess_stdin(&process);
        if (!input) {
            throw std::runtime_error(string_format("MCP server '%s': stdin is unavailable", config.name.c_str()));
        }

        const std::string payload = message.dump();
        const std::string header = string_format("Content-Length: %zu\r\n\r\n", payload.size());
        if (fwrite(header.data(), 1, header.size(), input) != header.size() ||
            fwrite(payload.data(), 1, payload.size(), input) != payload.size() ||
            fflush(input) != 0) {
            throw std::runtime_error(string_format("MCP server '%s': failed to write request", config.name.c_str()));
        }
    }

    json read_one_message(int timeout_seconds) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
        while (true) {
            json message;
            if (try_parse_frame(stdout_buffer, message).has_value()) {
                return message;
            }

            char out_chunk[4096];
            const unsigned out_count = subprocess_read_stdout(&process, out_chunk, sizeof(out_chunk));
            if (out_count > 0) {
                stdout_buffer.append(out_chunk, out_count);
                continue;
            }

            drain_stderr();
            if (!subprocess_alive(&process)) {
                int exit_code = -1;
                if (!joined) {
                    subprocess_join(&process, &exit_code);
                    joined = true;
                }
                throw std::runtime_error(string_format(
                    "MCP server '%s' exited unexpectedly (exit code %d)%s%s",
                    config.name.c_str(),
                    exit_code,
                    stderr_buffer.empty() ? "" : ": ",
                    stderr_buffer.empty() ? "" : stderr_buffer.c_str()));
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error(string_format("MCP server '%s' timed out after %d seconds", config.name.c_str(), timeout_seconds));
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    json request(const std::string & method, json params) {
        const int request_id = next_id++;
        write_message({
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"method", method},
            {"params", std::move(params)},
        });

        while (true) {
            json response = read_one_message(config.timeout_seconds);
            if (response.contains("method") && !response.contains("id")) {
                continue;
            }
            if (!response.contains("id") || response.at("id") != request_id) {
                continue;
            }
            if (response.contains("error")) {
                throw std::runtime_error(string_format(
                    "MCP server '%s' request '%s' failed: %s",
                    config.name.c_str(), method.c_str(), response.at("error").dump().c_str()));
            }
            if (!response.contains("result")) {
                throw std::runtime_error(string_format("MCP server '%s' request '%s' returned no result", config.name.c_str(), method.c_str()));
            }
            return response.at("result");
        }
    }

    void notify(const std::string & method, json params) {
        write_message({
            {"jsonrpc", "2.0"},
            {"method", method},
            {"params", std::move(params)},
        });
    }

    void perform_initialize() {
        json result = request("initialize", {
            {"protocolVersion", MCP_PROTOCOL_VERSION},
            {"capabilities", json::object()},
            {"clientInfo", {
                {"name", "llama-cli"},
                {"version", "0"},
            }},
        });
        (void) result;
        notify("notifications/initialized", json::object());
    }
};

cli_mcp_client::cli_mcp_client(cli_mcp_server_config config) : pimpl(std::make_unique<impl>(std::move(config))) {}

cli_mcp_client::~cli_mcp_client() {
    shutdown();
}

void cli_mcp_client::start() {
    pimpl->spawn();
    pimpl->perform_initialize();
}

void cli_mcp_client::shutdown() {
    if (!pimpl->started) {
        return;
    }

    try {
        if (subprocess_alive(&pimpl->process)) {
            try {
                pimpl->request("shutdown", json::object());
            } catch (...) {
            }
        }
    } catch (...) {
    }

    if (subprocess_alive(&pimpl->process)) {
        subprocess_terminate(&pimpl->process);
    }
    if (!pimpl->joined) {
        int exit_code = -1;
        subprocess_join(&pimpl->process, &exit_code);
        pimpl->joined = true;
    }
    subprocess_destroy(&pimpl->process);
    pimpl->started = false;
}

std::vector<common_chat_tool> cli_mcp_client::list_tools() {
    std::vector<common_chat_tool> tools;
    json params = json::object();

    while (true) {
        json result = pimpl->request("tools/list", params);
        if (!result.contains("tools") || !result.at("tools").is_array()) {
            throw std::runtime_error(string_format("MCP server '%s': tools/list returned invalid payload", pimpl->config.name.c_str()));
        }

        for (const auto & item : result.at("tools")) {
            if (!item.is_object() || !item.contains("name") || !item.at("name").is_string()) {
                throw std::runtime_error(string_format("MCP server '%s': tool entry missing string 'name'", pimpl->config.name.c_str()));
            }

            common_chat_tool tool;
            tool.name = item.at("name").get<std::string>();
            tool.description = item.value("description", std::string());
            if (item.contains("inputSchema")) {
                tool.parameters = item.at("inputSchema").dump();
            } else {
                tool.parameters = json::object().dump();
            }
            tools.push_back(std::move(tool));
        }

        if (!result.contains("nextCursor") || result.at("nextCursor").is_null()) {
            break;
        }
        if (!result.at("nextCursor").is_string()) {
            throw std::runtime_error(string_format("MCP server '%s': nextCursor must be a string", pimpl->config.name.c_str()));
        }
        params = {{"cursor", result.at("nextCursor")}};
    }

    return tools;
}

json cli_mcp_client::call_tool(const std::string & name, const json & arguments) {
    return pimpl->request("tools/call", {
        {"name", name},
        {"arguments", arguments},
    });
}

const cli_mcp_server_config & cli_mcp_client::config() const {
    return pimpl->config;
}

const std::string & cli_mcp_client::stderr_log() const {
    return pimpl->stderr_buffer;
}