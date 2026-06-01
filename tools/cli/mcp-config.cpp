#include "mcp-config.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <fstream>
#include <set>

using json = nlohmann::ordered_json;

static bool validate_server(const json & jsv, cli_mcp_server_config & out, std::string & err, size_t idx) {
    const std::string scope = string_format("servers[%zu]", idx);

    if (!jsv.is_object()) {
        err = scope + " must be an object";
        return false;
    }

    if (!jsv.contains("name") || !jsv["name"].is_string() || jsv["name"].get<std::string>().empty()) {
        err = scope + ".name must be a non-empty string";
        return false;
    }
    if (!jsv.contains("command") || !jsv["command"].is_string() || jsv["command"].get<std::string>().empty()) {
        err = scope + ".command must be a non-empty string";
        return false;
    }

    out.name = jsv["name"].get<std::string>();
    out.command = jsv["command"].get<std::string>();

    if (jsv.contains("args")) {
        if (!jsv["args"].is_array()) {
            err = scope + ".args must be an array of strings";
            return false;
        }
        for (const auto & v : jsv["args"]) {
            if (!v.is_string()) {
                err = scope + ".args must be an array of strings";
                return false;
            }
            out.args.push_back(v.get<std::string>());
        }
    }

    if (jsv.contains("env")) {
        if (!jsv["env"].is_object()) {
            err = scope + ".env must be an object of string values";
            return false;
        }
        for (const auto & item : jsv["env"].items()) {
            if (!item.value().is_string()) {
                err = scope + ".env values must be strings";
                return false;
            }
            out.env[item.key()] = item.value().get<std::string>();
        }
    }

    if (jsv.contains("timeout_seconds")) {
        if (!jsv["timeout_seconds"].is_number_integer()) {
            err = scope + ".timeout_seconds must be an integer";
            return false;
        }
        out.timeout_seconds = jsv["timeout_seconds"].get<int>();
        if (out.timeout_seconds < 1) {
            err = scope + ".timeout_seconds must be >= 1";
            return false;
        }
    }

    return true;
}

bool cli_mcp_config_load_file(const std::string & path, cli_mcp_config & out, std::string & err) {
    std::ifstream in(path);
    if (!in) {
        err = string_format("failed to open MCP config: %s", path.c_str());
        return false;
    }

    json root;
    try {
        in >> root;
    } catch (const std::exception & e) {
        err = string_format("invalid MCP config JSON: %s", e.what());
        return false;
    }

    if (!root.is_object()) {
        err = "MCP config root must be an object";
        return false;
    }

    if (!root.contains("servers") || !root["servers"].is_array()) {
        err = "MCP config must contain array field 'servers'";
        return false;
    }

    std::set<std::string> names;
    out.servers.clear();
    for (size_t i = 0; i < root["servers"].size(); ++i) {
        cli_mcp_server_config cfg;
        if (!validate_server(root["servers"][i], cfg, err, i)) {
            return false;
        }
        if (!names.insert(cfg.name).second) {
            err = string_format("duplicate MCP server name: %s", cfg.name.c_str());
            return false;
        }
        out.servers.push_back(std::move(cfg));
    }

    return true;
}
