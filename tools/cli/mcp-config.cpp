#include "mcp-config.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <fstream>
#include <set>
#include <stdexcept>

using json = nlohmann::ordered_json;

static json read_json_file(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(string_format("failed to open MCP config file '%s'", path.c_str()));
    }

    try {
        return json::parse(file);
    } catch (const std::exception & e) {
        throw std::runtime_error(string_format("failed to parse MCP config '%s': %s", path.c_str(), e.what()));
    }
}

static std::vector<std::string> parse_args(const json & jserver, const std::string & server_name) {
    std::vector<std::string> args;
    if (!jserver.contains("args")) {
        return args;
    }
    if (!jserver.at("args").is_array()) {
        throw std::runtime_error(string_format("MCP server '%s': 'args' must be an array", server_name.c_str()));
    }
    for (const auto & item : jserver.at("args")) {
        if (!item.is_string()) {
            throw std::runtime_error(string_format("MCP server '%s': each 'args' entry must be a string", server_name.c_str()));
        }
        args.push_back(item.get<std::string>());
    }
    return args;
}

static std::map<std::string, std::string> parse_env(const json & jserver, const std::string & server_name) {
    std::map<std::string, std::string> env;
    if (!jserver.contains("env")) {
        return env;
    }
    if (!jserver.at("env").is_object()) {
        throw std::runtime_error(string_format("MCP server '%s': 'env' must be an object", server_name.c_str()));
    }
    for (const auto & [key, value] : jserver.at("env").items()) {
        if (!value.is_string()) {
            throw std::runtime_error(string_format("MCP server '%s': env['%s'] must be a string", server_name.c_str(), key.c_str()));
        }
        env[key] = value.get<std::string>();
    }
    return env;
}

cli_mcp_config cli_mcp_config_load_from_file(const std::string & path) {
    json doc = read_json_file(path);
    if (!doc.is_object()) {
        throw std::runtime_error("MCP config root must be a JSON object");
    }
    if (!doc.contains("servers") || !doc.at("servers").is_array()) {
        throw std::runtime_error("MCP config must contain a 'servers' array");
    }

    cli_mcp_config config;
    std::set<std::string> seen_names;

    for (const auto & jserver : doc.at("servers")) {
        if (!jserver.is_object()) {
            throw std::runtime_error("Each MCP server entry must be an object");
        }

        cli_mcp_server_config server;
        if (!jserver.contains("name") || !jserver.at("name").is_string()) {
            throw std::runtime_error("Each MCP server entry must contain string field 'name'");
        }
        if (!jserver.contains("command") || !jserver.at("command").is_string()) {
            throw std::runtime_error(string_format("MCP server '%s' must contain string field 'command'", jserver.at("name").get<std::string>().c_str()));
        }

        server.name = jserver.at("name").get<std::string>();
        server.command = jserver.at("command").get<std::string>();
        server.args = parse_args(jserver, server.name);
        server.env = parse_env(jserver, server.name);
        server.cwd = jserver.value("cwd", std::string());
        server.timeout_seconds = jserver.value("timeout_seconds", 30);

        if (server.name.empty()) {
            throw std::runtime_error("MCP server 'name' must not be empty");
        }
        if (server.command.empty()) {
            throw std::runtime_error(string_format("MCP server '%s': 'command' must not be empty", server.name.c_str()));
        }
        if (server.timeout_seconds < 1) {
            throw std::runtime_error(string_format("MCP server '%s': 'timeout_seconds' must be >= 1", server.name.c_str()));
        }
        if (!seen_names.insert(server.name).second) {
            throw std::runtime_error(string_format("Duplicate MCP server name '%s'", server.name.c_str()));
        }

        config.servers.push_back(std::move(server));
    }

    return config;
}