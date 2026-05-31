#include "mcp-registry.h"

#include "mcp-client.h"
#include "mcp-config.h"

#include "common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <map>
#include <stdexcept>

using json = nlohmann::ordered_json;

struct cli_mcp_registry::route_entry {
    cli_mcp_client * client = nullptr;
    std::string server_name;
};

static json parse_tool_arguments(const common_chat_tool_call & tool_call) {
    if (tool_call.arguments.empty()) {
        return json::object();
    }

    try {
        json parsed = json::parse(tool_call.arguments);
        if (!parsed.is_object()) {
            throw std::runtime_error("tool arguments must be a JSON object");
        }
        return parsed;
    } catch (const std::exception & e) {
        throw std::runtime_error(string_format("tool '%s' arguments are invalid JSON: %s", tool_call.name.c_str(), e.what()));
    }
}

std::unique_ptr<cli_mcp_registry> cli_mcp_registry::load(const std::string & config_path) {
    auto registry = std::unique_ptr<cli_mcp_registry>(new cli_mcp_registry());
    cli_mcp_config config = cli_mcp_config_load_from_file(config_path);

    for (const auto & server : config.servers) {
        auto client = std::make_unique<cli_mcp_client>(server);
        client->start();

        auto server_tools = client->list_tools();
        for (const auto & tool : server_tools) {
            if (registry->routes.find(tool.name) != registry->routes.end()) {
                const auto & existing = registry->routes.at(tool.name);
                throw std::runtime_error(string_format(
                    "Duplicate MCP tool name '%s' from servers '%s' and '%s'",
                    tool.name.c_str(), existing.server_name.c_str(), server.name.c_str()));
            }
            registry->routes.emplace(tool.name, route_entry{client.get(), server.name});
            registry->tool_defs.push_back(tool);
        }

        registry->clients.push_back(std::move(client));
    }

    return registry;
}

cli_mcp_registry::~cli_mcp_registry() = default;

bool cli_mcp_registry::empty() const {
    return tool_defs.empty();
}

size_t cli_mcp_registry::server_count() const {
    return clients.size();
}

const std::vector<common_chat_tool> & cli_mcp_registry::tools() const {
    return tool_defs;
}

cli_mcp_tool_result cli_mcp_registry::invoke(const common_chat_tool_call & tool_call) {
    auto it = routes.find(tool_call.name);
    if (it == routes.end()) {
        return {
            json({
                {"is_error", true},
                {"error", string_format("unknown MCP tool '%s'", tool_call.name.c_str())},
            }).dump(),
            true,
        };
    }

    try {
        json result = it->second.client->call_tool(tool_call.name, parse_tool_arguments(tool_call));
        json wrapped = {
            {"server", it->second.server_name},
            {"tool", tool_call.name},
            {"result", result},
        };
        bool is_error = result.value("isError", false);
        return { wrapped.dump(), is_error };
    } catch (const std::exception & e) {
        return {
            json({
                {"server", it->second.server_name},
                {"tool", tool_call.name},
                {"is_error", true},
                {"error", e.what()},
            }).dump(),
            true,
        };
    }
}