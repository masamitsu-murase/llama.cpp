#include "common.h"

#include "mcp-registry.h"
#include "server-common.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

static std::string tool_result_to_content(const json & result) {
    std::string text;

    if (result.contains("content") && result["content"].is_array()) {
        for (const auto & part : result["content"]) {
            if (part.is_object() && json_value(part, "type", std::string("")) == "text") {
                std::string t = json_value(part, "text", std::string(""));
                if (!t.empty()) {
                    if (!text.empty()) {
                        text += "\n";
                    }
                    text += t;
                }
            }
        }
    }

    if (!text.empty()) {
        return text;
    }

    return result.dump();
}

bool cli_mcp_registry::initialize(const cli_mcp_config & config, int default_timeout_seconds, std::string & err) {
    clients_.clear();
    chat_tools_.clear();
    tool_to_client_.clear();

    for (const auto & server : config.servers) {
        auto client = std::make_unique<cli_mcp_client>(server);
        if (!client->start(err)) {
            err = string_format("failed to initialize MCP server '%s': %s", server.name.c_str(), err.c_str());
            return false;
        }

        std::vector<cli_mcp_tool_def> tools;
        if (!client->list_tools(tools, err)) {
            err = string_format("failed to list tools from MCP server '%s': %s", server.name.c_str(), err.c_str());
            return false;
        }

        const size_t client_index = clients_.size();

        for (const auto & tool : tools) {
            if (tool.name.empty()) {
                err = string_format("MCP server '%s' returned a tool with empty name", server.name.c_str());
                return false;
            }
            if (tool_to_client_.find(tool.name) != tool_to_client_.end()) {
                err = string_format("duplicate MCP tool name '%s' detected across servers", tool.name.c_str());
                return false;
            }

            common_chat_tool chat_tool;
            chat_tool.name = tool.name;
            chat_tool.description = tool.description;
            chat_tool.parameters = tool.input_schema.dump();

            tool_to_client_[tool.name] = client_index;
            chat_tools_.push_back(std::move(chat_tool));
        }

        clients_.push_back(std::move(client));
    }

    if (default_timeout_seconds > 0) {
        (void) default_timeout_seconds;
    }

    return true;
}

bool cli_mcp_registry::empty() const {
    return chat_tools_.empty();
}

const std::vector<common_chat_tool> & cli_mcp_registry::chat_tools() const {
    return chat_tools_;
}

bool cli_mcp_registry::invoke_tool(const common_chat_tool_call & tool_call, int timeout_seconds, cli_mcp_tool_exec_result & out, std::string & err) {
    auto it = tool_to_client_.find(tool_call.name);
    if (it == tool_to_client_.end()) {
        err = string_format("unknown MCP tool: %s", tool_call.name.c_str());
        out.content = string_format("error: %s", err.c_str());
        out.is_error = true;
        return true;
    }

    json args;
    try {
        if (tool_call.arguments.empty()) {
            args = json::object();
        } else {
            args = json::parse(tool_call.arguments);
        }
    } catch (const std::exception & e) {
        out.content = string_format("error: invalid tool arguments for '%s': %s", tool_call.name.c_str(), e.what());
        out.is_error = true;
        return true;
    }

    json result;
    std::string call_err;
    if (!clients_[it->second]->call_tool(tool_call.name, args, timeout_seconds, result, call_err)) {
        out.content = string_format("error: MCP tool '%s' failed: %s", tool_call.name.c_str(), call_err.c_str());
        out.is_error = true;
        return true;
    }

    out.content = tool_result_to_content(result);
    out.is_error = json_value(result, "isError", false);
    return true;
}
