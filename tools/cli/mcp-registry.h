#pragma once

#include "chat.h"
#include "mcp-client.h"
#include "mcp-config.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct cli_mcp_tool_exec_result {
    std::string content;
    bool is_error = false;
};

class cli_mcp_registry {
public:
    cli_mcp_registry() = default;
    ~cli_mcp_registry() = default;

    bool initialize(const cli_mcp_config & config, int default_timeout_seconds, std::string & err);

    bool empty() const;
    const std::vector<common_chat_tool> & chat_tools() const;

    bool invoke_tool(const common_chat_tool_call & tool_call, int timeout_seconds, cli_mcp_tool_exec_result & out, std::string & err);

private:
    std::vector<std::unique_ptr<cli_mcp_client>> clients_;
    std::vector<common_chat_tool> chat_tools_;
    std::unordered_map<std::string, size_t> tool_to_client_;
};
