#pragma once

#include "chat.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct common_chat_tool_call;

struct cli_mcp_tool_result {
    std::string content;
    bool is_error = false;
};

class cli_mcp_registry {
public:
    static std::unique_ptr<cli_mcp_registry> load(const std::string & config_path);

    ~cli_mcp_registry();

    cli_mcp_registry(const cli_mcp_registry &) = delete;
    cli_mcp_registry & operator=(const cli_mcp_registry &) = delete;

    bool empty() const;
    size_t server_count() const;
    const std::vector<common_chat_tool> & tools() const;

    cli_mcp_tool_result invoke(const common_chat_tool_call & tool_call);

private:
    cli_mcp_registry() = default;

    struct route_entry;
    std::vector<std::unique_ptr<class cli_mcp_client>> clients;
    std::vector<common_chat_tool> tool_defs;
    std::map<std::string, route_entry> routes;
};