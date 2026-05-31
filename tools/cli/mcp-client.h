#pragma once

#include "chat.h"
#include "mcp-config.h"

#include <memory>
#include <string>
#include <vector>

class cli_mcp_client {
public:
    explicit cli_mcp_client(cli_mcp_server_config config);
    ~cli_mcp_client();

    cli_mcp_client(const cli_mcp_client &) = delete;
    cli_mcp_client & operator=(const cli_mcp_client &) = delete;

    void start();
    void shutdown();

    std::vector<common_chat_tool> list_tools();
    json call_tool(const std::string & name, const json & arguments);

    const cli_mcp_server_config & config() const;
    const std::string & stderr_log() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};