#pragma once

#include <map>
#include <string>
#include <vector>

struct cli_mcp_server_config {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    std::map<std::string, std::string> env;
    int timeout_seconds = 30;
};

struct cli_mcp_config {
    std::vector<cli_mcp_server_config> servers;
};

bool cli_mcp_config_load_file(const std::string & path, cli_mcp_config & out, std::string & err);
