#pragma once

#include "common.h"
#include "mcp-config.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

struct subprocess_s;

struct cli_mcp_tool_def {
    std::string name;
    std::string description;
    nlohmann::ordered_json input_schema;
};

class cli_mcp_client {
public:
    explicit cli_mcp_client(cli_mcp_server_config cfg);
    ~cli_mcp_client();

    const std::string & server_name() const;

    bool start(std::string & err);
    void stop();

    bool list_tools(std::vector<cli_mcp_tool_def> & out_tools, std::string & err);
    bool call_tool(const std::string & tool_name, const nlohmann::ordered_json & arguments, int timeout_seconds, nlohmann::ordered_json & out_result, std::string & err);

private:
    bool send_initialize(std::string & err);
    bool send_shutdown();

    bool send_json_message(const nlohmann::ordered_json & msg, std::string & err);
    bool read_json_message(nlohmann::ordered_json & out_msg, int timeout_ms, std::string & err);
    bool send_request(const std::string & method, const nlohmann::ordered_json & params, int timeout_seconds, nlohmann::ordered_json & out_result, std::string & err);

    cli_mcp_server_config cfg_;

    std::unique_ptr<subprocess_s> proc_;
    bool started_ = false;
    int next_request_id_ = 1;

    std::string stdout_buffer_;
    std::string stderr_buffer_;
};
