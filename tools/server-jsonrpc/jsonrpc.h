#pragma once

#include "server-common.h"

#include <exception>
#include <optional>
#include <string>

namespace jsonrpc {

constexpr int PARSE_ERROR     = -32700;
constexpr int INVALID_REQUEST = -32600;
constexpr int METHOD_NOT_FOUND = -32601;
constexpr int INVALID_PARAMS  = -32602;
constexpr int INTERNAL_ERROR  = -32603;

struct request {
    std::optional<json> id;
    std::string method;
    json params = json::object();

    bool is_notification() const {
        return !id.has_value();
    }
};

struct rpc_error : public std::exception {
    int code;
    std::string message;
    json data;
    json id;

    rpc_error(int code, std::string message, json data = nullptr, json id = nullptr);

    const char * what() const noexcept override;

private:
    std::string what_message;
};

const char * default_message(int code);

request parse_request_line(const std::string & line);
void validate_params_object(const request & req);

json make_success(const json & id, json result);
json make_error(const json & id, int code, const std::string & message, json data = nullptr);
json make_notification(const std::string & method, json params);

} // namespace jsonrpc
