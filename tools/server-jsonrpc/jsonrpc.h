#pragma once

#include "server-common.h"

#include <string>

// JSON-RPC 2.0 error codes
enum jsonrpc_error_code {
    JSONRPC_PARSE_ERROR      = -32700,
    JSONRPC_INVALID_REQUEST  = -32600,
    JSONRPC_METHOD_NOT_FOUND = -32601,
    JSONRPC_INVALID_PARAMS   = -32602,
    JSONRPC_INTERNAL_ERROR   = -32603,
};

// Parsed JSON-RPC 2.0 request
struct jsonrpc_request {
    json id;            // can be number, string, or null (for notifications)
    std::string method;
    json params;        // params object (default: empty object)
    bool is_notification = false; // true if no "id" field
};

// Parse a JSON-RPC 2.0 request from a JSON object.
// Returns true on success, false on validation error (error_out is set).
bool jsonrpc_parse_request(const json & j, jsonrpc_request & out, json & error_out);

// Build a JSON-RPC 2.0 success response
json jsonrpc_result(const json & id, const json & result);

// Build a JSON-RPC 2.0 error response
json jsonrpc_error(const json & id, int code, const std::string & message, const json & data = nullptr);

// Build a JSON-RPC 2.0 notification (server-to-client, no id)
json jsonrpc_notification(const std::string & method, const json & params);
