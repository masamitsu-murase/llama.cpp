#include "jsonrpc.h"

bool jsonrpc_parse_request(const json & j, jsonrpc_request & out, json & error_out) {
    // Must be an object (not array — batch requests are not supported)
    if (!j.is_object()) {
        error_out = jsonrpc_error(nullptr, JSONRPC_INVALID_REQUEST,
            "Invalid Request: expected JSON object, batch requests are not supported");
        return false;
    }

    // "jsonrpc" must be "2.0"
    if (!j.contains("jsonrpc") || !j["jsonrpc"].is_string() || j["jsonrpc"] != "2.0") {
        json id = j.contains("id") ? j["id"] : json(nullptr);
        error_out = jsonrpc_error(id, JSONRPC_INVALID_REQUEST,
            "Invalid Request: missing or invalid \"jsonrpc\" field, must be \"2.0\"");
        return false;
    }

    // "method" must be a string
    if (!j.contains("method") || !j["method"].is_string()) {
        json id = j.contains("id") ? j["id"] : json(nullptr);
        error_out = jsonrpc_error(id, JSONRPC_INVALID_REQUEST,
            "Invalid Request: missing or invalid \"method\" field");
        return false;
    }

    out.method = j["method"].get<std::string>();

    // "params" is optional, defaults to empty object
    if (j.contains("params")) {
        if (!j["params"].is_object() && !j["params"].is_array()) {
            json id = j.contains("id") ? j["id"] : json(nullptr);
            error_out = jsonrpc_error(id, JSONRPC_INVALID_PARAMS,
                "Invalid params: must be an object or array");
            return false;
        }
        out.params = j["params"];
    } else {
        out.params = json::object();
    }

    // "id" determines if this is a request or notification
    if (j.contains("id")) {
        out.id = j["id"];
        out.is_notification = false;
    } else {
        out.id = nullptr;
        out.is_notification = true;
    }

    return true;
}

json jsonrpc_result(const json & id, const json & result) {
    return json{
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"result",  result},
    };
}

json jsonrpc_error(const json & id, int code, const std::string & message, const json & data) {
    json err = {
        {"code",    code},
        {"message", message},
    };
    if (!data.is_null()) {
        err["data"] = data;
    }
    return json{
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"error",   err},
    };
}

json jsonrpc_notification(const std::string & method, const json & params) {
    return json{
        {"jsonrpc", "2.0"},
        {"method",  method},
        {"params",  params},
    };
}
