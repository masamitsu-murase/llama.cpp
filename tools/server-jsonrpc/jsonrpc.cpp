#include "jsonrpc.h"

namespace jsonrpc {

namespace {

bool is_valid_id(const json & id) {
    return id.is_null()
        || id.is_string()
        || id.is_number_integer()
        || id.is_number_unsigned()
        || id.is_number_float();
}

json extract_id(const json & obj) {
    if (!obj.is_object() || !obj.contains("id")) {
        return nullptr;
    }

    const json & id = obj.at("id");
    if (!is_valid_id(id)) {
        return nullptr;
    }

    return id;
}

} // namespace

rpc_error::rpc_error(int code, std::string message, json data, json id)
    : code(code),
      message(std::move(message)),
      data(std::move(data)),
      id(std::move(id)),
      what_message(this->message) {
}

const char * rpc_error::what() const noexcept {
    return what_message.c_str();
}

const char * default_message(int code) {
    switch (code) {
        case PARSE_ERROR:      return "Parse error";
        case INVALID_REQUEST:  return "Invalid Request";
        case METHOD_NOT_FOUND: return "Method not found";
        case INVALID_PARAMS:   return "Invalid params";
        case INTERNAL_ERROR:   return "Internal error";
        default:               return "Internal error";
    }
}

request parse_request_line(const std::string & line) {
    json parsed;

    try {
        parsed = json::parse(line);
    } catch (const std::exception & e) {
        throw rpc_error(PARSE_ERROR, default_message(PARSE_ERROR), json {
            {"detail", e.what()},
        });
    }

    if (parsed.is_array()) {
        throw rpc_error(INVALID_REQUEST, default_message(INVALID_REQUEST), json {
            {"detail", "batch requests are not supported"},
        });
    }

    if (!parsed.is_object()) {
        throw rpc_error(INVALID_REQUEST, default_message(INVALID_REQUEST), json {
            {"detail", "request must be a JSON object"},
        });
    }

    const json id = extract_id(parsed);

    if (!parsed.contains("jsonrpc") || !parsed.at("jsonrpc").is_string() || parsed.at("jsonrpc") != "2.0") {
        throw rpc_error(INVALID_REQUEST, default_message(INVALID_REQUEST), json {
            {"detail", "\"jsonrpc\" must be \"2.0\""},
        }, id);
    }

    if (parsed.contains("id") && !is_valid_id(parsed.at("id"))) {
        throw rpc_error(INVALID_REQUEST, default_message(INVALID_REQUEST), json {
            {"detail", "\"id\" must be a string, number, or null"},
        });
    }

    if (!parsed.contains("method") || !parsed.at("method").is_string()) {
        throw rpc_error(INVALID_REQUEST, default_message(INVALID_REQUEST), json {
            {"detail", "\"method\" must be a string"},
        }, id);
    }

    request req;
    req.method = parsed.at("method").get<std::string>();

    if (parsed.contains("id")) {
        req.id = parsed.at("id");
    }

    if (parsed.contains("params")) {
        req.params = parsed.at("params");
    }

    return req;
}

void validate_params_object(const request & req) {
    if (req.params.is_null()) {
        return;
    }

    if (!req.params.is_object()) {
        throw rpc_error(INVALID_PARAMS, default_message(INVALID_PARAMS), json {
            {"detail", "\"params\" must be a JSON object"},
        }, req.id.value_or(nullptr));
    }
}

json make_success(const json & id, json result) {
    return json {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", std::move(result)},
    };
}

json make_error(const json & id, int code, const std::string & message, json data) {
    json error = {
        {"code", code},
        {"message", message},
    };

    if (!data.is_null()) {
        error["data"] = std::move(data);
    }

    return json {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", std::move(error)},
    };
}

json make_notification(const std::string & method, json params) {
    return json {
        {"jsonrpc", "2.0"},
        {"method", method},
        {"params", std::move(params)},
    };
}

} // namespace jsonrpc
