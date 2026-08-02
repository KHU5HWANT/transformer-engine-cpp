/**
 * @file    server/http_server.hpp
 * @brief   Minimal raw POSIX HTTP/1.1 server — no external libraries.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Design rationale — why raw sockets?
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The inference server is intentionally built without cpp-httplib, Boost.Beast,
 * or any other networking library.  Every byte of the HTTP protocol is handled
 * by STL + POSIX APIs (<sys/socket.h>, <netinet/in.h>, <unistd.h>).
 *
 * This proves the engineer understands:
 *   1. The TCP socket lifecycle: socket() → setsockopt() → bind() → listen()
 *                                → accept() → recv()/send() → close()
 *   2. HTTP/1.1 framing: request-line, headers, empty line, body
 *   3. Content-Length body framing for POST requests
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Supported subset of HTTP/1.1
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   Methods:  GET, POST (all others return 405 Method Not Allowed)
 *   Routing:  exact path match registered with add_route()
 *   Body:     read up to Content-Length bytes (required for POST /predict)
 *   Response: HTTP/1.1 with Content-Type, Content-Length, Connection: close
 *   Concurrency: single-threaded (one connection at a time)
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Usage
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   HttpServer srv(8080);
 *   srv.add_route("POST", "/predict", make_predict_handler(model));
 *   srv.serve();   // blocks; Ctrl-C to exit
 *
 * Target: Linux/WSL2, POSIX, C++17, pure STL + sys headers.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>    // std::pair

namespace server {

// ─────────────────────────────────────────────────────────────────────────────
// HttpRequest
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Parsed representation of one incoming HTTP/1.1 request.
 *
 * Populated by HttpServer::parse_request() from the raw socket bytes.
 */
struct HttpRequest {
    std::string method;                            ///< "GET", "POST", ...
    std::string path;                              ///< "/predict"
    std::string http_version;                      ///< "HTTP/1.1"
    std::map<std::string, std::string> headers;    ///< key → value (case-sensitive)
    std::string body;                              ///< request body (empty for GET)
};

// ─────────────────────────────────────────────────────────────────────────────
// HttpResponse
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief A complete HTTP response to be serialised and sent to the client.
 */
struct HttpResponse {
    int         status_code  = 200;
    std::string status_text  = "OK";
    std::string content_type = "application/json";
    std::string body;
};

// ─────────────────────────────────────────────────────────────────────────────
// RouteHandler
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Signature of a route handler.
 *
 * A handler receives a fully-parsed request and returns a complete response.
 * Registered with HttpServer::add_route().
 */
using RouteHandler = std::function<HttpResponse(const HttpRequest&)>;

// ─────────────────────────────────────────────────────────────────────────────
// HttpServer
// ─────────────────────────────────────────────────────────────────────────────

class HttpServer {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Create an HTTP server that will listen on the given port.
     *
     * The server socket is created and bound in the constructor.
     * Call serve() to start accepting connections.
     *
     * @param port  TCP port to bind (default: 8080).
     *
     * @throws std::runtime_error if socket(), setsockopt(), bind(), or listen()
     *         fails.
     */
    explicit HttpServer(uint16_t port = 8080);

    /** Destructor — closes the server socket. */
    ~HttpServer();

    // Non-copyable (owns a raw file descriptor)
    HttpServer(const HttpServer&)            = delete;
    HttpServer& operator=(const HttpServer&) = delete;

    // ── Routing ───────────────────────────────────────────────────────────────

    /**
     * @brief Register a handler for an exact (method, path) pair.
     *
     * @param method   HTTP method string: "GET" or "POST".
     * @param path     Exact URL path: "/predict".
     * @param handler  Callable invoked for matching requests.
     *
     * If two routes share the same (method, path), the last registration wins.
     */
    void add_route(const std::string& method,
                   const std::string& path,
                   RouteHandler        handler);

    // ── Serving ───────────────────────────────────────────────────────────────

    /**
     * @brief Start the blocking accept loop.
     *
     * Accepts one connection at a time (single-threaded), reads the request,
     * dispatches to the registered handler, and writes the response.
     * Returns only when stop() is called or a fatal accept() error occurs.
     */
    void serve();

    /**
     * @brief Signal the serve() loop to stop after the current connection.
     *
     * Safe to call from a signal handler (sets a volatile flag).
     */
    void stop() noexcept;

    // ── Accessors ─────────────────────────────────────────────────────────────

    [[nodiscard]] uint16_t port() const noexcept { return port_; }

private:
    uint16_t port_;
    int      server_fd_ = -1;     ///< Server socket file descriptor
    volatile bool running_ = false;

    // Routes: (method, path) → handler
    std::map<std::pair<std::string, std::string>, RouteHandler> routes_;

    // ── Private helpers ───────────────────────────────────────────────────────

    /**
     * @brief Handle a single accepted client connection.
     * Reads request, dispatches route, writes response, closes fd.
     */
    void handle_connection(int client_fd) const;

    /**
     * @brief Parse raw socket bytes into an HttpRequest.
     *
     * Finds the \r\n\r\n header/body separator, parses request-line and headers,
     * then reads exactly Content-Length bytes as the body.
     */
    [[nodiscard]] static HttpRequest parse_request(const std::string& raw);

    /**
     * @brief Serialise an HttpResponse into a raw HTTP/1.1 byte string.
     *
     * Adds Content-Length and Connection: close headers automatically.
     */
    [[nodiscard]] static std::string build_response(const HttpResponse& resp);
};

}  // namespace server
