/**
 * @file    server/http_server.cpp
 * @brief   Raw POSIX HTTP/1.1 server implementation.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * TCP socket lifecycle (POSIX)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  socket(AF_INET, SOCK_STREAM, 0)
 *    Creates an IPv4 TCP endpoint.  Returns a file descriptor.
 *
 *  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one))
 *    Allows rebinding the port immediately after a previous server exits.
 *    Without this, the OS holds the port in TIME_WAIT for ~60 s — making
 *    repeated server restarts during development extremely frustrating.
 *
 *  bind(fd, &addr, sizeof(addr))
 *    Associates the socket with INADDR_ANY:port (all network interfaces).
 *
 *  listen(fd, backlog=16)
 *    Marks the socket as passive (server).  backlog=16 is the pending-
 *    connection queue depth.  The kernel completes the TCP 3-way handshake
 *    for up to 16 connections before accept() is called.
 *
 *  accept(fd, &client_addr, &client_len)
 *    Dequeues one completed connection from the listen queue.  Returns a
 *    NEW file descriptor representing the per-connection socket.
 *    The original server fd continues listening.
 *
 *  recv(client_fd, buf, len, 0)
 *    Reads up to len bytes from the client.  May return fewer bytes than
 *    requested — HTTP framing (Content-Length) is used to know when the
 *    full request has arrived.
 *
 *  send(client_fd, data, len, 0)
 *    Writes response bytes back to the client.
 *
 *  close(client_fd)
 *    Signals Connection: close (no persistent connections in this server).
 *    The OS sends TCP FIN to the client.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * HTTP/1.1 request framing
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  POST /predict HTTP/1.1\r\n
 *  Host: localhost:8080\r\n
 *  Content-Type: application/json\r\n
 *  Content-Length: 20\r\n
 *  \r\n
 *  {"prompt": "1+1="}
 *
 *  The \r\n\r\n sequence marks the end of headers.  The body is exactly
 *  Content-Length bytes following the separator.  We read in a loop until
 *  we have all of: headers + Content-Length body bytes.
 *
 * Target: Linux/WSL2, POSIX C++17.
 */

#include "server/http_server.hpp"

// ── POSIX networking ──────────────────────────────────────────────────────────
#include <arpa/inet.h>        // htons(), INADDR_ANY
#include <netinet/in.h>       // sockaddr_in
#include <sys/socket.h>       // socket(), bind(), listen(), accept(), setsockopt()
#include <unistd.h>           // close(), recv(), send()

// ── STL ───────────────────────────────────────────────────────────────────────
#include <cerrno>
#include <cstring>            // strerror()
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace server {

// ─────────────────────────────────────────────────────────────────────────────
// Construction / Destruction
// ─────────────────────────────────────────────────────────────────────────────

HttpServer::HttpServer(uint16_t port) : port_(port)
{
    // ── 1. Create TCP socket ──────────────────────────────────────────────────
    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        throw std::runtime_error(
            std::string("HttpServer::socket() failed: ") + std::strerror(errno));
    }

    // ── 2. SO_REUSEADDR — allow immediate port reuse after server restart ──────
    const int one = 1;
    if (::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR,
                     &one, sizeof(one)) < 0) {
        ::close(server_fd_);
        throw std::runtime_error(
            std::string("HttpServer::setsockopt(SO_REUSEADDR) failed: ") +
            std::strerror(errno));
    }

    // ── 3. Bind to all interfaces on the given port ───────────────────────────
    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd_);
        throw std::runtime_error(
            std::string("HttpServer::bind() on port ") + std::to_string(port_) +
            " failed: " + std::strerror(errno) +
            ".  Is the port already in use?  Try: fuser -k " +
            std::to_string(port_) + "/tcp");
    }

    // ── 4. Mark socket as passive (server) ───────────────────────────────────
    constexpr int BACKLOG = 16;   // max pending connections before accept()
    if (::listen(server_fd_, BACKLOG) < 0) {
        ::close(server_fd_);
        throw std::runtime_error(
            std::string("HttpServer::listen() failed: ") + std::strerror(errno));
    }

    std::cout << "[HttpServer] Listening on http://0.0.0.0:" << port_ << "\n";
}

HttpServer::~HttpServer()
{
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Routing
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::add_route(const std::string& method,
                            const std::string& path,
                            RouteHandler        handler)
{
    routes_[{method, path}] = std::move(handler);
    std::cout << "[HttpServer] Registered route: " << method << " " << path << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// serve() — blocking accept loop
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::serve()
{
    running_ = true;
    std::cout << "[HttpServer] Entering accept loop.  Ctrl-C to stop.\n\n";

    while (running_) {
        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);

        const int client_fd = ::accept(
            server_fd_,
            reinterpret_cast<sockaddr*>(&client_addr),
            &client_len);

        if (client_fd < 0) {
            // EINTR fires when Ctrl-C sends SIGINT — not an error
            if (errno == EINTR) { running_ = false; break; }
            if (!running_) break;
            std::cerr << "[HttpServer] accept() error: " << std::strerror(errno) << "\n";
            continue;
        }

        // Convert client IP to human-readable for logging
        char client_ip[INET_ADDRSTRLEN] = {};
        ::inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        std::cout << "[HttpServer] Connection from " << client_ip << "\n";

        handle_connection(client_fd);
        ::close(client_fd);
    }

    std::cout << "[HttpServer] Stopped.\n";
}

void HttpServer::stop() noexcept
{
    running_ = false;
    // Closing the server fd causes the blocking accept() to return EBADF/EINTR
    if (server_fd_ >= 0) {
        ::close(server_fd_);
        server_fd_ = -1;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// handle_connection — read request, dispatch, write response
// ─────────────────────────────────────────────────────────────────────────────

void HttpServer::handle_connection(int client_fd) const
{
    // ── 1. Read until we have the full headers (\r\n\r\n) ────────────────────
    std::string raw;
    raw.reserve(4096);
    char buf[4096];

    while (true) {
        const ssize_t n = ::recv(client_fd, buf, sizeof(buf), 0);
        if (n <= 0) return;    // connection closed or error — nothing to do

        raw.append(buf, static_cast<size_t>(n));

        // Check if we have the full header section
        const auto sep = raw.find("\r\n\r\n");
        if (sep == std::string::npos) continue;

        // ── 2. If POST, read remaining body bytes ─────────────────────────────
        // Scan for Content-Length in the headers block
        size_t content_length = 0;
        const std::string cl_key = "Content-Length: ";
        const auto cl_pos = raw.find(cl_key);
        if (cl_pos != std::string::npos && cl_pos < sep) {
            const auto eol = raw.find("\r\n", cl_pos);
            const std::string cl_val = raw.substr(cl_pos + cl_key.size(),
                                                   eol - cl_pos - cl_key.size());
            content_length = std::stoul(cl_val);
        }

        // Bytes of body already in the buffer
        const size_t body_received = raw.size() - sep - 4;

        // Read remaining body bytes
        while (body_received < content_length &&
               raw.size() - sep - 4 < content_length) {
            const ssize_t m = ::recv(client_fd, buf, sizeof(buf), 0);
            if (m <= 0) break;
            raw.append(buf, static_cast<size_t>(m));
        }

        break;
    }

    if (raw.empty()) return;

    // ── 3. Parse request ──────────────────────────────────────────────────────
    const HttpRequest req = parse_request(raw);

    std::cout << "  → " << req.method << " " << req.path
              << "  body=" << req.body.size() << " bytes\n";

    // ── 4. Dispatch to registered route ───────────────────────────────────────
    HttpResponse resp;

    auto it = routes_.find({req.method, req.path});
    if (it != routes_.end()) {
        try {
            resp = it->second(req);
        } catch (const std::exception& ex) {
            resp.status_code = 500;
            resp.status_text = "Internal Server Error";
            resp.body = std::string(R"({"error": ")") + ex.what() + "\"}";
        }
    } else {
        // Check if path exists but method is wrong
        auto get_it = routes_.find({"GET",  req.path});
        auto post_it= routes_.find({"POST", req.path});
        if (get_it != routes_.end() || post_it != routes_.end()) {
            resp = {405, "Method Not Allowed", "application/json",
                    R"({"error": "method not allowed"})"};
        } else {
            resp = {404, "Not Found", "application/json",
                    R"({"error": "route not found"})"};
        }
    }

    // ── 5. Send response ──────────────────────────────────────────────────────
    const std::string response_str = build_response(resp);
    ::send(client_fd, response_str.c_str(),
           static_cast<int>(response_str.size()), 0);

    std::cout << "  ← " << resp.status_code << " " << resp.status_text
              << "  " << resp.body.size() << " bytes\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_request — HTTP/1.1 request parser
// ─────────────────────────────────────────────────────────────────────────────

HttpRequest HttpServer::parse_request(const std::string& raw)
{
    HttpRequest req;

    // Split at header/body separator
    const std::string SEP = "\r\n\r\n";
    const auto sep_pos = raw.find(SEP);

    const std::string header_section = (sep_pos != std::string::npos)
        ? raw.substr(0, sep_pos)
        : raw;

    std::istringstream ss(header_section);
    std::string line;

    // ── Request-line: METHOD PATH HTTP/1.1 ───────────────────────────────────
    if (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream rl(line);
        rl >> req.method >> req.path >> req.http_version;
    }

    // ── Headers: Key: Value ───────────────────────────────────────────────────
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) break;

        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // Trim leading whitespace from value
            const auto first = val.find_first_not_of(" \t");
            if (first != std::string::npos) val = val.substr(first);
            req.headers[key] = val;
        }
    }

    // ── Body: Content-Length bytes after \r\n\r\n ─────────────────────────────
    if (sep_pos != std::string::npos) {
        req.body = raw.substr(sep_pos + SEP.size());

        // Honour Content-Length header (trim excess bytes from pipelining)
        const auto it = req.headers.find("Content-Length");
        if (it != req.headers.end()) {
            const size_t len = std::stoul(it->second);
            if (req.body.size() > len) req.body.resize(len);
        }
    }

    return req;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_response — HTTP/1.1 response serialiser
// ─────────────────────────────────────────────────────────────────────────────

std::string HttpServer::build_response(const HttpResponse& resp)
{
    std::ostringstream oss;
    oss << "HTTP/1.1 " << resp.status_code << " " << resp.status_text << "\r\n";
    oss << "Content-Type: "   << resp.content_type << "; charset=utf-8\r\n";
    oss << "Content-Length: " << resp.body.size()   << "\r\n";
    oss << "Connection: close\r\n";    // no persistent connections
    oss << "Access-Control-Allow-Origin: *\r\n";   // allow browser fetch()
    oss << "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n";
    oss << "Access-Control-Allow-Headers: Content-Type\r\n";
    oss << "\r\n";                     // end of headers
    oss << resp.body;
    return oss.str();
}

}  // namespace server
