/**
 * @file    server/inference_handler.cpp
 * @brief   /predict handler — JSON parse → tokenise → forward → greedy decode.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Data flow
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  HTTP POST body (string)
 *        │
 *        ▼  extract_prompt()
 *  prompt string (e.g. "1+1=")
 *        │
 *        ▼  char_tokenise()
 *  token_ids: vector<size_t>  (e.g. [49, 43, 49, 61])
 *        │
 *        ▼  model.forward(token_ids, B=1, T=prompt.size())
 *  logits: NodePtr  shape [1, T, 256]
 *        │
 *        ▼  greedy_decode()
 *  predicted_id: size_t  (e.g. 50 = '2')
 *        │
 *        ▼  cast to char
 *  completion: string  (e.g. "2")
 *        │
 *        ▼  build JSON
 *  response body: {"completion": "2"}
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Why argmax at the LAST position?
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The Transformer is trained with causal language modelling: at each position
 * t, the model predicts the token at t+1 given tokens at 0..t.
 *
 * After processing the full prompt [t_0, ..., t_{T-1}]:
 *   logits[0, T-1, :] = distribution over the NEXT token (position T)
 *
 * Taking argmax of this last-position slice gives the greedy prediction for
 * the character immediately following the prompt — exactly what we want.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * JSON extraction — why not a library?
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Adding nlohmann/json would require either a header-only vendored copy or
 * a FetchContent dependency.  For a single field ("prompt") the find/substr
 * approach adds zero dependencies and zero build time.
 *
 * The extraction is deliberately defensive:
 *   - Checks for "prompt" key before accessing the value.
 *   - Validates both opening and closing quotes are present.
 *   - Handles internal whitespace around the colon.
 *
 * For a production service, replace with a proper JSON parser.
 *
 * Target: Linux/WSL2, C++17.
 */

#include "server/inference_handler.hpp"

#include <algorithm> // std::max_element
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cctype>

namespace server {
using engine::NodePtr;
// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Extract the string value of the "prompt" key from a JSON body.
 *
 * Handles: {"prompt": "1+1="}, { "prompt" : "hello world" }, etc.
 * Does NOT handle escaped quotes inside the value (not needed for char-level).
 *
 * @return prompt string on success
 * @throws std::invalid_argument on missing/malformed key
 */
static std::string extract_prompt(const std::string &body) {
  // Step 1: locate the "prompt" key
  const std::string KEY = "\"prompt\"";
  const auto key_pos = body.find(KEY);
  if (key_pos == std::string::npos) {
    throw std::invalid_argument("missing prompt");
  }

  // Step 2: find the colon after the key
  const auto colon_pos = body.find(':', key_pos + KEY.size());
  if (colon_pos == std::string::npos) {
    throw std::invalid_argument("malformed json: no colon after 'prompt'");
  }

  // Step 3: find the opening quote of the value
  const auto open_q = body.find('"', colon_pos + 1);
  if (open_q == std::string::npos) {
    throw std::invalid_argument(
        "malformed json: no opening quote for prompt value");
  }

  // Step 4: find the closing quote of the value (first unescaped `"` after
  // open_q+1)
  std::string prompt;
  bool escaped = false;
  for (size_t i = open_q + 1; i < body.size(); ++i) {
    const char c = body[i];
    if (escaped) {
      // Handle common JSON escape sequences
      switch (c) {
      case '"':
        prompt += '"';
        break;
      case '\\':
        prompt += '\\';
        break;
      case 'n':
        prompt += '\n';
        break;
      case 't':
        prompt += '\t';
        break;
      case 'r':
        prompt += '\r';
        break;
      default:
        prompt += c;
        break; // passthrough for others
      }
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      return prompt; // closing quote found
    } else {
      prompt += c;
    }
  }

  throw std::invalid_argument(
      "malformed json: no closing quote for prompt value");
}

/**
 * @brief Character-level tokenisation: each byte of the string becomes one
 * token.
 *
 * Token ID = raw uint8_t value of the character (0–255).
 * This matches the preprocess.py encoding and the vocab_size=256 model.
 *
 * @throws std::invalid_argument if the string is empty.
 */
static std::vector<size_t> char_tokenise(const std::string &text) {
  if (text.empty()) {
    throw std::invalid_argument("empty prompt");
  }

  std::vector<size_t> ids;
  ids.reserve(text.size());
  for (const unsigned char c : text) {
    ids.push_back(static_cast<size_t>(c));
  }
  return ids;
}

/**
 * @brief Greedy decode: argmax over logit vector at a given pointer.
 *
 * @param logit_ptr  Pointer to the first element of the logit slice.
 * @param vocab_size Number of logit values to scan.
 * @return Index of the maximum logit value.
 */
static size_t greedy_decode(const double *logit_ptr, size_t vocab_size) {
  size_t best = 0;
  double max_val = logit_ptr[0];
  for (size_t v = 1; v < vocab_size; ++v) {
    if (logit_ptr[v] > max_val) {
      max_val = logit_ptr[v];
      best = v;
    }
  }
  return best;
}

static double extract_temperature(const std::string &body) {
  const std::string KEY = "\"temperature\"";
  const auto key_pos = body.find(KEY);
  if (key_pos == std::string::npos) {
    return 0.8; // default
  }
  const auto colon_pos = body.find(':', key_pos + KEY.size());
  if (colon_pos == std::string::npos) {
    return 0.8;
  }
  size_t start = colon_pos + 1;
  while (start < body.size() && std::isspace(body[start])) start++;
  size_t end = start;
  while (end < body.size() && (std::isdigit(body[end]) || body[end] == '.')) end++;
  
  if (start == end) return 0.8;
  try {
    return std::stod(body.substr(start, end - start));
  } catch (...) {
    return 0.8;
  }
}

/**
 * @brief Sample with temperature from logit vector.
 */
static size_t sample_with_temperature(const double *logit_ptr, size_t vocab_size, double temperature) {
  if (temperature <= 0.0) {
    return greedy_decode(logit_ptr, vocab_size);
  }
  
  std::vector<double> probs(vocab_size);
  double max_logit = logit_ptr[0];
  for (size_t v = 1; v < vocab_size; ++v) {
    if (logit_ptr[v] > max_logit) max_logit = logit_ptr[v];
  }
  
  for (size_t v = 0; v < vocab_size; ++v) {
    probs[v] = std::exp((logit_ptr[v] - max_logit) / temperature);
  }
  
  std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
  static thread_local std::mt19937 gen(std::random_device{}());
  
  return dist(gen);
}

/**
 * @brief Build a safe JSON string: escape backslashes and double-quotes.
 *
 * Prevents injection if the completion character happens to be " or \.
 */
static std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 4);
  for (const char c : s) {
    if (c == '"') {
      out += "\\\"";
    } else if (c == '\\') {
      out += "\\\\";
    } else if (c < 0x20) {
      out += ' ';
    } // strip control chars
    else {
      out += c;
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// make_predict_handler
// ─────────────────────────────────────────────────────────────────────────────

RouteHandler make_predict_handler(engine::nn::Transformer &model,
                                  size_t context_len) {
  // Capture model by reference (must outlive the handler)
  return [&model, context_len](const HttpRequest &req) -> HttpResponse {
    // ── 1. Extract prompt from JSON body ──────────────────────────────────
    std::string prompt;
    try {
      prompt = extract_prompt(req.body);
    } catch (const std::invalid_argument &ex) {
      return HttpResponse{400, "Bad Request", "application/json",
                          std::string(R"({"error": ")") + ex.what() + "\"}"};
    }

    if (prompt.empty()) {
      return HttpResponse{400, "Bad Request", "application/json",
                          R"({"error": "empty prompt"})"};
    }

    // ── 2. Context-length guard ───────────────────────────────────────────
    // ── 2. Context-length guard ───────────────────────────────────────────
    if (prompt.size() > context_len) {
      return HttpResponse{400, "Bad Request", "application/json",
                          "{\"error\": \"prompt too long (max " +
                              std::to_string(context_len) + " chars)\"}"};
    }

    // ── 3. Char-level tokenisation ────────────────────────────────────────
    std::vector<size_t> token_ids;
    try {
      token_ids = char_tokenise(prompt);
    } catch (const std::invalid_argument &ex) {
      return HttpResponse{400, "Bad Request", "application/json",
                          std::string(R"({"error": ")") + ex.what() + "\"}"};
    }

    const size_t T = token_ids.size();

    std::cout << "  [inference] prompt=\"" << prompt << "\"  T=" << T << "\n";

    // ── 4. Transformer forward pass ───────────────────────────────────────
    // model.forward(token_ids, batch_size=1, seq_len=T)
    // Returns NodePtr with data shape [1, T, vocab_size]
    NodePtr logits;
    try {
      logits = model.forward(token_ids, /*batch_size=*/1, T);
    } catch (const std::exception &ex) {
      std::cerr << "  [inference] forward() threw: " << ex.what() << "\n";
      return HttpResponse{500, "Internal Server Error", "application/json",
                          std::string(R"({"error": "model forward failed: ")") +
                              ex.what() + "\"}"};
    }

    // ── 5. Greedy decode — argmax at last position ────────────────────────
    // logits->data shape: [1, T, V]
    // Stride: [T*V, V, 1]  → last position slice starts at offset (T-1)*V
    if (logits->data.ndim() != 3) {
      return HttpResponse{500, "Internal Server Error", "application/json",
                          R"({"error": "unexpected logit shape"})"};
    }

    const size_t V = logits->data.shape()[2];
    const double *logit_base = logits->data.data_ptr();
    const double *last_logits = logit_base + (T - 1) * V; // shape [V]

    const double temp = extract_temperature(req.body);
    const size_t predicted_id = sample_with_temperature(last_logits, V, temp);

    // ── 6. Decode token back to character ─────────────────────────────────
    const char predicted_char =
        static_cast<char>(static_cast<unsigned char>(predicted_id));
    const std::string completion(1, predicted_char);

    std::cout << "  [inference] predicted='" << completion
              << "'  (token=" << predicted_id << ")\n";

    // ── 7. Build JSON response ─────────────────────────────────────────────
    const std::string resp_body =
        "{\"completion\": \"" + json_escape(completion) + "\"}";

    return HttpResponse{200, "OK", "application/json", resp_body};
  };
}

} // namespace server
