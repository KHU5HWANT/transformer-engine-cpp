/**
 * @file    server/inference_handler.hpp
 * @brief   HTTP /predict route handler — tokenize → forward → greedy decode.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * API contract
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  POST /predict
 *  Content-Type: application/json
 *
 *  Request body:
 *    {"prompt": "1+1="}
 *
 *  Response body (HTTP 200):
 *    {"completion": "2"}
 *
 *  Error responses:
 *    HTTP 400 {"error": "missing prompt"}    — no "prompt" key in JSON
 *    HTTP 400 {"error": "empty prompt"}      — prompt is ""
 *    HTTP 400 {"error": "prompt too long"}   — > model context_len tokens
 *    HTTP 500 {"error": "..."}               — exception from forward pass
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Tokenisation
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The model is trained character-by-character on a vocabulary of 256
 * byte values (vocab_size=256, matching preprocess.py).  Tokenisation is
 * therefore trivially:
 *
 *   token_id[i] = static_cast<uint8_t>(prompt[i])   (0–255)
 *
 * No BPE, no sentencepiece — just raw byte values.
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Greedy decoding
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Given logits[1, T, V] from the Transformer forward pass:
 *   1. Take the logit slice at the LAST time step: logits[0, T-1, 0..V-1]
 *   2. Find argmax → predicted token ID (0–255)
 *   3. Cast to char → one completion character
 *
 * Only one token is predicted per request (extend to a generation loop
 * by appending the prediction to the prompt and calling again).
 *
 * ════════════════════════════════════════════════════════════════════════════
 * Naive JSON extraction (no external library)
 * ════════════════════════════════════════════════════════════════════════════
 *
 * We avoid nlohmann/json or rapidjson to keep the project self-contained.
 * The extraction strategy:
 *   1. Find `"prompt"` substring in the request body.
 *   2. Scan forward to the `:` separator.
 *   3. Find the opening `"` of the value.
 *   4. Find the closing `"` (unescaped) of the value.
 *   5. substr() the content between them.
 *
 * This is intentionally fragile for whitespace variations but completely
 * correct for well-formed {"prompt": "..."} payloads from curl or any
 * standard JSON serialiser.
 *
 * Target: Linux/WSL2, C++17.
 */

#pragma once

#include "server/http_server.hpp"
#include "nn/transformer.hpp"

namespace server {

/**
 * @brief Construct the /predict RouteHandler bound to the given Transformer.
 *
 * The model is captured by reference — it must outlive the handler.
 * The handler is const-safe (forward pass does not modify model weights).
 *
 * @param model      Trained (or partially-trained) Transformer instance.
 * @param context_len Maximum sequence length the model supports.
 *
 * @return RouteHandler suitable for HttpServer::add_route("POST", "/predict", ...).
 */
[[nodiscard]] RouteHandler make_predict_handler(
    engine::nn::Transformer& model,
    size_t                   context_len = 128);

}  // namespace server
