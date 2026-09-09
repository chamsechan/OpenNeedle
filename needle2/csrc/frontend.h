#ifndef NEEDLE2_FRONTEND_H
#define NEEDLE2_FRONTEND_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* OpenNeedle ABI, not a binary-compatible replacement for official needle.h.
 * Handles are independently owned. UTF-8/JSON input buffers are borrowed only
 * during the call; tokenizer/grammar constructors copy all persistent data.
 * Errors are available on the calling thread until its next failed call.
 */
const char *needle2_frontend_error(void);
void *needle2_tokenizer_create(const unsigned char *blob, size_t size);
void needle2_tokenizer_free(void *tokenizer);
/* dummy: -1 uses metadata, 0 disables, 1 enables the initial dummy prefix.
 * Return required element/byte count, or -1 on error. If capacity is too small,
 * nothing is written. Decoded text is length-delimited, not NUL-terminated.
 */
int needle2_tokenizer_encode(void *tokenizer, const char *text, size_t size, int dummy, int *output,
                             int capacity);
int needle2_tokenizer_decode(void *tokenizer, const int *tokens, int count, char *output,
                             int capacity);
void *needle2_grammar_compile(void *tokenizer, const char *tools_json, size_t size);
void needle2_grammar_free(void *grammar);
struct DFAStateDesc;
const struct DFAStateDesc *needle2_grammar_dfa(void *grammar);
/* output must have room for limit tokens. Engine must have consumed a prompt;
 * logits must be its last full-vocabulary projection. NULL grammar selects
 * unconstrained greedy decoding. Calls on one engine must be serialized.
 */
int needle2_engine_decode_compiled(void *engine, const float *logits, int vocab, int limit,
                                   void *grammar, int *output, int *count);
#ifdef __cplusplus
}
#endif
#endif
