#include "frontend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* Build against the OpenNeedle shared library; pass an extracted tokenizer blob. */
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s tokenizer.bin\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f)
        return 1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    unsigned char *b = malloc(n);
    if (fread(b, 1, n, f) != (size_t)n)
        return 2;
    fclose(f);
    void *t = needle2_tokenizer_create(b, n);
    free(b);
    if (!t)
        return 3;
    int ids[64];
    int count = needle2_tokenizer_encode(t, "hello world", 11, -1, ids, 64);
    char text[256];
    int size = needle2_tokenizer_decode(t, ids, count, text, 256);
    if (size != 11 || memcmp(text, "hello world", 11))
        return 4;
    const char *json =
        "[{\"name\":\"ping\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}]";
    void *g = needle2_grammar_compile(t, json, strlen(json));
    if (!g) {
        puts(needle2_frontend_error());
        return 5;
    }
    needle2_tokenizer_free(t); /* Compiled grammar owns its state tables. */
    needle2_grammar_free(g);
    puts("C ABI tokenizer + grammar passed without Python.");
    return 0;
}
