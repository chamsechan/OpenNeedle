// Independent native text/schema frontend. Included after engine.cpp by cq.cpp.
#include <array>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>

namespace frontend {
static int utf8_next(int s, unsigned char b) {
    if (!s) {
        if (b < 128)
            return 0;
        if (b >= 0xc2 && b <= 0xdf)
            return 1;
        if (b == 0xe0)
            return 4;
        if (b == 0xed)
            return 5;
        if (b >= 0xe1 && b <= 0xef)
            return 2;
        if (b == 0xf0)
            return 6;
        if (b == 0xf4)
            return 7;
        if (b >= 0xf1 && b <= 0xf3)
            return 3;
    } else if (s <= 3) {
        if (b >= 0x80 && b <= 0xbf)
            return s - 1;
    } else if (s == 4) {
        if (b >= 0xa0 && b <= 0xbf)
            return 1;
    } else if (s == 5) {
        if (b >= 0x80 && b <= 0x9f)
            return 1;
    } else if (s == 6) {
        if (b >= 0x90 && b <= 0xbf)
            return 2;
    } else if (s == 7) {
        if (b >= 0x80 && b <= 0x8f)
            return 2;
    }
    return -1;
}
static void valid_utf8(const std::string &s) {
    int state = 0;
    for (unsigned char b : s) {
        state = utf8_next(state, b);
        if (state < 0)
            throw std::runtime_error("invalid UTF-8");
    }
    if (state)
        throw std::runtime_error("incomplete UTF-8");
}
static std::string replace_all(std::string s, const std::string &a, const std::string &b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) {
        s.replace(p, a.size(), b);
        p += b.size();
    }
    return s;
}
static void append_utf8(std::string &s, unsigned c) {
    if (c < 128)
        s += char(c);
    else if (c < 2048) {
        s += char(0xc0 | (c >> 6));
        s += char(0x80 | (c & 63));
    } else if (c < 65536) {
        s += char(0xe0 | (c >> 12));
        s += char(0x80 | ((c >> 6) & 63));
        s += char(0x80 | (c & 63));
    } else {
        s += char(0xf0 | (c >> 18));
        s += char(0x80 | ((c >> 12) & 63));
        s += char(0x80 | ((c >> 6) & 63));
        s += char(0x80 | (c & 63));
    }
}
struct Json {
    char kind = 'n';
    std::string text;
    std::vector<Json> array;
    std::vector<std::pair<std::string, Json>> object;
    const Json *get(const std::string &k) const {
        for (auto &v : object)
            if (v.first == k)
                return &v.second;
        return nullptr;
    }
    const Json &at(const std::string &k) const {
        auto *p = get(k);
        if (!p)
            throw std::runtime_error("missing JSON field: " + k);
        return *p;
    }
    std::string str() const {
        if (kind != 's')
            throw std::runtime_error("expected JSON string");
        return text;
    }
    bool integer() const { return kind == 'd' && text.find_first_of(".eE") == std::string::npos; }
};
struct Parser {
    const std::string &s;
    size_t p = 0;
    void ws() {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\n' || s[p] == '\r' || s[p] == '\t'))
            ++p;
    }
    char take() {
        if (p == s.size())
            throw std::runtime_error("truncated JSON");
        return s[p++];
    }
    void expect(char c) {
        if (take() != c)
            throw std::runtime_error("invalid JSON syntax");
    }
    unsigned hex4() {
        unsigned v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = take();
            int n = c >= '0' && c <= '9'   ? c - '0'
                    : c >= 'a' && c <= 'f' ? c - 'a' + 10
                    : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                           : -1;
            if (n < 0)
                throw std::runtime_error("invalid Unicode escape");
            v = v * 16 + n;
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string r;
        for (;;) {
            unsigned char c = take();
            if (c == '"')
                return r;
            if (c < 32)
                throw std::runtime_error("control byte in JSON string");
            if (c != '\\') {
                r += char(c);
                continue;
            }
            c = take();
            switch (c) {
            case '"':
            case '\\':
            case '/':
                r += char(c);
                break;
            case 'b':
                r += '\b';
                break;
            case 'f':
                r += '\f';
                break;
            case 'n':
                r += '\n';
                break;
            case 'r':
                r += '\r';
                break;
            case 't':
                r += '\t';
                break;
            case 'u': {
                unsigned v = hex4();
                if (v >= 0xd800 && v <= 0xdbff) {
                    expect('\\');
                    expect('u');
                    unsigned lo = hex4();
                    if (lo < 0xdc00 || lo > 0xdfff)
                        throw std::runtime_error("invalid surrogate pair");
                    v = 0x10000 + ((v - 0xd800) << 10) + lo - 0xdc00;
                } else if (v >= 0xdc00 && v <= 0xdfff)
                    throw std::runtime_error("unpaired surrogate");
                append_utf8(r, v);
                break;
            }
            default:
                throw std::runtime_error("invalid JSON escape");
            }
        }
    }
    Json value(int depth = 0) {
        if (depth > 64)
            throw std::runtime_error("JSON nesting limit");
        ws();
        if (p == s.size())
            throw std::runtime_error("missing JSON value");
        Json r;
        char c = s[p];
        if (c == '"') {
            r.kind = 's';
            r.text = string();
        } else if (c == '{' || c == '[') {
            ++p;
            r.kind = c;
            ws();
            char close = c == '{' ? '}' : ']';
            if (p < s.size() && s[p] == close) {
                ++p;
                return r;
            }
            for (;;) {
                ws();
                if (c == '{') {
                    std::string k = string();
                    if (r.get(k))
                        throw std::runtime_error("duplicate JSON key");
                    ws();
                    expect(':');
                    r.object.push_back({k, value(depth + 1)});
                } else
                    r.array.push_back(value(depth + 1));
                ws();
                char next = take();
                if (next == close)
                    break;
                if (next != ',')
                    throw std::runtime_error("invalid JSON separator");
            }
        } else if (c == 't' || c == 'f' || c == 'n') {
            std::string word = c == 't' ? "true" : c == 'f' ? "false" : "null";
            for (char b : word)
                expect(b);
            r.kind = c;
            r.text = word;
        } else {
            r.kind = 'd';
            size_t a = p;
            if (s[p] == '-')
                ++p;
            if (p == s.size())
                throw std::runtime_error("invalid number");
            if (s[p] == '0')
                ++p;
            else {
                if (s[p] < '1' || s[p] > '9')
                    throw std::runtime_error("invalid number");
                while (p < s.size() && s[p] >= '0' && s[p] <= '9')
                    ++p;
            }
            if (p < s.size() && s[p] == '.') {
                ++p;
                size_t a = p;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9')
                    ++p;
                if (p == a)
                    throw std::runtime_error("invalid fraction");
            }
            if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
                ++p;
                if (p < s.size() && (s[p] == '+' || s[p] == '-'))
                    ++p;
                size_t a = p;
                while (p < s.size() && s[p] >= '0' && s[p] <= '9')
                    ++p;
                if (p == a)
                    throw std::runtime_error("invalid exponent");
            }
            r.text = s.substr(a, p - a);
        }
        return r;
    }
    Json parse() {
        valid_utf8(s);
        Json r = value();
        ws();
        if (p != s.size())
            throw std::runtime_error("trailing JSON data");
        return r;
    }
};
static std::string quote(const std::string &s) {
    std::string r = "\"";
    const char *hex = "0123456789abcdef";
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            r += "\\\"";
            break;
        case '\\':
            r += "\\\\";
            break;
        case '\b':
            r += "\\b";
            break;
        case '\f':
            r += "\\f";
            break;
        case '\n':
            r += "\\n";
            break;
        case '\r':
            r += "\\r";
            break;
        case '\t':
            r += "\\t";
            break;
        default:
            if (c < 32) {
                r += "\\u00";
                r += hex[c >> 4];
                r += hex[c & 15];
            } else
                r += char(c);
        }
    }
    return r + '"';
}
static std::string literal(const Json &j) {
    if (j.kind == 's')
        return quote(j.text);
    if (j.kind == 'd' || j.kind == 't' || j.kind == 'f')
        return j.text;
    if (j.kind == 'n')
        return "null";
    throw std::runtime_error("only scalar enums are supported");
}
struct Tokenizer {
    std::vector<std::string> pieces;
    std::vector<float> scores;
    std::vector<int> types, markers;
    std::unordered_map<std::string, int> ids;
    std::array<int, 256> bytes;
    bool dummy = false, fallback = false;
    int unk = 0;
    Tokenizer(const unsigned char *data, size_t n) {
        if (!data || n < 24)
            throw std::runtime_error("truncated tokenizer");
        size_t off = 0;
        auto read = [&](int k) {
            if (off + size_t(k) > n)
                throw std::runtime_error("truncated tokenizer record");
            uint32_t v = 0;
            for (int i = 0; i < k; ++i)
                v |= uint32_t(data[off++]) << (8 * i);
            return v;
        };
        uint32_t count = read(4);
        read(4);
        read(4);
        read(4);
        unk = int(read(4));
        dummy = read(1) != 0;
        fallback = read(1) != 0;
        read(2);
        if (count > 1000000 || unk < 0 || uint32_t(unk) >= count)
            throw std::runtime_error("invalid tokenizer geometry");
        bytes.fill(-1);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t bits = read(4);
            float score;
            std::memcpy(&score, &bits, 4);
            if (std::isnan(score))
                throw std::runtime_error("NaN tokenizer score");
            int type = read(1);
            size_t len = read(2);
            if (off + len > n || type > 4)
                throw std::runtime_error("invalid tokenizer record");
            std::string piece(reinterpret_cast<const char *>(data + off), len);
            off += len;
            valid_utf8(piece);
            pieces.push_back(piece);
            scores.push_back(score);
            types.push_back(type);
            ids[piece] = int(i);
            if (type == 3) {
                if (piece.empty())
                    throw std::runtime_error("empty special token");
                markers.push_back(i);
            }
            if (type == 4) {
                if (piece.size() != 6 || piece.substr(0, 3) != "<0x" || piece[5] != '>')
                    throw std::runtime_error("invalid byte token");
                size_t used = 0;
                int b = std::stoi(piece.substr(3, 2), &used, 16);
                if (used != 2 || b < 0 || b > 255)
                    throw std::runtime_error("invalid byte token");
                bytes[b] = i;
            }
        }
        auto chars = [&](int i) {
            size_t c = 0;
            for (unsigned char b : pieces[i])
                if ((b & 0xc0) != 0x80)
                    ++c;
            return c;
        };
        std::stable_sort(markers.begin(), markers.end(),
                         [&](int a, int b) { return chars(a) > chars(b); });
    }
    int id(const std::string &s, int def) const {
        auto it = ids.find(s);
        return it == ids.end() ? def : it->second;
    }
    void bpe(const std::string &s, std::vector<int> &out) const {
        struct Node {
            std::string s;
            int prev, next, version = 0;
            bool live = true;
        };
        std::vector<Node> nodes;
        for (size_t i = 0; i < s.size();) {
            size_t end = i + 1;
            while (end < s.size() && (uint8_t(s[end]) & 0xc0) == 0x80)
                ++end;
            int j = nodes.size();
            nodes.push_back({s.substr(i, end - i), j - 1, j + 1});
            i = end;
        }
        if (nodes.empty())
            return;
        nodes.back().next = -1;
        struct Merge {
            float score;
            int left, right, lv, rv;
            bool operator<(const Merge &o) const {
                return score != o.score ? score < o.score : left > o.left;
            }
        };
        std::priority_queue<Merge> heap;
        auto offer = [&](int l) {
            if (l < 0 || !nodes[l].live || nodes[l].next < 0)
                return;
            int r = nodes[l].next;
            auto it = ids.find(nodes[l].s + nodes[r].s);
            if (it != ids.end())
                heap.push({scores[it->second], l, r, nodes[l].version, nodes[r].version});
        };
        for (size_t i = 0; i < nodes.size(); ++i)
            offer(i);
        while (!heap.empty()) {
            auto m = heap.top();
            heap.pop();
            auto &l = nodes[m.left];
            auto &r = nodes[m.right];
            if (!l.live || !r.live || l.next != m.right || l.version != m.lv || r.version != m.rv)
                continue;
            l.s += r.s;
            l.next = r.next;
            ++l.version;
            r.live = false;
            if (l.next >= 0)
                nodes[l.next].prev = m.left;
            offer(l.prev);
            offer(m.left);
        }
        for (int i = 0; i >= 0; i = nodes[i].next) {
            auto it = ids.find(nodes[i].s);
            if (it != ids.end())
                out.push_back(it->second);
            else if (fallback) {
                for (unsigned char b : nodes[i].s) {
                    if (bytes[b] < 0)
                        throw std::runtime_error("missing byte fallback token");
                    out.push_back(bytes[b]);
                }
            } else
                out.push_back(unk);
        }
    }
    std::vector<int> encode(const std::string &text, int add_dummy = -1) const {
        valid_utf8(text);
        std::vector<int> out;
        if (text.empty())
            return out;
        std::string s = replace_all(text, " ", "▁");
        if (add_dummy < 0 ? dummy : add_dummy != 0)
            s = "▁" + s;
        size_t start = 0, i = 0;
        while (i < s.size()) {
            int found = -1;
            for (int m : markers)
                if (s.compare(i, pieces[m].size(), pieces[m]) == 0) {
                    found = m;
                    break;
                }
            if (found >= 0) {
                bpe(s.substr(start, i - start), out);
                out.push_back(found);
                i += pieces[found].size();
                start = i;
            } else
                ++i;
        }
        bpe(s.substr(start), out);
        return out;
    }
    std::string decode(const int *tokens, int count) const {
        std::string raw;
        for (int i = 0; i < count; ++i) {
            int t = tokens[i];
            if (t < 0 || size_t(t) >= pieces.size())
                throw std::runtime_error("token outside tokenizer vocabulary");
            if (types[t] == 4)
                raw += char(std::stoi(pieces[t].substr(3, 2), nullptr, 16));
            else if (types[t] != 1 && types[t] != 2)
                raw += pieces[t];
        }
        std::string s;
        for (size_t i = 0; i < raw.size();) {
            size_t j = i;
            int state = utf8_next(0, uint8_t(raw[j++]));
            if (state < 0) {
                s += "�";
                i = j;
                continue;
            }
            while (state > 0 && j < raw.size()) {
                int next = utf8_next(state, uint8_t(raw[j]));
                if (next < 0)
                    break;
                state = next;
                ++j;
            }
            if (state == 0)
                s += raw.substr(i, j - i);
            else
                s += "�";
            i = j;
        }
        s = replace_all(s, "▁", " ");
        if (dummy && !s.empty() && s[0] == ' ')
            s.erase(0, 1);
        return s;
    }
};
using Part = std::pair<int, int>;
struct NFA {
    struct Edge {
        std::array<uint64_t, 4> mask{};
        int dest;
    };
    struct Node {
        std::vector<Edge> edges;
        std::vector<int> eps;
    };
    std::vector<Node> nodes;
    int state() {
        if (nodes.size() >= 20000)
            throw std::runtime_error("grammar NFA exceeds 20000 states");
        nodes.emplace_back();
        return nodes.size() - 1;
    }
    Part chars(const std::string &s) {
        int a = state(), b = state();
        Edge e;
        e.dest = b;
        for (unsigned char c : s)
            e.mask[c / 64] |= uint64_t(1) << (c % 64);
        nodes[a].edges.push_back(e);
        return {a, b};
    }
    Part seq(const std::vector<Part> &v) {
        if (v.empty()) {
            int a = state();
            return {a, a};
        }
        for (size_t i = 1; i < v.size(); ++i)
            nodes[v[i - 1].second].eps.push_back(v[i].first);
        return {v.front().first, v.back().second};
    }
    Part lit(const std::string &s) {
        std::vector<Part> v;
        for (char c : s)
            v.push_back(chars(std::string(1, c)));
        return seq(v);
    }
    Part alt(const std::vector<Part> &v) {
        int a = state(), b = state();
        for (auto p : v) {
            nodes[a].eps.push_back(p.first);
            nodes[p.second].eps.push_back(b);
        }
        return {a, b};
    }
    Part opt(Part p) {
        auto r = alt({p});
        nodes[r.first].eps.push_back(r.second);
        return r;
    }
    Part star(Part p) {
        auto r = opt(p);
        nodes[p.second].eps.push_back(p.first);
        return r;
    }
    Part ws() { return star(chars(" \t\n\r")); }
    Part integer() {
        return seq(
            {opt(lit("-")), alt({lit("0"), seq({chars("123456789"), star(chars("0123456789"))})})});
    }
    Part string() {
        std::string body;
        for (int c = 32; c < 256; ++c)
            if (c != 34 && c != 92)
                body += char(c);
        std::vector<Part> hex{lit("u")};
        for (int i = 0; i < 4; ++i)
            hex.push_back(chars("0123456789abcdefABCDEF"));
        auto escaped = seq({lit("\\"), alt({chars("\"\\/bfnrt"), seq(hex)})});
        return seq({lit("\""), star(alt({chars(body), escaped})), lit("\"")});
    }
    static void fields(const Json &j, const std::set<std::string> &allowed) {
        if (j.kind != '{')
            throw std::runtime_error("schema must be an object");
        for (auto &kv : j.object)
            if (!allowed.count(kv.first))
                throw std::runtime_error("unsupported schema keyword: " + kv.first);
    }
    Part schema(const Json &s, int depth = 0) {
        if (depth > 32)
            throw std::runtime_error("schema nesting exceeds 32");
        if (auto *alternatives = s.get("anyOf")) {
            fields(s, {"anyOf", "title", "description", "default", "examples", "$comment",
                       "deprecated", "readOnly", "writeOnly"});
            if (alternatives->kind != '[' || alternatives->array.empty())
                throw std::runtime_error("anyOf must be a nonempty array");
            std::vector<Part> branches;
            for (auto &branch : alternatives->array)
                branches.push_back(schema(branch, depth + 1));
            return alt(branches);
        }
        if (auto *kinds = s.get("type"))
            if (kinds->kind == '[') {
                if (kinds->array.empty())
                    throw std::runtime_error("type union must be nonempty");
                std::vector<Part> branches;
                std::set<std::string> seen;
                for (auto &k : kinds->array) {
                    auto name = k.str();
                    if (!seen.insert(name).second)
                        throw std::runtime_error("duplicate union type");
                    Json branch = s;
                    for (auto &kv : branch.object)
                        if (kv.first == "type")
                            kv.second = k;
                    branches.push_back(schema(branch, depth + 1));
                }
                return alt(branches);
            }
        std::string kind = s.get("type") ? s.at("type").str() : "";
        std::set<std::string> allowed = {"title",    "description", "default",  "examples",
                                         "$comment", "deprecated",  "readOnly", "writeOnly",
                                         "type",     "enum"};
        if (kind == "object")
            for (auto k : {"properties", "required", "additionalProperties"})
                allowed.insert(k);
        if (kind == "array")
            for (auto k : {"items", "minItems", "maxItems"})
                allowed.insert(k);
        fields(s, allowed);
        if (auto *v = s.get("enum")) {
            if (v->kind != '[' || v->array.empty())
                throw std::runtime_error("enum must be a nonempty array");
            std::vector<Part> parts;
            for (auto &value : v->array) {
                bool ok = kind.empty() || (kind == "string" && value.kind == 's') ||
                          (kind == "boolean" && (value.kind == 't' || value.kind == 'f')) ||
                          (kind == "null" && value.kind == 'n') ||
                          (kind == "integer" && value.integer()) ||
                          (kind == "number" && value.kind == 'd');
                if (!ok)
                    throw std::runtime_error("enum values do not match type");
                if (value.kind == 'd' && !value.integer() && !std::isfinite(std::stod(value.text)))
                    throw std::runtime_error("nonfinite enum");
                parts.push_back(lit(literal(value)));
            }
            return alt(parts);
        }
        if (kind == "string")
            return string();
        if (kind == "integer")
            return integer();
        if (kind == "number") {
            auto digits = [&]() { return seq({chars("0123456789"), star(chars("0123456789"))}); };
            return seq({integer(), opt(seq({lit("."), digits()})),
                        opt(seq({chars("eE"), opt(chars("+-")), digits()}))});
        }
        if (kind == "boolean")
            return alt({lit("true"), lit("false")});
        if (kind == "null")
            return lit("null");
        if (kind == "object") {
            Json empty;
            empty.kind = '{';
            const Json &props = s.get("properties") ? s.at("properties") : empty;
            if (props.kind != '{')
                throw std::runtime_error("properties must be an object");
            std::set<std::string> required;
            if (auto *r = s.get("required")) {
                if (r->kind != '[')
                    throw std::runtime_error("required must be an array");
                for (auto &v : r->array) {
                    auto k = v.str();
                    if (!props.get(k) || !required.insert(k).second)
                        throw std::runtime_error("required must name distinct properties");
                }
            }
            if (auto *a = s.get("additionalProperties"))
                if (a->kind != 'f')
                    throw std::runtime_error("additionalProperties must be false");
            auto begin = lit("{");
            int vacant = begin.second, nonempty = -1;
            for (auto &kv : props.object) {
                auto field = seq({ws(), lit(quote(kv.first)), ws(), lit(":"), ws(),
                                  schema(kv.second, depth + 1)});
                if (vacant >= 0)
                    nodes[vacant].eps.push_back(field.first);
                if (nonempty >= 0) {
                    auto sep = seq({ws(), lit(",")});
                    nodes[nonempty].eps.push_back(sep.first);
                    nodes[sep.second].eps.push_back(field.first);
                }
                if (required.count(kv.first)) {
                    vacant = -1;
                    nonempty = field.second;
                } else {
                    int join = state();
                    nodes[field.second].eps.push_back(join);
                    if (nonempty >= 0)
                        nodes[nonempty].eps.push_back(join);
                    nonempty = join;
                }
            }
            auto close = seq({ws(), lit("}")});
            for (int p : {vacant, nonempty})
                if (p >= 0)
                    nodes[p].eps.push_back(close.first);
            return {begin.first, close.second};
        }
        if (kind == "array") {
            auto &item = s.at("items"); // Validate item even for an empty-only array.
            auto first = schema(item, depth + 1);
            auto bound = [&](const char *k, int def) {
                auto *v = s.get(k);
                if (!v)
                    return def;
                if (!v->integer())
                    throw std::runtime_error("array bound must be integer");
                long long n = std::stoll(v->text);
                if (n < 0 || n > 1024)
                    throw std::runtime_error("array bounds must be in 0..1024");
                return int(n);
            };
            int minimum = bound("minItems", 0), maximum = bound("maxItems", -1);
            if (maximum >= 0 && maximum < minimum)
                throw std::runtime_error("invalid array bounds");
            auto begin = seq({lit("["), ws()}), close = seq({ws(), lit("]")});
            if (!minimum)
                nodes[begin.second].eps.push_back(close.first);
            if (!maximum)
                return {begin.first, close.second};
            nodes[begin.second].eps.push_back(first.first);
            int cursor = first.second, count = maximum < 0 ? std::max(1, minimum) : maximum;
            for (int n = 1; n <= count; ++n) {
                if (n >= minimum)
                    nodes[cursor].eps.push_back(close.first);
                if (n < count || maximum < 0) {
                    auto more = seq({ws(), lit(","), ws(), schema(item, depth + 1)});
                    nodes[cursor].eps.push_back(more.first);
                    if (n == count)
                        nodes[more.second].eps.push_back(cursor);
                    else
                        cursor = more.second;
                }
            }
            return {begin.first, close.second};
        }
        throw std::runtime_error("explicit supported type or scalar enum required");
    }
    Part tools(const Json &tools) {
        if (tools.kind != '[')
            throw std::runtime_error("tools must be an array");
        std::vector<Part> calls;
        std::set<std::string> names;
        for (auto &original : tools.array) {
            const Json *t = &original;
            if (auto *type = t->get("type")) {
                if (type->kind == 's' && type->text == "function") {
                    fields(*t, {"type", "function"});
                    t = &t->at("function");
                }
            }
            fields(*t, {"name", "description", "parameters", "strict"});
            std::string name = t->at("name").str();
            if (name.empty() || !names.insert(name).second)
                throw std::runtime_error("tool names must be distinct and nonempty");
            Json def = Parser{std::string("{\"type\":\"object\",\"properties\":{}}")}.parse();
            const Json &params = t->get("parameters") ? t->at("parameters") : def;
            if (params.at("type").str() != "object")
                throw std::runtime_error("tool parameters must be object schema");
            auto args = schema(params);
            calls.push_back(seq({lit("{"), ws(), lit("\"name\""), ws(), lit(":"), ws(),
                                 lit(quote(name)), ws(), lit(","), ws(), lit("\"arguments\""), ws(),
                                 lit(":"), ws(), args, ws(), lit("}")}));
        }
        auto begin = seq({ws(), lit("["), ws()}), close = seq({ws(), lit("]"), ws()});
        nodes[begin.second].eps.push_back(close.first);
        if (!calls.empty()) {
            auto call = alt(calls), sep = seq({ws(), lit(","), ws()});
            nodes[begin.second].eps.push_back(call.first);
            nodes[call.second].eps.push_back(close.first);
            nodes[call.second].eps.push_back(sep.first);
            nodes[sep.second].eps.push_back(call.first);
        }
        return {begin.first, close.second};
    }
    std::vector<int> closure(const std::vector<int> &v) const {
        std::set<int> seen(v.begin(), v.end());
        std::vector<int> stack(v);
        while (!stack.empty()) {
            int i = stack.back();
            stack.pop_back();
            for (int d : nodes[i].eps)
                if (seen.insert(d).second)
                    stack.push_back(d);
        }
        return {seen.begin(), seen.end()};
    }
};
struct Grammar {
    std::vector<int> types, fallback, offsets, tokens, next;
    DFAStateDesc desc{};
    int vocab;
    Grammar(const Json &tools, const Tokenizer &tok) : vocab(tok.pieces.size()) {
        NFA nfa;
        auto body = nfa.tools(tools);
        struct Trie {
            std::map<unsigned char, int> children;
            std::vector<int> leaves;
        };
        std::vector<Trie> trie(1);
        for (size_t t = 0; t < tok.pieces.size(); ++t) {
            std::string bytes;
            if (tok.types[t] == 4)
                bytes += char(std::stoi(tok.pieces[t].substr(3, 2), nullptr, 16));
            else if (tok.types[t] == 0)
                bytes = replace_all(tok.pieces[t], "▁", " ");
            if (bytes.empty())
                continue;
            int node = 0;
            for (unsigned char b : bytes) {
                auto it = trie[node].children.find(b);
                if (it == trie[node].children.end()) {
                    int child = trie.size();
                    trie[node].children[b] = child;
                    trie.emplace_back();
                    node = child;
                } else
                    node = it->second;
            }
            trie[node].leaves.push_back(t);
        }
        using Key = std::pair<std::vector<int>, int>;
        struct ByteState {
            Key key;
            std::array<int, 256> move;
            ByteState(Key k) : key(std::move(k)) { move.fill(-2); }
        };
        Key initial = {nfa.closure({body.first}), 0};
        std::vector<ByteState> states;
        states.emplace_back(initial);
        std::map<Key, int> index{{initial, 0}};
        auto move = [&](int s, unsigned char b) {
            int cached = states[s].move[b];
            if (cached != -2)
                return cached;
            int utf = utf8_next(states[s].key.second, b);
            std::vector<int> targets;
            if (utf >= 0)
                for (int i : states[s].key.first)
                    for (auto &e : nfa.nodes[i].edges)
                        if ((e.mask[b / 64] >> (b % 64)) & 1)
                            targets.push_back(e.dest);
            int dst = -1;
            if (!targets.empty()) {
                Key key = {nfa.closure(targets), utf};
                auto it = index.find(key);
                if (it == index.end()) {
                    if (states.size() >= 20000)
                        throw std::runtime_error("grammar byte DFA exceeds 20000 states");
                    dst = states.size();
                    index.emplace(key, dst);
                    states.emplace_back(std::move(key));
                } else
                    dst = it->second;
            }
            states[s].move[b] = dst;
            return dst;
        };
        int start_id = tok.id("<tool_call>", 10), end_id = tok.id("</tool_call>", 11);
        std::vector<std::map<int, int>> transitions(3);
        transitions[0][start_id] = 2;
        std::map<int, int> token_index{{0, 2}};
        std::queue<int> pending;
        pending.push(0);
        size_t edges = 1;
        while (!pending.empty()) {
            int bs = pending.front();
            pending.pop();
            std::map<int, int> candidates;
            auto &key = states[bs].key;
            if (key.second == 0 &&
                std::binary_search(key.first.begin(), key.first.end(), body.second))
                candidates[end_id] = 1;
            std::vector<std::pair<int, int>> stack{{0, bs}};
            while (!stack.empty()) {
                auto entry = stack.back();
                stack.pop_back();
                int node = entry.first, s = entry.second;
                for (int token : trie[node].leaves) {
                    auto it = token_index.find(s);
                    int ti;
                    if (it == token_index.end()) {
                        if (token_index.size() >= 4096)
                            throw std::runtime_error("grammar token DFA exceeds 4096 states");
                        ti = transitions.size();
                        token_index[s] = ti;
                        transitions.emplace_back();
                        pending.push(s);
                    } else
                        ti = it->second;
                    candidates[token] = ti;
                }
                for (auto &e : trie[node].children) {
                    int dst = move(s, e.first);
                    if (dst >= 0)
                        stack.push_back({e.second, dst});
                }
            }
            edges += candidates.size();
            if (edges > 4000000)
                throw std::runtime_error("grammar DFA exceeds 4000000 transitions");
            transitions[token_index[bs]] = std::move(candidates);
        }
        std::vector<std::vector<int>> reverse(transitions.size());
        for (size_t i = 0; i < transitions.size(); ++i)
            for (auto &e : transitions[i])
                reverse[e.second].push_back(i);
        std::vector<bool> live(transitions.size());
        live[1] = true;
        pending.push(1);
        while (!pending.empty()) {
            int s = pending.front();
            pending.pop();
            for (int src : reverse[s])
                if (!live[src]) {
                    live[src] = true;
                    pending.push(src);
                }
        }
        if (!live[2])
            throw std::runtime_error("tokenizer cannot represent a complete tool call");
        live[0] = true;
        std::vector<int> remap(live.size(), -1);
        int count = 0;
        for (size_t i = 0; i < live.size(); ++i)
            if (live[i])
                remap[i] = count++;
        offsets.push_back(0);
        for (size_t i = 0; i < live.size(); ++i)
            if (live[i]) {
                types.push_back(i == 0 ? 0 : i == 1 ? 4 : 1);
                fallback.push_back(i == 0 ? 0 : -1);
                for (auto &e : transitions[i])
                    if (live[e.second]) {
                        tokens.push_back(e.first);
                        next.push_back(remap[e.second]);
                    }
                offsets.push_back(tokens.size());
            }
        desc = {count,
                0,
                tok.id("</s>", 1),
                tok.id("<stop>", 5),
                start_id,
                end_id,
                types.data(),
                fallback.data(),
                offsets.data(),
                tokens.data(),
                next.data()};
        for (int t : {desc.eos_id, desc.stop_id, start_id, end_id})
            if (t < 0 || t >= vocab)
                throw std::runtime_error("grammar special token outside vocabulary");
    }
};
} // namespace frontend

static thread_local std::string frontend_error;
extern "C" {
const char *needle2_frontend_error() { return frontend_error.c_str(); }
void *needle2_tokenizer_create(const unsigned char *blob, size_t size) {
    try {
        return new frontend::Tokenizer(blob, size);
    } catch (const std::exception &e) {
        frontend_error = e.what();
        return nullptr;
    }
}
void needle2_tokenizer_free(void *p) { delete static_cast<frontend::Tokenizer *>(p); }
// Return required capacity when the output is too small; never truncate.
int needle2_tokenizer_encode(void *p, const char *text, size_t size, int dummy, int *out,
                             int capacity) {
    try {
        if (!p || (!text && size) || capacity < 0)
            throw std::runtime_error("invalid tokenizer arguments");
        auto ids = static_cast<frontend::Tokenizer *>(p)->encode(
            std::string(text ? text : "", size), dummy);
        if (ids.size() > INT32_MAX)
            throw std::runtime_error("too many tokens");
        if (out && size_t(capacity) >= ids.size())
            std::copy(ids.begin(), ids.end(), out);
        return ids.size();
    } catch (const std::exception &e) {
        frontend_error = e.what();
        return -1;
    }
}
int needle2_tokenizer_decode(void *p, const int *ids, int count, char *out, int capacity) {
    try {
        if (!p || count < 0 || (!ids && count) || capacity < 0)
            throw std::runtime_error("invalid tokenizer arguments");
        auto text = static_cast<frontend::Tokenizer *>(p)->decode(ids, count);
        if (text.size() > INT32_MAX)
            throw std::runtime_error("decoded text too large");
        if (out && size_t(capacity) >= text.size())
            std::memcpy(out, text.data(), text.size());
        return text.size();
    } catch (const std::exception &e) {
        frontend_error = e.what();
        return -1;
    }
}
void *needle2_grammar_compile(void *tokenizer, const char *tools, size_t size) {
    try {
        if (!tokenizer || (!tools && size))
            throw std::runtime_error("invalid grammar arguments");
        std::string text(tools ? tools : "", size);
        auto json = frontend::Parser{text}.parse();
        return new frontend::Grammar(json, *static_cast<frontend::Tokenizer *>(tokenizer));
    } catch (const std::exception &e) {
        frontend_error = e.what();
        return nullptr;
    }
}
void needle2_grammar_free(void *p) { delete static_cast<frontend::Grammar *>(p); }
const DFAStateDesc *needle2_grammar_dfa(void *p) {
    return p ? &static_cast<frontend::Grammar *>(p)->desc : nullptr;
}
int needle2_engine_decode_compiled(void *engine, const float *logits, int vocab, int limit,
                                   void *grammar, int *out, int *count) {
    try {
        if (!engine || !logits || !out || !count || limit < 0 ||
            vocab != static_cast<Engine *>(engine)->c.vocab)
            throw std::runtime_error("invalid compiled decode arguments");
        auto *g = static_cast<frontend::Grammar *>(grammar);
        if (g && g->vocab != vocab)
            throw std::runtime_error("grammar vocabulary mismatch");
        int best = 0;
        for (int i = 0; i < vocab; ++i) {
            if (std::isnan(logits[i]))
                throw std::runtime_error("NaN logits are not supported");
            if (logits[i] > logits[best])
                best = i;
        }
        return static_cast<Engine *>(engine)->decode_loop(best, limit, g ? &g->desc : nullptr, out,
                                                          count);
    } catch (const std::exception &e) {
        frontend_error = e.what();
        return -1;
    }
}
}
