"""Tokenizer adapted from Cactus Compute's Apache-2.0 reference exporter.

Uses the tokenizer embedded in .cact, without JAX or SentencePiece dependencies.
"""
import struct
TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_SP_META_SPACE = "▁"


def parse_tokenizer_blob(blob):
    off = struct.calcsize(_TK_HDR)
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
    rec = struct.calcsize(_TK_REC)
    pieces, scores, types = [], [], []
    for _ in range(n):
        score, t, ln = struct.unpack_from(_TK_REC, blob, off)
        off += rec
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    return {"pieces": pieces, "scores": scores, "types": types, "pad_id": pad,
            "eos_id": eos, "bos_id": bos, "unk_id": unk,
            "add_dummy_prefix": bool(add_dummy), "byte_fallback": bool(byte_fb)}


class RefTokenizer:
    def __init__(self, meta):
        self.pieces = meta["pieces"]
        self.scores = meta["scores"]
        self.types = meta["types"]
        self.add_dummy = meta["add_dummy_prefix"]
        self.byte_fallback = meta["byte_fallback"]
        self.unk_id = meta["unk_id"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i
                        for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types) if t == TK_USER_DEFINED),
                              key=len, reverse=True)

    @classmethod
    def from_cact(cls, path):
        from .archive import Archive
        blob = Archive.load(path).tensors["tokenizer"].blob
        return cls(parse_tokenizer_blob(blob))

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best_score, best_j = None, -1
            for j in range(len(syms) - 1):
                idx = self.p2id.get(syms[j] + syms[j + 1])
                if idx is not None and (best_score is None or self.scores[idx] > best_score):
                    best_score, best_j = self.scores[idx], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids = []
        for s in syms:
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append(idx)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf)); buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i]); i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text

