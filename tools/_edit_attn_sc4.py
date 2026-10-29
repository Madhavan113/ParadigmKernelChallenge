#!/usr/bin/env python3
"""Patch: OKC_SC4 -- sum_copies with four workspace copies loads all of them in one round trip (twin of OKC_SC8)."""
import pathlib, sys

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "muse_attention_decode/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("""#ifndef OKC_SC8
#define OKC_SC8 1       // coop finalize: the eight workspace copies' loads in one round trip
#endif""",
    """#ifndef OKC_SC8
#define OKC_SC8 1       // coop finalize: the eight workspace copies' loads in one round trip
#endif
#ifndef OKC_SC4
#define OKC_SC4 1       // ... and four copies (the rolled 4-copy loop below took two dependent L2 round trips)
#endif""")
rep("""  if (OKC_SC8 && p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }""",
    """  if (OKC_SC8 && p.nc == 8) {   // eight copies (the one-request q_len 8 plan): every copy's loads in one round trip
    if (act) {                   // (the rolled 4-copy loop below took two dependent L2 round trips)
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
  if (OKC_SC4 && p.nc == 4) {   // four copies: the same, one round trip
    if (act) {
      float4 xa[4], xb[4];
      float lc[4];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 4 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 4 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 4 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 4 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }""")
p.write_text(s)
print("patched", p)
