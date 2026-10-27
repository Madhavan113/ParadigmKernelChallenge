"""Edit (round 3, after the sk2 attribution): small-R sk2 configs (3-4 CTAs per SM: more bytes in flight per SM, and the
other CTAs' loads overlap one CTA's mma/decode phase) as SK2_ALT candidates for every layer / tokr; dead gqx configs
pruned; DIAG_SK2 off in production (the bench generator turns it on). Run: python3 tools/_edit_round3.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


a = s.index("GQX_CONFIGS = {")
b = s.index("USE_GQX = True")
s = s[:a] + '''GQX_CONFIGS = {
    # judged (runs f3e50d8b / prefill-slot 11047a44, raw graph-replay us): qkv m128 (1,3,8) 17.9-18.1 beats every wider /
    # deeper split ((2,5,6) 19.6, (2,4,6) 20.1, (2,6,6) 29 second wave, (3,8,5) 21.8); pooler m128 gq (1,2,8) 25.4 beats
    # gqx (1,2,8) 26.0 (its DSMEM epilogue costs 0.6 us over gq's st.async one), (2,3,6) 25.9, (2,4,6) 37 (wave), (4,8,4)
    # 44 (wave); unpooler m128 torchT 22.7 beats (1,1,8) 23.5, (2,2,6) 24.7, (4,3,4) 28.7, (4,4,4) 42 (wave); o m128
    # torchT 10.4-10.6 vs (2,2,6) 10.5-12.9; m256: pooler (1,2,5) 32.6 beats torch 33.9, loses to CUTLASS 31.3; qkv (1,3,5)
    # 24 vs torch 23; unpooler (1,1,5) 29.5 vs torchT 27.6; o (2,2,4) 13.3-16.6 vs 12.4; BN 256 with 2 warpgroups 49-56.
    # Max active clusters at ~200 KB smem: cl2 66, cl3 39, cl4 30, cl5 22, cl6 17, cl8 15 (a grid beyond runs two waves).
    ("qkv_gate", 128): [(1, 3, 8)],   # v141: 36 row blocks x 3 K-thirds = 108 CTAs (cuBLAS 16.4 -> 14.0 us at m=128)
    ("pooler", 256): [(1, 2, 5)],     # 128 CTAs: 32.6 vs cuBLAS 33.9 (CUTLASS 31.3 when its build makes it in time)
}
''' + s[b:]

rep('''    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
]
SK2_FUSED_CONFIG = 'pooler\'''', '''    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    # small-R configs (attribution 11047a44: every m <= 32 shape is load-bound at ~3.5 TB/s with 1-2 CTAs per SM, the
    # mma/decode phases not overlapped at m=32): 14-22 KB stages, 3-stage rings of ~45-70 KB -> 3-4 CTAs per SM
    ('pooler8q_r48', 'pooler', 4096, 8, 1, 1, 4, 2, 2, 1),          # 512 CTAs, 4 consumer warps, 512 k/stage
    ('pooler16q_r72', 'pooler', 4096, 8, 1, 1, 4, 2, 2, 1),         # 512 CTAs
    ('pooler32q_r60', 'pooler', 4096, 8, 1, 1, 4, 1, 2, 1),         # 512 CTAs, 256 k/stage
    ('pooler16h_r60', 'pooler', 4096, 16, 2, 1, 4, 1, 2, 1),        # 256 CTAs, 256 k/stage
    ('pooler32h_r72', 'pooler', 4096, 16, 2, 1, 4, 1, 1, 1),        # 256 CTAs
    ('qkv8q_r48', 'qkv_gate', 2304, 8, 1, 1, 4, 2, 2, 1),           # 288 CTAs
    ('qkv16q_r72', 'qkv_gate', 2304, 8, 1, 1, 4, 2, 2, 1),
    ('qkv32q_r60', 'qkv_gate', 2304, 8, 1, 1, 4, 1, 2, 1),
    ('unpooler16q_r48', 'unpooler', 8192, 16, 2, 1, 4, 1, 2, 1),    # 512 CTAs, 256 k/stage (K = 4096: 16 stages)
    ('unpooler32q_r72', 'unpooler', 8192, 16, 2, 1, 4, 1, 1, 1),
    ('o8q_r48', 'o', 8192, 16, 2, 1, 4, 1, 2, 1),                   # 512 CTAs (K = 1024: 4 stages)
    ('o16q_r48', 'o', 8192, 16, 2, 1, 4, 1, 2, 1),
    ('o32q_r72', 'o', 8192, 16, 2, 1, 4, 1, 1, 1),
    ('slice8q_r48', 'unpooler', 1024, 8, 1, 1, 4, 1, 2, 1),         # the m <= 8 slice: 128 CTAs, 256 k/stage
]
SK2_FUSED_CONFIG = 'pooler\'''')

rep('''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's
    # judged (run f3e50d8b, raw us): qkv32c2a 17.7 / qkv32c2b 16.7 vs qkv32_r208 14.4; pooler32c2 22.1 vs 19.1;
    # pooler16c2 20.5 vs 18.0; unpooler32c2 21.8 vs 18.5: halving the x bytes per SM never paid for the cluster epilogue
}
''', '''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's
    # judged (run f3e50d8b, raw us): qkv32c2a 17.7 / qkv32c2b 16.7 vs qkv32_r208 14.4; pooler32c2 22.1 vs 19.1;
    # pooler16c2 20.5 vs 18.0; unpooler32c2 21.8 vs 18.5: halving the x bytes per SM never paid for the cluster epilogue
    ('pooler', 8): ['pooler8q_r48'],
    ('pooler', 16): ['pooler16q_r72', 'pooler16h_r60'],
    ('pooler', 32): ['pooler32q_r60', 'pooler32h_r72'],
    ('qkv_gate', 8): ['qkv8q_r48'],
    ('qkv_gate', 16): ['qkv16q_r72'],
    ('qkv_gate', 32): ['qkv32q_r60'],
    ('unpooler', 8): ['slice8q_r48'],
    ('unpooler', 16): ['unpooler16q_r48'],
    ('unpooler', 32): ['unpooler32q_r72'],
    ('o', 8): ['o8q_r48'],
    ('o', 16): ['o16q_r48'],
    ('o', 32): ['o32q_r72'],
}
''')
rep('''DIAG_SK2 = True   # measurement runs: compile the sk2 diagnostics in and time the dg modes from prepare (see _diag_sk2)
''', '''DIAG_SK2 = False   # measurement runs: compile the sk2 diagnostics in and time the dg modes from prepare (see _diag_sk2)
''')
p.write_text(s)
print("ok")
