"""Edit (round 5): keep the round-4 winners in SK2_ALT for a clean (non-DIAG) autotune, add bigger-copy ("w": UPW 4-8)
variants, retire the round-4 losers. Run: python3 tools/_edit_round5.py"""
import pathlib
p = pathlib.Path("muse_bf16_proj/submission.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''    ('pooler8h_r100', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),        # pooler8h_r72 with a 100 KB ring (5 x 20 KB), 2 CTAs/SM
    ('pooler8t_r192', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),        # 128 CTAs, 6 x 32 KB (24 KB weight copies)
    ('pooler16h_r100', 'pooler', 4096, 16, 2, 1, 2, 4, 1, 1),       # 256 CTAs, 2/SM, 3 x 28 KB (12 KB copies)
    ('pooler16t_r220', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # pooler16t_r192 with 5 stages
    ('pooler32_r216', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),        # pooler32_r144 with 7 stages
    ('pooler32w_r216', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # 48 KB stages (24 KB copies), 4 stages
    ('pooler32h_r100', 'pooler', 4096, 16, 2, 1, 2, 4, 1, 1),       # 256 CTAs, 2/SM, 12 KB copies, 28 KB x 3
    ('qkv8t_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),         # qkv8t_r144 with 8 stages
    ('qkv16t_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1),        # qkv16t_r200 with 6 stages
    ('qkv32_r220', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),         # qkv32_r208 (4 x 50 KB stays 4)
    ('qkv8w_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 4, 2, 1),         # 36 KB weight copies (1024 k per stage)
    ('unpooler16t_r100', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # unpooler16t_r72 with 5 stages
    ('unpooler32_r216', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),    # unpooler32_r144 with 5 stages
    ('unpooler32h_r100', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # 256 CTAs, 2/SM (28 KB x 3)
''', '''    # round 4 (prefill-slot 69f30550, DIAG build, raw us): loads-only is 17.8-18.5 for EVERY pooler-class config (ring
    # 3-7 stages, 12-24 KB copies, 1-2 CTAs/SM, producer spin vs suspend-hint +-0.1, CTA-contiguous layout +0.3-0.5):
    # the stream rate is not in the ring mechanics; what moves the full time is bigger per-stage work (fewer barrier
    # round trips per byte): pooler32w 23.0 vs 26.2, qkv8w 15.1 vs 16.1, unpooler32h 19.5 vs 20.4. Retired: pooler8h_r100
    # 21.9 vs 20.4, pooler8t_r192 21.3, pooler32_r216 25.9, pooler32h_r100 28.3, qkv8t_r216 16.1, qkv16t_r216 15.7 vs 15.6,
    # unpooler16t_r100 19.3 vs 18.5, unpooler32_r216 20.6 vs 20.4.
    ('pooler16h_r100', 'pooler', 4096, 16, 2, 1, 2, 4, 1, 1),       # 256 CTAs, 2/SM, 3 x 28 KB (12 KB copies)
    ('pooler16t_r220', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # pooler16t_r192 with 5 stages
    ('pooler32w_r216', 'pooler', 4096, 32, 2, 2, 2, 4, 1, 1),       # 56 KB stages (24 KB copies), 3 stages
    ('qkv32_r220', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),         # qkv32_r208 (4 x 50 KB stays 4)
    ('qkv8w_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 4, 2, 1),         # 36 KB weight copies (1024 k per stage)
    ('unpooler32h_r100', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1),   # 256 CTAs, 2/SM (28 KB x 3)
    # round 5: more bigger-copy variants
    ('pooler8w_r216', 'pooler', 4096, 32, 2, 2, 2, 8, 1, 1),        # 48 KB copies (1024 k per stage), 3 x 64 KB, 128 CTAs
    ('qkv16w_r216', 'qkv_gate', 2304, 24, 3, 1, 4, 4, 1, 1),        # 36 KB copies, 68 KB stages x 3
    ('unpooler16w_r216', 'unpooler', 8192, 32, 2, 2, 2, 4, 1, 1),   # 24 KB copies, 40 KB stages x 5, 128 CTAs
    ('o16w_r100', 'o', 8192, 32, 2, 2, 2, 4, 1, 1),                 # 24 KB copies, 40 KB x 2 (K = 1024), 256 CTAs
''')
rep('''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's (see the retired list)
    ('pooler', 8): ['pooler8h_r100', 'pooler8t_r192'],
    ('pooler', 16): ['pooler16h_r100', 'pooler16t_r220'],
    ('pooler', 32): ['pooler32_r216', 'pooler32w_r216', 'pooler32h_r100'],
    ('qkv_gate', 8): ['qkv8t_r216', 'qkv8w_r216'],
    ('qkv_gate', 16): ['qkv16t_r216'],
    ('qkv_gate', 32): ['qkv32_r220'],
    ('unpooler', 16): ['unpooler16t_r100'],
    ('unpooler', 32): ['unpooler32_r216', 'unpooler32h_r100'],
}
''', '''SK2_ALT = {  # (layer, tokr) -> alternative config names the autotuner times against SK2_PICK's (see the retired list)
    ('pooler', 8): ['pooler8w_r216'],
    ('pooler', 16): ['pooler16h_r100', 'pooler16t_r220'],
    ('pooler', 32): ['pooler32w_r216'],
    ('qkv_gate', 8): ['qkv8w_r216'],
    ('qkv_gate', 16): ['qkv16w_r216'],
    ('qkv_gate', 32): ['qkv32_r220'],
    ('unpooler', 16): ['unpooler16w_r216'],
    ('unpooler', 32): ['unpooler32h_r100'],
    ('o', 16): ['o16w_r100'],
}
''')
p.write_text(s)
print("ok")
