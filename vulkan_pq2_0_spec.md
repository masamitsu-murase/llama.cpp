# 目的

PrismML-Eng/llama.cpp の ggml/src/ggml-vulkan に対して、
PQ2_0 を fixed group-128 signed 2-bit format として直接処理する
Vulkan GLSL shader を実装する。Bonsai 2 では q=3 が現れないことを
期待できるが、GGML_TYPE_PQ2_0 自体は {-1, 0, +1, +2} を表現する。
したがって shader は q=3 も +2 として正しく処理しなければならない。

最終目的は Intel GPU 上で Vulkan backend を使用した
Ternary Bonsai 2 27B の推論速度を最大化することである。

特に以下を重視する。

1. PQ2_0 の decode overhead を最小化する
2. integer dot product / DP4A を使用する
3. B 側の Q8_1 データを効率よく利用する
4. Xe1 subgroup=8 と Xe2 subgroup=16 の両方を考慮する
5. 既存 llama.cpp の PQ2_0 数値結果と一致させる
6. 既存の汎用 MMQ shader と比較できる形にする
7. existing test/benchmark infrastructure で correctness と performance を
    両方検証する

---

# 1. 最初に必ず調査すること

実装を開始する前に、現在 checkout されている
PrismML-Eng/llama.cpp のソースを調査すること。

特に以下を読む。

ggml/src/ggml-vulkan/
    CMakeLists.txt
    ggml-vulkan.cpp
    vulkan-shaders/
        mul_mat_vecq.comp
        mul_mat_vecq_funcs.glsl
        mul_mat_vec_base.glsl
        vulkan-shaders-gen.cpp
    各 shader の生成・登録処理

GitHub の最新版を直接参照する場合は、

https://github.com/PrismML-Eng/llama.cpp/tree/prism/ggml/src/ggml-vulkan

を基準にする。

現在の実装を勝手に想定せず、実際の checkout のコードを確認する。

---

# 2. 現在の PQ2_0 format を厳密に維持する

現在の PQ2_0 の基本形式は以下。

block_pq2_0:

    fp16 d
    2-bit quantized values
    group size = 128

128 個の weight を 2 bit ずつ格納する。

Bonsai 2 の ternary weight では q は基本的に

    q = 0 -> -1
    q = 1 ->  0
    q = 2 -> +1

として扱う。

一般 PQ2_0 の q=3 は

    q = 3 -> +2

である。

したがって decode は単純に

    signed_value = q - 1

でよい。

重要:

「ternary 専用」とは q=3 を無視するという意味ではない。

実際の Bonsai 2 データで q=3 が存在しないことは format の保証ではない。
q=3 の不在を host-side selection の条件にしてはならず、decoder 自体は
q=3 -> +2 になるようにしておく。

---

# 3. 既存の PQ2_0 decoder を確認する

現在の mul_mat_vecq_funcs.glsl には概ね以下の decode がある。

uint unpack_pq2_0(uint bits) {
    bits &= 0xffu;
    bits = (bits | (bits << 12u)) & 0x000f000fu;
    return (bits | (bits << 6u)) & 0x03030303u;
}

repack4() では、

const uint qs_idx = (ib & 3u) * 4u + iqs * 2u;

const uint bits =
    pack32(
        u16vec2(
            data_a_packed16[ib / 4].qs[qs_idx],
            data_a_packed16[ib / 4].qs[qs_idx + 1]
        )
    );

を使用している。

この memory layout を絶対に変更しない。

---

# 4. signed decoder を実装する

q=0,1,2,3 を signed int8 の bit pattern

    -1,0,+1,+2

へ変換する。

4 byte を一度に処理する。

推奨実装:

uint pq2_ternary4(uint bits)
{
    bits &= 0xffu;

    bits = (bits | (bits << 12u)) & 0x000f000fu;
    bits = (bits | (bits << 6u)) & 0x03030303u;

    return ((bits ^ 0x80808080u)
            - 0x01010101u)
           ^ 0x80808080u;
}

返り値の `uint` の各 byte は、signed int8 として解釈すると

    0 -> FF (-1)
    1 -> 00 ( 0)
    2 -> 01 (+1)
    3 -> 02 (+2)

となる。

`pq2_ternary4()` は packed bit pattern を保持するため `uint` を返す。
`dotPacked4x8EXT()` の signed overload に渡す前に、各 packed word を
`int32_t` として扱う。

単純に

    bits - 0x01010101

としてはいけない。

byte 境界を越えた borrow が発生する可能性があるためである。

---

# 5. DP4A を利用する

GLSL では既存実装と同じ

    dotPacked4x8EXT()

を利用する。

extension:

    #extension GL_EXT_integer_dot_product : require

を利用する。

A 側は signed int8 packed data とする。

B 側も既存 Q8_1 の signed int8 packed data を使用する。

つまり、

    DP4A(A[0..3], B[0..3])

を実行する。

16 weight については DP4A を4回実行する。

例:

int32_t pq2_dot16(
    uint a_block_idx,
    uint iqs
)
{
    i32vec4 qa = pq2_load16(a_block_idx, iqs);

    int32_t sum = 0;

    sum += dotPacked4x8EXT(qa.x, cache_b_qs[0]);
    sum += dotPacked4x8EXT(qa.y, cache_b_qs[1]);
    sum += dotPacked4x8EXT(qa.z, cache_b_qs[2]);
    sum += dotPacked4x8EXT(qa.w, cache_b_qs[3]);

    return sum;
}

現在の `mul_mat_vecq.comp` の PQ2_0 Q8_1 path は、すでに
`dotPacked4x8EXT()` を4回使用している。Phase 1 の目的は DP4A の
新規導入ではなく、A を unsigned q ではなく signed (q - 1) として
pack した場合にも、scale を含む最終 output を一致させることである。

---

# 6. pq2_load16() を実装する

現在の repack4() と完全に同じ memory layout から
16 quant を取り出す。

概念的には、

    qs_idx = (ib_a & 3) * 4 + iqs * 2

とする。

uint16 x 2 = 32 bit = 16 x 2-bit quant

を読み出す。

各 uint16 の low byte / high byte に
4 quant が入っている。

したがって、

    low byte  -> 4 quant
    high byte -> 4 quant

を decode し、

    i32vec4

として返す。

重要なのは、memory layout と量子化値の順序が既存

    repack4(ib_a, iqs)

と一致することだけである。`repack4()` は unsigned q を返す。一方、
`pq2_load16()` は同じ q を signed (q - 1) に変換して返すため、packed
int8 のビット列および `q_sum` が `repack4()` と同じになることを要求しては
ならない。

---

# 7. まず既存汎用 shader と同じ最終計算結果を作る

いきなり subgroup 最適化をしない。

第一段階では、

    mul_mat_vecq.comp

の PQ2_0 path の decoder を signed decoder に置き換える形で実装する。
目的は decoder と DP4A の correctness を確認することである。

既存 path の `q_sum` は unsigned q の dot product である。signed (q - 1)
を DP4A に渡す `q_sum` と同じ値にはならないため、`q_sum` 単体を比較対象に
してはならない。scale と Q8_1 correction を含む最終 output を、既存
generic shader と CPU reference に比較する。

---

# 8. scale / Q8_1 補正を A の表現に合わせる

現在の PQ2_0 path では、

    mul_q8_1()

が概ね

    da * (q_sum * ds.x - ds.y / 2)

を使用している。

`ds.x` は Q8_1 activation scale、`ds.y` は
`ds.x * sum(q_b)` である。補正の有無は A に何を pack したかで決まる。

* 既存 generic path のように unsigned q を DP4A に渡す場合、16-element
    call ごとに `da * (q_sum * ds.x - ds.y / 2)` を使う。2 call の合計で
    1つの Q8_1 block の `ds.y` を一度だけ減算し、q - 1 を表現する。
* signed (q - 1) を DP4A に渡す場合、補正は不要であり、
    `da * q_sum * ds.x` を使う。ここで `ds.y / 2` を引くと二重補正になる。

decoder、dot product、scale formula は不可分の組として実装し、異なる
表現の式を混在させない。

---

# 9. 次に専用 PQ2_0 signed shader を作る

第一段階の correctness が確認できたら、
汎用 shader とは別に専用 shader を作る。

例えば:

    mul_mat_vecq_pq2_signed.comp

など。

この shader は PQ2_0 専用とする。

他の量子化形式との共用コードを増やさない。

目的は compiler に以下を明示すること。

    A = PQ2_0 signed q - 1
    B = Q8_1
    dot = signed int8 DP4A
    K = 128

---

# 10. 128 weight を subgroup 単位で処理する

PQ2_0 の 1 block は 128 weight。

ここを shader の基本処理単位にする。

理想的な構造:

    one subgroup
        |
        +-- 128 A weights
        |
        +-- corresponding B Q8_1 values
        |
        +-- DP4A
        |
        +-- subgroup reduction
        |
        +-- row/column partial sum

K が128を超える通常の mat-vec では、1 subgroup が複数の PQ2_0 block を
stride で処理し、lane-local accumulator に加算してから1回だけ subgroup
reduction を行う。1 block ごとに subgroup reduction をしてそのまま output
する設計は K=128 にしか対応せず、複数 block の partial sum を失うため不可。

これにより、

    generic K_PER_ITER loop
    repeated B loads
    repeated address calculation

を削減する。

---

# 11. Xe1 の subgroup=8 を最適化する

Xe1 では subgroup size 8 を想定した variant を作る。

8 lanes × 16 values = 128 values。

したがって lane mapping は、

    lane = 0..7

    chunk32 = lane >> 1
    half16  = lane & 1

とする。

chunk32:

    0,1,2,3

half16:

    0,1

各 lane は 16 values を処理する。

各 lane は、

    16 values / 4 = 4 DP4A

を実行する。

したがって、

    8 lanes × 4 DP4A
    = 32 DP4A

で 128 weight の dot product を計算する。

---

# 12. Xe1 の A memory mapping

Xe1 では、

    qs_idx =
        (chunk32 & 3) * 4
        + half16 * 2

を使用する。

既存 repack4() と同じ layout にする。

16 quant を

    uint16 x 2

から取得する。

各 uint16 の low/high byte を decode して
4-byte packed int8 を得る。

最終的に 16 A values を

    i32vec4

として保持する。

---

# 13. Xe1 の B memory mapping

B は Q8_1。

1 Q8_1 block = 32 values。

128 A values に対応して

    4 Q8_1 blocks

が存在する。

lane の

    chunk32

が Q8_1 block を指定する。

half16=0 の場合:

    B index = 0..3

half16=1 の場合:

    B index = 4..7

とする。

つまり既存コードの

    b_inner * 8 + b_qs_idx * 4 + ...

と一致させる。

---

# 14. Xe2 の subgroup=16 を最適化する

Xe2 では subgroup size 16 を想定した variant を作る。

16 lanes × 8 values = 128 values。

lane mapping:

    chunk32 = lane >> 2
    half8   = lane & 3

とする。

chunk32:

    0..3

half8:

    0..3

各 lane は 8 values を処理する。

8 values = 2 DP4A。

したがって、

    16 lanes × 2 DP4A
    = 32 DP4A

となる。

Xe1 と同じく、1 subgroup で
PQ2_0 の 128 values を処理する。

---

# 15. Xe2 の A memory mapping

1 uint16 には 8 個の 2-bit quant が入っている。

したがって half8 に応じて

    uint16 index =
        (chunk32 & 3) * 4
        + half8

を使用する。この uint16 の low byte と high byte の両方を
`pq2_ternary4()` に渡す。各 byte は4 values、両 byte で8 values なので、
2つの packed signed int8 を得て DP4A を2回実行する。

`(half8 >> 1)` と byte selection の組合せは lane ごとに4 valuesしか
処理しないため、16 lanes で64 valuesとなり、この mapping には使わない。

---

# 16. Xe2 の B mapping

1 Q8_1 block = 32 values。

half8=0..3 に対して、

    B packed index =
        b_inner * 8
        + half8 * 2

および

    +1

を使う。

つまり各 lane が 8 B values を読み、

    DP4A
    DP4A

を実行する。

---

# 17. B scale の処理

B の各 Q8_1 block は

    ds.x
    ds.y

を持つ。

本仕様の dedicated path は signed (q - 1) を DP4A に渡す。したがって
Xe1 と Xe2 のどちらでも `ds.y` correction は適用しない。lane ごとの
contribution は `d_a * d_b * signed_dot` である。

unsigned q を使う別 design を評価する場合だけ、Q8_1 block ごとに全32
activation の `ds.y` を合計で一度だけ減算する必要がある。その場合は
Xe1 の2 laneに `ds.y / 2`、Xe2 の4 laneに `ds.y / 4` を分配できるが、
これは signed decoder と併用してはならない。

---

# 18. subgroup reduction

各 lane は部分 dot product を計算する。

その後、

    subgroupAdd()

等の subgroup reduction を使用して
128 values の合計を求める。

既存の `reduce_result()` と同じ partial sum を二重に reduction しない。
one-workgroup/one-subgroup variant では `USE_SUBGROUP_ADD_NO_SHMEM` の
`reduce_result()` をそのまま使うか、同等の subgroup reduction と output
write を専用 shader に実装する。複数 subgroup の workgroup を使う場合は、
既存の shared-memory cross-subgroup reduction を一度だけ行う。

mul_mat_vec_base.glsl を必ず調査して、

    reduce_result()
    temp buffer
    workgroup reduction

の構造を確認してから統合する。

---

# 19. 専用 shader で重要な最適化

以下を行う。

### A

PQ2_0 の scale d は
128 weight block に対して1つだけ。

したがって loop の内側で毎回 load しない。

可能なら block 単位で一度だけ取得する。

---

### B

B Q8_1 scale も block 単位で取得する。

同じ Q8_1 block に属する複数 lane が
同じ ds を読む場合、

subgroup broadcast が有効なら使用する。

ただし使用する extension が
現在の Prism shader infrastructure で利用可能か
必ず確認する。

---

### C

address calculation を減らす。

特に、

    / QUANT_K_Q8_1
    % 4
    * 8

などを lane ごとに何度も実行しない。

subgroup/lane mapping から
直接 index を計算できるようにする。

---

### D

dynamic branch を避ける。

Xe1/Xe2 variant では、

    subgroup size
    K_PER_ITER

などを可能な限り compile-time constant にする。

---

# 20. Xe1/Xe2 mapping をどう選択するか

第一候補は、shared source に compile-time constants を与える variant と、
runtime pipeline selection の組合せとする。必要なら次のように source を
分けてもよい。

    pq2_ternary_xe1.comp
    pq2_ternary_xe2.comp

の2種類。

理由:

Xe1:

    subgroup = 8
    16 values/lane

Xe2:

    subgroup = 16
    8 values/lane

で最適 mapping が異なるため。

source を分けても、device architecture は shader compile 時には分からない。
現在の backend と同様に `VK_EXT_subgroup_size_control` で pipeline creation
時に required subgroup size を要求し、実際に作成できた pipeline だけを
選択する。specialization constants は lane mapping を定数化する手段にはなる
が、required subgroup size の runtime request の代替ではない。

最終的には、生成された SPIR-V が
実際に Xe1/Xe2 で効率的か benchmark で確認する。

---

# 21. Vulkan subgroup size を勝手に固定しない

重要。

Vulkan の subgroup size は
shader source 上で単純に

    local_size_x = 8

とすれば必ず subgroup=8 になる、
とは考えない。

以下を確認する。

    VkPhysicalDeviceSubgroupProperties
    subgroupSize
    supportedOperations
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT
    minSubgroupSize
    maxSubgroupSize
    requiredSubgroupSizeStages
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT
    subgroupSizeControl
    computeFullSubgroups

`VK_EXT_subgroup_size_control` が compute stage で使用可能で、要求する size
が min/max の範囲内にある場合だけ、pipeline creation 時に required subgroup
size を指定する。`gl_SubgroupSize` は runtime value を観測するための値であり、
lane mapping を静的に安全にするための要求そのものではない。

実際の Intel Vulkan driver が
Xe1/Xe2 でどう報告するか確認する。

---

# 22. local_size_x と subgroup の関係

Xe1 variant:

    local_size_x = 8

Xe2 variant:

    local_size_x = 16

を基本候補とする。ただしこの local size だけでは subgroup size は固定されず、
one workgroup = one subgroup も保証されない。pipeline の required subgroup
size を同じ値で要求し、利用可能なら full-subgroup support も要求する。

ただし、

    one workgroup = one subgroup

になることを前提にしすぎない。

現在の Prism の dispatch geometry と
reduce_result() の設計を調査して、
正しい workgroup/subgroup mapping を決める。

---

# 23. output mapping

現在の

    first_row
    row
    col
    temp
    reduce_result()

の仕組みを変更する場合は、
既存 shader と同じ output mapping を維持する。

特に、

    matrix A row
    matrix B column
    output C

の対応を壊さない。

---

# 24. host-side dispatch

GLSL だけ変更して終わりにしない。

Vulkan backend の shader registration / pipeline creation を調査し、

    GGML_TYPE_PQ2_0 + Q8_1

の場合に専用 shader が選択されるようにする。GGML tensor type には
"Bonsai 2 ternary" と "q=3 を含む一般 PQ2_0" を区別する metadata はない。
よって q=3 の不在を selection 条件にせず、通常の PQ2_0 にも正しい
signed decoder を使う。selection は Q8_1 path、integer-dot capability、
subgroup-size-control capability、および作成できた pipeline に基づける。

---

# 25. 最初から q=3 を仮定して分岐しない

悪い実装:

    if (q == 3)
        ...

これは避ける。

q-1 を SIMD/DP4A friendly に実装する。

これにより

    0 -> -1
    1 ->  0
    2 -> +1
    3 -> +2

を branchless に処理する。

---

# 26. correctness test

最低限、以下を比較する。

A:

    existing generic PQ2_0 shader

B:

    new PQ2_0 signed shader

C:

    CPU/reference implementation

同じ入力について、

    C = A * B

の結果を比較する。

許容誤差は既存 PQ2_0 backend の
floating point tolerance に合わせる。

特に以下をテストする。

    q = 0
    q = 1
    q = 2
    q = 3

すべてを人工的に生成する。

---

# 27. 境界値 test

以下のパターンを必ずテストする。

1.
全 q=0

2.
全 q=1

3.
全 q=2

4.
全 q=3

5.
0,1,2 の周期パターン

6.
ランダム q

7.
128要素境界

8.
複数 PQ2 block

9.
K が128の倍数である複数の legal case (128, 256, 384, ...)

10.
matrix row/column が複数の場合

PQ2_0 tensor の row width は128の倍数でなければならない。K が128の倍数で
ない入力を PQ2_0 matmul test として生成してはならない。padding、view、または
dispatch boundary の一般的な挙動を確認する場合は、PQ2_0 tensor を不正に作らず、
別の legal tensor test として扱う。

---

# 28. 特に byte borrow bug をテストする

以下の decode を個別に確認する。

    00
    01
    02
    03
    00 01 02 03
    03 02 01 00

期待値:

    FF
    00
    01
    02

signed int8 として、

    -1
     0
    +1
    +2

となることを確認する。

---

# 29. performance benchmark

最低でも以下を比較する。

1.
existing generic PQ2_0 Q8_1 integer-dot shader

2.
generic PQ2_0 signed-decoder shader

3.
PQ2_0 signed dedicated shader

4.
Xe1 variant

5.
Xe2 variant

計測:

    tokens/sec
    prompt processing tok/s
    decode tok/s
    shader execution time
    GPU utilization
    memory bandwidth
    CPU overhead

---

# 30. micro benchmark を用意する

LLM 全体だけではなく、128-element PQ2 dot product の micro benchmark を
既存の benchmark infrastructure で実行できるようにする。新しい
`tests/*` file は maintainer approval なしに追加しない。専用 benchmark が
既存 infrastructure に収まらない場合は、追加前に maintainer approval を得る。

例えば、

    N = 128
    256
    512
    1024
    4096
    16384

について、

    existing generic integer-dot
    signed dedicated

を比較する。

目的は、

    decode overhead
    DP4A throughput
    memory load
    subgroup reduction

のどこがボトルネックかを分離すること。

---

# 31. expected instruction structure

理想的には各 lane について、

Xe1:

    16 A values
    16 B values

    4 × DP4A

Xe2:

    8 A values
    8 B values

    2 × DP4A

とする。

128 values 全体では

    32 DP4A

である。

重要なのは、

    2-bit decode
    → packed int8
    → DP4A

を可能な限り少ない instruction で行うこと。

---

# 32. generic shader との最大の違い

generic implementation は、

    K_PER_ITER
    b_qs_idx
    a_block_idx
    cache_b_qs
    mmvq_dot_product()
    reduce_result()

という汎用 abstraction を通る。

dedicated implementation では、
PQ2_0 の固定構造を利用して、

    128 A weights
    =
    4 × 32-value B blocks

を直接処理する。

この固定化によって、

    index calculation
    function abstraction
    repeated loads
    unnecessary decode

を減らす。

---

# 33. ただし最初から全て変更しない

実装は以下の3段階に分ける。

Phase 1:

    existing integer-dot shader
    +
    branchless signed PQ2 decoder
    +
    signed scale formula

目的:
correctness

Phase 2:

    dedicated PQ2 signed shader

目的:
generic overhead 削減

Phase 3:

    subgroup 8 / 16 specialization
    +
    subgroup reduction
    +
    scale hoisting
    +
    B load reuse

目的:
Intel GPU performance

各 phase ごとに benchmark を取る。

---

# 34. コンパイル確認

shader compiler で必ず確認する。

    glslc
    spirv-val
    spirv-dis

等、現在の Prism build system が使用している
既存 toolchain を利用する。

SPIR-V で `OpSDotKHR` (toolchain 表記により対応する integer-dot opcode) が
出力されることを確認する。GLSL source に `dotPacked4x8EXT()` があるだけでは
不十分である。

単に GLSL がコンパイルできただけでは不十分。

---

# 35. DP4A が本当に生成されているか確認する

以下を確認する。

GLSL: `dotPacked4x8EXT()`

SPIR-V: integer-dot instruction

device: `integerDotProduct4x8BitPackedSignedAccelerated` と
`shaderIntegerDotProduct` が有効であること

もし shader compiler が
大量の scalar integer multiply/add に展開しているなら、
実装は成功とはみなさない。

---

# 36. Intel GPU での確認

最低限、

    Intel Xe1
    Intel Xe2

で別々に benchmark する。

GPU 名、driver version、Vulkan API version、
subgroup size をログに出す。

例:

    GPU:
    Vulkan:
    driver:
    subgroupSize:
    shader:
    localSize:
    tokens/sec:

---

# 37. correctness が最優先

性能最適化によって
既存 PQ2_0 と結果が変わる場合は、
性能測定より先に原因を調べる。

特に確認するもの:

    signedness
    int8 packing
    q-1 mapping
    uint16 endian/order
    qs index
    Q8_1 ds.x
    Q8_1 ds.y
    ds.y correction matches the selected unsigned or signed representation
    subgroup reduction
    row/column mapping

---

# 38. 重要な禁止事項

以下を行わない。

- PQ2_0 の memory layout を変更する
- q=3 を勝手に q=1 にする
- signed/unsigned dot product を混同する
- unsigned-q path の ds.y correction を削除する
- signed (q - 1) path に ds.y correction を追加する
- generic reduce と専用 reduce を二重に行う
- subgroup size を仮定したまま runtime check をしない
- correctness test なしで performance optimization を進める
- CPU 側の tensor format を変更して shader だけ合わせる
- Bonsai 2 の Hadamard transform を shader 内に勝手に追加する

Hadamard transform が必要な場合は、
PQ2_0 dot kernel とは別の問題として扱う。

---

# 39. 最終的なファイル構成の候補

例えば、

ggml/src/ggml-vulkan/
    mul_mat_vecq.comp
    mul_mat_vecq_funcs.glsl

    mul_mat_vecq_pq2_signed.comp
    mul_mat_vecq_pq2_signed.glsl

など。

ただし現在の Prism の shader generation convention を
最優先し、既存構造に合わせる。

新しいファイルを増やす前に、
既存 shader の生成方法を確認する。

---

# 40. 最終成果物

実装完了時には以下を提示する。

1. 変更したファイル一覧

2. 各ファイルの変更内容

3. PQ2 decoder のコード

4. DP4A dot-product のコード

5. Xe1 mapping

6. Xe2 mapping

7. subgroup reduction の方法

8. host-side shader selection

9. correctness test の結果

10. benchmark 結果

11. SPIR-V の integer dot instruction の確認結果

12. generic PQ2_0 との性能差

13. Xe1 / Xe2 の性能差

14. bottleneck の分析

---

# 最終目標

最終的な専用 kernel は概念的に、

    PQ2_0 block
          |
          v
    128 × 2-bit
          |
          v
    branchless q-1
          |
          v
    packed signed int8
          |
          v
    DP4A
          |
          v
    subgroup reduction
          |
          v
    FP scale (signed decoder path: Q8_1 correctionなし)
          |
          v
    output

という構造にする。

Xe1:

    8 lanes
    × 16 weights
    × 4 DP4A

Xe2:

    16 lanes
    × 8 weights
    × 2 DP4A

を基本設計とする。

ただし実際の local_size / subgroup size / reduction は
現在の PrismML-Eng/llama.cpp の dispatch と
Intel Vulkan driver の仕様を確認した上で決定する。

「コードを書いて終了」ではなく、

    correctness
    → SPIR-V確認
    → microbenchmark
    → LLM benchmark
    → Xe1/Xe2比較

まで実施すること。
