// This file is a part of Julia. License is MIT: https://julialang.org/license

// Processor feature detection

#include "fix_llvm_assert.h"

#include "processor.h"

#include "julia.h"
#include "julia_internal.h"

#include <map>
#include <array>
#include <algorithm>

// CPU target string is a list of strings separated by `;` each string starts with a CPU
// or architecture name and followed by an optional list of features separated by `,`.
// A "generic" or empty CPU name means the basic required feature set of the target ISA
// which is at least the architecture the C/C++ runtime is compiled with.

// CPU dispatch needs to determine the version to be used by the sysimg as well as
// the target and feature used by the JIT. Currently the only limitation on JIT target
// and feature is matching register size between the sysimg and JIT so that SIMD vectors
// can be passed correctly. This means disabling AVX and AVX2 if AVX was not enabled
// in sysimg and disabling AVX512 if it was not enabled in sysimg.
// This also possibly means that SVE needs to be disabled on AArch64 if sysimg doesn't have it
// enabled.

// CPU dispatch starts by first deciding the max feature set and CPU requested for JIT.
// This is the host or the target specified on the command line with features unavailable
// on the host disabled. All sysimg targets that require features not available in this set
// will be ignored.

// The next step is matching CPU name.
// If exact name match with compatible feature set exists, all versions without name match
// are ignored.
// This step will query LLVM first so it can accept CPU names that is recognized by LLVM but
// not by us (yet) when LLVM is enabled.

// If there are still more than one candidates, a feature match is performed.
// The ones with the largest register size will be used
// (i.e. AVX512 > AVX2/AVX > SSE, SVE > ASIMD). If there's a tie, the one with the most features
// enabled will be used. If there's still a tie the one that appears earlier in the list will be
// used. (i.e. the order in the version list is significant in this case).

// Features that are not recognized will be passed to LLVM directly during codegen
// but ignored otherwise.

namespace {

// Helper functions to test/set feature bits

template<typename T1, typename T2, typename T3>
static inline bool test_bits(T1 v, T2 mask, T3 test)
{
    return T3(v & mask) == test;
}

template<typename T1, typename T2>
static inline bool test_bits(T1 v, T2 mask)
{
    return test_bits(v, mask, mask);
}

template<typename T1, typename T2>
static inline bool test_nbit(const T1 &bits, T2 _bitidx)
{
    auto bitidx = static_cast<uint32_t>(_bitidx);
    auto u32idx = bitidx / 32;
    auto bit = bitidx % 32;
    return (bits[u32idx] & (1 << bit)) != 0;
}

template<typename T>
static inline void unset_bits(T &bits)
{
    (void)bits;
}

template<typename T, typename T1, typename... Rest>
static inline void unset_bits(T &bits, T1 _bitidx, Rest... rest)
{
    auto bitidx = static_cast<uint32_t>(_bitidx);
    auto u32idx = bitidx / 32;
    auto bit = bitidx % 32;
    bits[u32idx] = bits[u32idx] & ~uint32_t(1 << bit);
    unset_bits(bits, rest...);
}

template<typename T, typename T1>
static inline void set_bit(T &bits, T1 _bitidx, bool val)
{
    auto bitidx = static_cast<uint32_t>(_bitidx);
    auto u32idx = bitidx / 32;
    auto bit = bitidx % 32;
    if (val) {
        bits[u32idx] = bits[u32idx] | uint32_t(1 << bit);
    }
    else {
        bits[u32idx] = bits[u32idx] & ~uint32_t(1 << bit);
    }
}

// Helper functions to create feature masks

template<size_t n>
using FeatureList = std::array<uint32_t,n>;

template<size_t n>
static inline bool features_le(const FeatureList<n> &a, const FeatureList<n> &b)
{
    for (size_t i = 0; i < n; i++) {
        if (a[i] & ~b[i]) {
            return false;
        }
    }
    return true;
}

template<size_t n>
static inline int count_bits(const FeatureList<n> &a)
{
    int cnt = 0;
    for (size_t i = 0; i < n; i++) {
#ifdef __GNUC__
        cnt += __builtin_popcount(a[i]);
#else
        uint32_t v = a[i];
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        cnt += ((v + (v >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif
    }
    return cnt;
}

template<size_t n>
static inline int is_empty(const FeatureList<n> &a)
{
    for (size_t i = 0; i < n; i++) {
        if (a[i]) {
            return false;
        }
    }
    return true;
}

static inline constexpr uint32_t add_feature_mask_u32(uint32_t mask, uint32_t u32idx)
{
    return mask;
}

template<typename T, typename... Rest>
static inline constexpr uint32_t add_feature_mask_u32(uint32_t mask, uint32_t u32idx,
                                                      T bit, Rest... args)
{
    return add_feature_mask_u32(mask | ((int(bit) >= 0 && int(bit) / 32 == (int)u32idx) ?
                                        (1 << (int(bit) % 32)) : 0),
                                u32idx, args...);
}

template<typename... Args>
static inline constexpr uint32_t get_feature_mask_u32(uint32_t u32idx, Args... args)
{
    return add_feature_mask_u32(uint32_t(0), u32idx, args...);
}

template<uint32_t... Is> struct seq{};
template<uint32_t N, uint32_t... Is>
struct gen_seq : gen_seq<N-1, N-1, Is...>{};
template<uint32_t... Is>
struct gen_seq<0, Is...> : seq<Is...>{};

template<size_t n, uint32_t... I, typename... Args>
static inline constexpr FeatureList<n>
_get_feature_mask(seq<I...>, Args... args)
{
    return FeatureList<n>{{get_feature_mask_u32(I, args...)...}};
}

template<size_t n, typename... Args>
static inline constexpr FeatureList<n> get_feature_masks(Args... args)
{
    return _get_feature_mask<n>(gen_seq<n>(), args...);
}

template<size_t n>
static inline void mask_features(FeatureList<n> masks, uint32_t *features)
{
    for (size_t i = 0; i < n; i++) {
        features[i] = features[i] & masks[i];
    }
}

static inline std::string join_feature_strs(const std::vector<std::string> &strs)
{
    size_t nstr = strs.size();
    if (!nstr)
        return std::string("");
    std::string str = strs[0];
    for (size_t i = 1; i < nstr; i++)
        str += ',' + strs[i];
    return str;
}

static inline void append_ext_features(std::string &features, const std::string &ext_features)
{
    if (ext_features.empty())
        return;
    if (!features.empty())
        features.push_back(',');
    features.append(ext_features);
}

static inline void append_ext_features(std::vector<std::string> &features,
                                       const std::string &ext_features)
{
    if (ext_features.empty())
        return;
    const char *start = ext_features.c_str();
    for (const char *p = start; *p; p++) {
        if (*p == ',' || *p == '\0') {
            features.emplace_back(start, p - start);
            start = p + 1;
        }
    }
}

/**
 * Target specific type/constant definitions, always enable.
 */

struct FeatureName {
    const char *name;
    uint32_t bit; // bit index into a `uint32_t` array;
    uint32_t llvmver; // 0 if it is available on the oldest LLVM version we support
};

template<typename CPU, size_t n>
struct CPUSpec {
    const char *name;
    CPU cpu;
    CPU fallback;
    uint32_t llvmver;
    FeatureList<n> features;
};

struct FeatureDep {
    uint32_t feature;
    uint32_t dep;
};

template<size_t n>
static inline void enable_depends(FeatureList<n> &features, const FeatureDep *deps, size_t ndeps)
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (ssize_t i = ndeps - 1; i >= 0; i--) {
            auto &dep = deps[i];
            if (!test_nbit(features, dep.feature) || test_nbit(features, dep.dep))
                continue;
            set_bit(features, dep.dep, true);
            changed = true;
        }
    }
}

template<size_t n>
static inline void disable_depends(FeatureList<n> &features, const FeatureDep *deps, size_t ndeps)
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (ssize_t i = ndeps - 1; i >= 0; i--) {
            auto &dep = deps[i];
            if (!test_nbit(features, dep.dep) || test_nbit(features, dep.feature))
                continue;
            unset_bits(features, dep.feature);
            changed = true;
        }
    }
}

template<typename CPU, size_t n>
static const CPUSpec<CPU,n> *find_cpu(uint32_t cpu, const CPUSpec<CPU,n> *cpus, uint32_t ncpus)
{
    for (uint32_t i = 0; i < ncpus; i++) {
        if (cpu == uint32_t(cpus[i].cpu)) {
            return &cpus[i];
        }
    }
    return nullptr;
}

template<typename CPU, size_t n>
static const CPUSpec<CPU,n> *find_cpu(const char *name, const CPUSpec<CPU,n> *cpus, uint32_t ncpus)
{
    for (uint32_t i = 0; i < ncpus; i++) {
        if (strcmp(name, cpus[i].name) == 0) {
            return &cpus[i];
        }
    }
    return nullptr;
}

template<typename CPU, size_t n>
static const char *find_cpu_name(uint32_t cpu, const CPUSpec<CPU,n> *cpus, uint32_t ncpus)
{
    if (auto *spec = find_cpu(cpu, cpus, ncpus))
        return spec->name;
    return "generic";
}

template<typename CPU, size_t n>
static CPU find_cpu_id(const char *name, const CPUSpec<CPU,n> *cpus, uint32_t ncpus,
                       uint32_t def=0)
{
    if (auto *spec = find_cpu(name, cpus, ncpus))
        return spec->cpu;
    return static_cast<CPU>(def);
}

static uint32_t find_feature_bit(const FeatureName *features, size_t nfeatures,
                                 const char *str, size_t len)
{
    for (size_t i = 0; i < nfeatures; i++) {
        auto &feature = features[i];
        if (strncmp(feature.name, str, len) == 0 && feature.name[len] == 0) {
            return feature.bit;
        }
    }
    return (uint32_t)-1;
}

static inline std::vector<uint8_t> serialize_target_data(const char *name,
                                                         const uint32_t *features,
                                                         uint32_t nfeature,
                                                         const char *ext_features)
{
    std::vector<uint8_t> res;
    auto add_data = [&] (const void *data, size_t sz) {
        size_t old_sz = res.size();
        res.resize(old_sz + sz);
        memcpy(&res[old_sz], data, sz);
    };
    add_data(&nfeature, 4);
    add_data(features, 4 * nfeature);
    uint32_t namelen = strlen(name);
    add_data(&namelen, 4);
    add_data(name, namelen);
    uint32_t ext_features_len = strlen(ext_features);
    add_data(&ext_features_len, 4);
    add_data(ext_features, ext_features_len);
    return res;
}

template<size_t n>
static inline std::vector<uint8_t> serialize_target_data(const char *name,
                                                         const FeatureList<n> &features,
                                                         const char *ext_features)
{
    return serialize_target_data(name, &features[0], n, ext_features);
}

template<size_t n>
struct TargetData {
    std::string name;
    std::string ext_features;
    FeatureList<n> features;
    uint32_t flags;
};

template<size_t n>
static inline std::vector<TargetData<n>> deserialize_target_data(const uint8_t *data)
{
    auto load_data = [&] (void *dest, size_t sz) {
        memcpy(dest, data, sz);
        data += sz;
    };
    auto load_string = [&] () {
        uint32_t len;
        load_data(&len, 4);
        std::string res((const char*)data, len);
        data += len;
        return res;
    };
    uint32_t ntarget;
    load_data(&ntarget, 4);
    std::vector<TargetData<n>> res(ntarget);
    for (uint32_t i = 0; i < ntarget; i++) {
        auto &target = res[i];
        load_data(&target.flags, 4);
        // Starting serialized target data
        uint32_t nfeature;
        load_data(&nfeature, 4);
        if (nfeature < n) {
            load_data(&target.features[0], 4 * nfeature);
            memset(&target.features[nfeature], 0, (n - nfeature) * 4);
        }
        else {
            load_data(&target.features[0], 4 * n);
            data += 4 * (nfeature - n);
        }
        target.name = load_string();
        target.ext_features = load_string();
    }
    return res;
}

template<size_t n>
struct TargetArg : TargetData<n> {
    FeatureList<n> disable;
};

template<size_t n, typename F>
static inline std::vector<TargetArg<n>>
parse_cmdline(const char *option, F &&feature_cb)
{
    std::vector<TargetArg<n>> res;
    if (!option)
        return res;
    TargetArg<n> arg;
    auto reset_arg = [&] {
        arg.name.clear();
        arg.ext_features.clear();
        memset(&arg.features[0], 0, 4 * n);
        memset(&arg.disable[0], 0, 4 * n);
        arg.flags = 0;
    };
    const char *start = option;
    for (const char *p = option; ; p++) {
        switch (*p) {
        case ',':
            if (arg.name.empty()) {
                if (p == start)
                    jl_error("Empty CPU name");
                arg.name.append(start, p - start);
                start = p + 1;
                continue;
            }
            JL_FALLTHROUGH;
        case ';':
        case '\0': {
            bool done = *p == '\0';
            bool next_target = *p == ';';
            bool disable = false;
            const char *full = start;
            const char *fname = full;
            start = p + 1;
            if (*full == '-') {
                disable = true;
                fname++;
            }
            else if (*full == '+') {
                fname++;
            }
            const char *clone_all = "clone_all";
            size_t clone_all_len = strlen(clone_all);
            if (p - fname == clone_all_len && memcmp(clone_all, fname, clone_all_len) == 0) {
                if (!disable) {
                    arg.flags |= JL_TARGET_CLONE_ALL;
                }
            }
            else {
                FeatureList<n> &list = disable ? arg.disable : arg.features;
                if (!feature_cb(fname, p - fname, list)) {
                    if (!arg.ext_features.empty())
                        arg.ext_features += ',';
                    arg.ext_features.append(full, p - full);
                }
            }
            if (done) {
                return res;
            }
            if (next_target) {
                res.push_back(arg);
                reset_arg();
            }
        }
            JL_FALLTHROUGH;
        default:
            continue;
        }
    }
}

template<size_t n, typename F>
static inline std::vector<TargetArg<n>> &get_cmdline_targets(F &&feature_cb)
{
    static std::vector<TargetArg<n>> targets =
        parse_cmdline<n>(jl_options.cpu_target, std::forward<F>(feature_cb));
    return targets;
}

template<typename F>
static inline jl_sysimg_fptrs_t parse_sysimg(void *hdl, F &&callback)
{
    jl_sysimg_fptrs_t res = {nullptr, 0, nullptr, 0, nullptr, nullptr};
    res.base = (const char*)jl_dlsym(hdl, "jl_sysimg_fvars_base");
    auto offsets = ((const int32_t*)jl_dlsym(hdl, "jl_sysimg_fvars_offsets")) + 1;
    uint32_t nfunc = ((const uint32_t*)offsets)[-1];
    res.offsets = offsets;

    void *ids = jl_dlsym(hdl, "jl_dispatch_target_ids");
    int target_idx = callback(ids);

    auto got = (const void**)jl_dlsym(hdl, "jl_dispatch_got");
    auto got_idxs = ((const uint32_t*)jl_dlsym(hdl, "jl_dispatch_got_idxs")) + 1;
    uint32_t num_got = got_idxs[-1];

    auto clone_offsets = (const int32_t*)jl_dlsym(hdl, "jl_dispatch_fptr_offsets");
    auto clone_idxs = (const uint32_t*)jl_dlsym(hdl, "jl_dispatch_fptr_idxs");
    uint32_t nclone = clone_idxs[0];
    clone_idxs += 1;

    // Find target
    for (int i = 0;i < target_idx;i++) {
        clone_idxs += (nclone == nfunc ? 0 : nclone) + 1;
        clone_offsets += nclone;
        nclone = clone_idxs[-1];
    }

    // Fill in return value
    if (nclone == nfunc) {
        // All function cloned
        res.offsets = clone_offsets;
        // Also assume that no GOT slots needs to be filled.
        return res;
    }
    else {
        res.nclones = nclone;
        res.clone_offsets = clone_offsets;
        res.clone_idxs = clone_idxs;
    }

    // Do relocation
    uint32_t got_i = 0;
    for (uint32_t i = 0;i < nclone;i++) {
        const uint32_t mask = 1u << 31;
        uint32_t idx = clone_idxs[i];
        if (!(idx & mask))
            continue;
        idx = idx & ~mask;
        bool found = false;
        for (;got_i < num_got;got_i++) {
            if (got_idxs[got_i] == idx) {
                found = true;
                got[got_i] = clone_offsets[i] + res.base;
                break;
            }
        }
        assert(found && "Cannot find GOT entry for cloned function.");
        (void)found;
    }

    return res;
}


// Debug helper

template<typename CPU, size_t n>
static inline void dump_cpu_spec(uint32_t cpu, const FeatureList<n> &features,
                                 const FeatureName *feature_names, uint32_t nfeature_names,
                                 const CPUSpec<CPU,n> *cpus, uint32_t ncpus)
{
    bool cpu_found = false;
    for (uint32_t i = 0;i < ncpus;i++) {
        if (cpu == uint32_t(cpus[i].cpu)) {
            cpu_found = true;
            jl_safe_printf("CPU: %s\n", cpus[i].name);
            break;
        }
    }
    if (!cpu_found)
        jl_safe_printf("CPU: generic\n");
    jl_safe_printf("Features:");
    bool first = true;
    for (uint32_t i = 0;i < nfeature_names;i++) {
        if (test_nbit(&features[0], feature_names[i].bit)) {
            if (first) {
                jl_safe_printf(" %s", feature_names[i].name);
                first = false;
            }
            else {
                jl_safe_printf(", %s", feature_names[i].name);
            }
        }
    }
    jl_safe_printf("\n");
}

}

namespace X86 {
enum class CPU : uint32_t {
    generic = 0,
    intel_nocona,
    intel_prescott,
    intel_atom_bonnell,
    intel_atom_silvermont,
    intel_core2,
    intel_core2_penryn,
    intel_yonah,
    intel_corei7_nehalem,
    intel_corei7_westmere,
    intel_corei7_sandybridge,
    intel_corei7_ivybridge,
    intel_corei7_haswell,
    intel_corei7_broadwell,
    intel_corei7_skylake,
    intel_corei7_skylake_avx512,
    intel_corei7_cannonlake,
    intel_knights_landing,

    amd_fam10h,
    amd_athlon_fx,
    amd_athlon_64,
    amd_athlon_64_sse3,
    amd_bdver1,
    amd_bdver2,
    amd_bdver3,
    amd_bdver4,
    amd_btver1,
    amd_btver2,
    amd_k8,
    amd_k8_sse3,
    amd_opteron,
    amd_opteron_sse3,
    amd_barcelona,
    amd_znver1,
};

static constexpr size_t feature_sz = 9;
static constexpr FeatureName feature_names[] = {
#define X86_FEATURE_DEF(name, bit, llvmver) {#name, bit, llvmver},
#include "features_x86.h"
#undef X86_FEATURE_DEF
};
static constexpr uint32_t nfeature_names = sizeof(feature_names) / sizeof(FeatureName);

template<typename... Args>
static inline constexpr FeatureList<feature_sz> get_feature_masks(Args... args)
{
    return ::get_feature_masks<feature_sz>(args...);
}

static constexpr auto feature_masks = get_feature_masks(
#define X86_FEATURE_DEF(name, bit, llvmver) bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
    -1);

namespace Feature {
enum : uint32_t {
#define X86_FEATURE_DEF(name, bit, llvmver) name = bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
};
static constexpr FeatureDep deps[] = {
    {ssse3, sse3},
    {fma, avx},
    {sse41, ssse3},
    {sse42, sse41},
    {avx, sse42},
    {f16c, avx},
    {avx2, avx},
    {avx512f, avx2},
    {avx512dq, avx512f},
    {avx512ifma, avx512f},
    {avx512pf, avx512f},
    {avx512er, avx512f},
    {avx512cd, avx512f},
    {avx512bw, avx512f},
    {avx512vl, avx512f},
    {avx512vbmi, avx512bw},
    {avx512vpopcntdq, avx512f},
    {sse4a, sse3},
    {xop, fma4},
    {fma4, avx},
    {fma4, sse4a}
};

// Some of these can be simplified in c++14 by reusing the common part
// since `[]` on `std::array` is constexpr.

constexpr auto generic = get_feature_masks();
constexpr auto bonnell = get_feature_masks(sse3, cx16, movbe, sahf);
constexpr auto silvermont = get_feature_masks(sse3, ssse3, sse41, sse42, cx16, movbe, popcnt,
                                              pclmul, aes, prfchw, sahf);
constexpr auto core2 = get_feature_masks(sse3, ssse3, cx16, sahf);
constexpr auto yonah = get_feature_masks(sse3);
constexpr auto prescott = get_feature_masks(sse3);
constexpr auto nocona = get_feature_masks(sse3, cx16);
constexpr auto penryn = get_feature_masks(sse3, ssse3, sse41, cx16, sahf);
constexpr auto nehalem = get_feature_masks(sse3, ssse3, sse41, sse42, cx16, popcnt, sahf);
constexpr auto westmere = get_feature_masks(sse3, ssse3, sse41, sse42, cx16, popcnt, aes,
                                            pclmul, sahf);
constexpr auto sandybridge = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                               pclmul, xsave, xsaveopt, sahf);
constexpr auto ivybridge = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                             pclmul, xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase);
constexpr auto haswell = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                           pclmul, xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase,
                                           avx2, bmi, bmi2, fma, lzcnt, movbe);
constexpr auto broadwell = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                             pclmul, xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase,
                                             avx2, bmi, bmi2, fma, lzcnt, movbe, adx, rdseed,
                                             prfchw);
constexpr auto skylake = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                           pclmul, xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase,
                                           avx2, bmi, bmi2, fma, lzcnt, movbe, adx, rdseed, mpx,
                                           rtm, xsavec, xsaves, clflushopt,
                                           prfchw); // ignore sgx; hle
constexpr auto knl = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes, pclmul,
                                       xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase, avx512f, avx2,
                                       avx512er, avx512cd, avx512pf, prefetchwt1, adx, rdseed,
                                       movbe, lzcnt, bmi, bmi2, fma, prfchw);
constexpr auto skx = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes, pclmul,
                                       xsave, xsaveopt, sahf, rdrnd, f16c, fsgsbase, avx2, bmi,
                                       bmi2, fma, lzcnt, movbe, adx, rdseed, mpx, rtm, xsavec,
                                       xsaves, clflushopt, prfchw, avx512f, avx512cd, avx512dq,
                                       avx512bw, avx512vl, pku, clwb); // ignore sgx; hle
constexpr auto cannonlake = get_feature_masks(sse3, ssse3, sse41, sse42, avx, cx16, popcnt, aes,
                                              pclmul, xsave, xsaveopt, sahf, rdrnd, f16c,
                                              fsgsbase, avx2, bmi, bmi2, fma, lzcnt, movbe, adx,
                                              rdseed, mpx, rtm, xsavec, xsaves, clflushopt,
                                              prfchw, avx512f, avx512cd, avx512dq, avx512bw,
                                              avx512vl, pku, clwb, avx512vbmi, avx512ifma,
                                              sha); // ignore sgx; hle

constexpr auto k8_sse3 = get_feature_masks(sse3, cx16);
constexpr auto amdfam10 = get_feature_masks(sse3, sse4a, cx16, lzcnt, popcnt, sahf);

constexpr auto btver1 = get_feature_masks(sse3, ssse3, sse4a, cx16, prfchw, lzcnt, popcnt, sahf);
constexpr auto btver2 = get_feature_masks(sse3, ssse3, sse41, sse42, avx,
                                          sse4a, cx16, prfchw, aes, pclmul, bmi, f16c,
                                          movbe, lzcnt, popcnt, xsave, xsaveopt, sahf);

constexpr auto bdver1 = get_feature_masks(xop, fma4, avx, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, sse4a,
                                          lzcnt, popcnt, xsave, lwp, sahf);
constexpr auto bdver2 = get_feature_masks(xop, fma4, avx, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, sse4a,
                                          f16c, lzcnt, popcnt, xsave, bmi, tbm, lwp, fma, sahf);
constexpr auto bdver3 = get_feature_masks(xop, fma4, avx, sse3, ssse3, sse41,
                                          sse42, cx16, aes, prfchw, pclmul, sse4a,
                                          f16c, lzcnt, popcnt, xsave, bmi, tbm, lwp, fma,
                                          xsaveopt, fsgsbase, sahf);
constexpr auto bdver4 = get_feature_masks(avx2, avx, sse3, ssse3, sse41,
                                          sse42, xop, fma4, cx16, aes, prfchw, pclmul,
                                          f16c, lzcnt, popcnt, xsave, bmi, bmi2, tbm, lwp, fma,
                                          xsaveopt, fsgsbase, sahf, mwaitx);

constexpr auto znver1 = get_feature_masks(adx, aes, avx2, avx, sse3, ssse3, sse41,
                                          sse42, bmi, bmi2, clflushopt, clzero, cx16, f16c, fma,
                                          fsgsbase, sahf, lzcnt, movbe, mwaitx, pclmul,
                                          popcnt, prfchw, rdrnd, rdseed, sha, sse4a, xsave,
                                          xsavec, xsaveopt, xsaves);

}

static constexpr CPUSpec<CPU, feature_sz> cpus[] = {
    {"generic", CPU::generic, CPU::generic, 0, Feature::generic},
    {"bonnell", CPU::intel_atom_bonnell, CPU::generic, 0, Feature::bonnell},
    {"silvermont", CPU::intel_atom_silvermont, CPU::generic, 0, Feature::silvermont},
    {"core2", CPU::intel_core2, CPU::generic, 0, Feature::core2},
    {"yonah", CPU::intel_yonah, CPU::generic, 0, Feature::yonah},
    {"prescott", CPU::intel_prescott, CPU::generic, 0, Feature::prescott},
    {"nocona", CPU::intel_nocona, CPU::generic, 0, Feature::nocona},
    {"penryn", CPU::intel_core2_penryn, CPU::generic, 0, Feature::penryn},
    {"nehalem", CPU::intel_corei7_nehalem, CPU::generic, 0, Feature::nehalem},
    {"westmere", CPU::intel_corei7_westmere, CPU::generic, 0, Feature::westmere},
    {"sandybridge", CPU::intel_corei7_sandybridge, CPU::generic, 0, Feature::sandybridge},
    {"ivybridge", CPU::intel_corei7_ivybridge, CPU::generic, 0, Feature::ivybridge},
    {"haswell", CPU::intel_corei7_haswell, CPU::generic, 0, Feature::haswell},
    {"broadwell", CPU::intel_corei7_broadwell, CPU::generic, 0, Feature::broadwell},
    {"skylake", CPU::intel_corei7_skylake, CPU::generic, 0, Feature::skylake},
    {"knl", CPU::intel_knights_landing, CPU::generic, 0, Feature::knl},
    {"skylake-avx512", CPU::intel_corei7_skylake_avx512, CPU::generic, 0, Feature::skx},
    {"cannonlake", CPU::intel_corei7_cannonlake, CPU::intel_corei7_skylake_avx512, 40000,
     Feature::cannonlake},

    {"athlon64", CPU::amd_athlon_64, CPU::generic, 0, Feature::generic},
    {"athlon-fx", CPU::amd_athlon_fx, CPU::generic, 0, Feature::generic},
    {"k8", CPU::amd_k8, CPU::generic, 0, Feature::generic},
    {"opteron", CPU::amd_opteron, CPU::generic, 0, Feature::generic},

    {"athlon64-sse3", CPU::amd_athlon_64_sse3, CPU::generic, 0, Feature::k8_sse3},
    {"k8-sse3", CPU::amd_k8_sse3, CPU::generic, 0, Feature::k8_sse3},
    {"opteron-sse3", CPU::amd_opteron_sse3, CPU::generic, 0, Feature::k8_sse3},

    {"amdfam10", CPU::amd_fam10h, CPU::generic, 0, Feature::amdfam10},
    {"barcelona", CPU::amd_barcelona, CPU::generic, 0, Feature::amdfam10},

    {"btver1", CPU::amd_btver1, CPU::generic, 0, Feature::btver1},
    {"btver2", CPU::amd_btver2, CPU::generic, 0, Feature::btver2},

    {"bdver1", CPU::amd_bdver1, CPU::generic, 0, Feature::bdver1},
    {"bdver2", CPU::amd_bdver2, CPU::generic, 0, Feature::bdver2},
    {"bdver3", CPU::amd_bdver3, CPU::generic, 0, Feature::bdver3},
    {"bdver4", CPU::amd_bdver4, CPU::generic, 0, Feature::bdver4},

    {"znver1", CPU::amd_znver1, CPU::generic, 0, Feature::znver1},
};
static constexpr size_t ncpu_names = sizeof(cpus) / sizeof(cpus[0]);

}

namespace ARM {
enum class CPU : uint32_t {
    generic = 0,

    // ARM
    // armv6l
    arm_mpcore,
    arm_1136j_s,
    arm_1136jf_s,
    arm_1156t2_s,
    arm_1156t2f_s,
    arm_1176jz_s,
    arm_1176jzf_s,
    arm_cortex_m0,
    arm_cortex_m1,
    // armv7ml
    arm_cortex_m3,
    arm_cortex_m4,
    arm_cortex_m7,
    // armv7l
    arm_cortex_a5,
    arm_cortex_a7,
    arm_cortex_a8,
    arm_cortex_a9,
    arm_cortex_a12,
    arm_cortex_a15,
    arm_cortex_a17,
    arm_cortex_r4,
    arm_cortex_r5,
    arm_cortex_r7,
    arm_cortex_r8,
    // armv8ml
    arm_cortex_m23,
    arm_cortex_m33,
    // armv8l
    arm_cortex_a32,
    arm_cortex_r52,
    // aarch64
    arm_cortex_a35,
    arm_cortex_a53,
    arm_cortex_a57,
    arm_cortex_a72,
    arm_cortex_a73,

    // Cavium
    // aarch64
    cavium_thunderx2t99p1,
    cavium_thunderx,
    cavium_thunderx88,
    cavium_thunderx88p1,
    cavium_thunderx81,
    cavium_thunderx83,
    cavium_thunderx2t99,

    // NVIDIA
    // aarch64
    nvidia_denver1,
    nvidia_denver2,

    // AppliedMicro
    // aarch64
    apm_xgene1,
    apm_xgene2,
    apm_xgene3,

    // Qualcomm
    // armv7l
    qualcomm_scorpion,
    qualcomm_krait,
    // aarch64
    qualcomm_kyro,
    qualcomm_falkor,

    // Samsung
    // aarch64
    samsung_exynos_m1,
    samsung_exynos_m2,
    samsung_exynos_m3,

    // Apple
    // armv7l
    apple_swift,
    // aarch64
    apple_cyclone,
    apple_typhoon,
    apple_twister,
    apple_hurricane,

    // Marvell
    // armv7l
    marvell_pj4,

    // Intel
    // armv7l
    intel_3735d,
};

}

#if defined(_CPU_X86_) || defined(_CPU_X86_64_)

#include "processor_x86.cpp"

#elif defined(_CPU_AARCH64_)

// TODO
extern "C" JL_DLLEXPORT jl_value_t *jl_get_cpu_name(void)
{
    return jl_cstr_to_string("generic");
}

// FZ, bit [24]
static const uint32_t fpcr_fz_mask = 1 << 24;

static inline uint32_t get_fpcr_aarch64(void)
{
    uint32_t fpcr;
    asm volatile("mrs %0, fpcr" : "=r"(fpcr));
    return fpcr;
}

static inline void set_fpcr_aarch64(uint32_t fpcr)
{
    asm volatile("msr fpcr, %0" :: "r"(fpcr));
}

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return (get_fpcr_aarch64() & fpcr_fz_mask) != 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    uint32_t fpcr = get_fpcr_aarch64();
    fpcr = isZero ? (fpcr | fpcr_fz_mask) : (fpcr & ~fpcr_fz_mask);
    set_fpcr_aarch64(fpcr);
    return 0;
}

#else

// TODO
extern "C" JL_DLLEXPORT jl_value_t *jl_get_cpu_name(void)
{
    return jl_cstr_to_string("generic");
}

extern "C" JL_DLLEXPORT int32_t jl_get_zero_subnormals(void)
{
    return 0;
}

extern "C" JL_DLLEXPORT int32_t jl_set_zero_subnormals(int8_t isZero)
{
    return isZero;
}

#endif
