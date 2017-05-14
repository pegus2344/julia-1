// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "support/dtypes.h"

#include "julia.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Related sysimg exported symbols
 *
 * # Global function pointers
 * `jl_sysimg_fvars_base`:
 *     The address of this symbol is the base function pointer
 *     (all other pointers are stored as offsets to this address)
 * `jl_sysimg_fvars_offsets`: [static data]
 *     The array of function offsets (`int32_t`) from the base pointer.
 *     This includes all functions in sysimg.
 *     The default implementation is used if the function is cloned.
 *     The first element is the size of the array.
 *
 * # Target data and dispatch slots (Only needed by runtime during loading)
 * `jl_dispatch_target_ids`: [static data] serialize target data.
 *     This contains the number of targets which is needed to decode `jl_dispatch_fvars_idxs`
 *     in additional to the name and feature set of each target.
 * `jl_dispatch_got`: [uninitialized data] The runtime dispatch slots.
 *     The code in the sysimg will load from these slots.
 *     Not all functions being cloned are assigned a slot, only those that are directly called
 *     from the generic version do.
 * `jl_dispatch_got_idxs`: [static data] Dispatch slots indices.
 *     `uint32_t`s in `jl_sysimg_fvars_offsets` corresponding to each of the slots in
 *     `jl_dispatch_got`into . (The slots are sorted so that this array is ascending.)
 *     The first element is the number of got slots.
 *
 * # Target functions
 * `jl_dispatch_fvars_offsets`: [static data] Target specific function pointer offsets.
 *     This contains all the functions that are different from the default version.
 *     Note that not all of these needs to be filled into `jl_dispatch_got` (see below).
 * `jl_dispatch_fvars_idxs`: [static data] Target specific functions indices.
 *     The function pointer count and global index corresponding to
 *     each of the function pointer offsets in `jl_dispatch_fvars_offsets`.
 *     There is one group corresponds to each target, which contains a `uint32_t` count follows
 *     by corresponding numbers of `uint32_t` indices.
 *     When the count equals to the size of `jl_sysimg_fvars_offsets`, all of the functions
 *     are cloned so the indices array is omitted and no function will be filled into the GOT.
 *     The count is also needed to decode `jl_dispatch_fvars_offsets`.
 *     Each of the indices arrays are sorted.
 *     The highest bit of each indices is a tag bit. When it is set, the function pointer needs
 *     To be filled into the `jl_dispatch_got`.
 */

enum {
    JL_TARGET_VEC_CALL = 1 << 0,
    JL_TARGET_INCOMPLETE_FEATURE = 1 << 1,
    // Clone all functions
    JL_TARGET_CLONE_ALL = 1 << 2,
    // Clone when there's scalar math operations that can benefit from target specific
    // optimizations. This includes `muladd`, `fma`, `fast`/`contract` flags.
    JL_TARGET_CLONE_MATH = 1 << 3,
    // Clone when the function has a loop
    JL_TARGET_CLONE_LOOP = 1 << 4,
    // Clone when the function uses any vectors
    // When this is specified, the cloning pass should also record if any of the cloned functions
    // used this in any function call (including the signature of the function itself)
    JL_TARGET_CLONE_SIMD = 1 << 5,
};

typedef enum {
#define X86_FEATURE_DEF(name, bit, llvmver) JL_X86_##name = bit,
#include "features_x86.h"
#undef X86_FEATURE_DEF
} jl_cpu_feature_t;

int jl_test_cpu_feature(jl_cpu_feature_t feature);

typedef struct {
    // base function pointer
    const char *base;
    // number of functions
    uint32_t noffsets;
    // function pointer offsets
    const int32_t *offsets;

    // Following fields contains the information about the selected target.
    // All of these fields are 0 if the selected targets have all the functions cloned.
    // Instead the offsets are stored in `noffsets` and `offsets`.

    // number of cloned functions
    uint32_t nclones;
    // function pointer offsets of cloned functions
    const int32_t *clone_offsets;
    // sorted indices of the cloned functions (including the tag bit)
    const uint32_t *clone_idxs;
} jl_sysimg_fptrs_t;

/**
 * Initialize the processor dispatch system with sysimg `hdl` (also initialize the sysimg itself).
 * The dispatch system will find the best implementation to be used in this session.
 * The decision will be based on the host CPU and features as well as the `cpu_target`
 * option. This must be called before initializing JIT and should only be called once.
 * An error will be raised if this is called more than once or none of the implementation
 * supports the current system.
 *
 * Return the data about the function pointers selected.
 */
jl_sysimg_fptrs_t jl_init_processor_sysimg(void *hdl);

#if defined(_CPU_X86_) || defined(_CPU_X86_64_)
JL_DLLEXPORT void jl_cpuid(int32_t CPUInfo[4], int32_t InfoType);
JL_DLLEXPORT void jl_cpuidex(int32_t CPUInfo[4], int32_t InfoType, int32_t subInfoType);
#endif
// Return the name of the host CPU as a julia string.
JL_DLLEXPORT jl_value_t *jl_get_cpu_name(void);
// Dump the name and feature set of the host CPU
// For debugging only
JL_DLLEXPORT void jl_dump_host_cpu(void);

#ifdef __cplusplus
}

#include <utility>
#include <string>
#include <vector>

/**
 * Returns the CPU name and feature string to be used by LLVM JIT.
 *
 * If the detected/specified CPU name is not available on the LLVM version specified,
 * a fallback CPU name will be used. Unsupported features will be ignored.
 */
std::pair<std::string,std::vector<std::string>> jl_get_llvm_target(bool imaging, uint32_t llvmver);

/**
 * Returns the CPU name and feature string to be used by LLVM disassembler.
 *
 * This will return a generic CPU name and a full feature string.
 */
std::pair<std::string,std::string> jl_get_llvm_disasm_target(uint32_t llvmver);

struct jl_target_spec_t {
    // LLVM target name
    std::string cpu_name;
    // LLVM feature string
    std::string cpu_features;
    // serialized identification data
    std::vector<uint8_t> data;
    // Clone condition.
    uint32_t cond;
};
/**
 * Return the list of targets to clone
 */
std::vector<jl_target_spec_t> jl_get_llvm_clone_targets(uint32_t llvmver);
std::string jl_get_cpu_name_llvm(void);
#endif
