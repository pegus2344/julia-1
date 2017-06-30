// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "julia.h"
#include "julia_internal.h"

// Ref https://www.uclibc.org/docs/tls.pdf
// For variant 1 JL_ELF_TLS_INIT_SIZE is the size of the thread control block (TCB)
// For variant 2 JL_ELF_TLS_INIT_SIZE is 0
#ifdef _OS_LINUX_
#  if defined(_CPU_X86_64_) || defined(_CPU_X86_)
#    define JL_ELF_TLS_VARIANT 2
#    define JL_ELF_TLS_INIT_SIZE 0
#  elif defined(_CPU_AARCH64_)
#    define JL_ELF_TLS_VARIANT 1
#    define JL_ELF_TLS_INIT_SIZE 16
#  elif defined(__ARM_ARCH) && __ARM_ARCH >= 7
#    define JL_ELF_TLS_VARIANT 1
#    define JL_ELF_TLS_INIT_SIZE 8
#  endif
#endif

#ifdef JL_ELF_TLS_VARIANT
#  include <link.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "threading.h"

// The tls_states buffer:
//
// On platforms that do not use ELF (i.e. where `__thread` is emulated with
// lower level API) (Mac, Windows), we use the platform runtime API to create
// TLS variable directly.
// This is functionally equivalent to using `__thread` but can be
// more efficient since we can have better control over the creation and
// initialization of the TLS buffer.
//
// On platforms that use ELF (Linux, FreeBSD), we use a `__thread` variable
// as the fallback in the shared object. For better efficiency, we also
// create a `__thread` variable in the main executable using a static TLS
// model.
#ifdef JULIA_ENABLE_THREADING
#  if defined(_OS_DARWIN_)
// Mac doesn't seem to have static TLS model so the runtime TLS getter
// registration will only add overhead to TLS access. The `__thread` variables
// are emulated with `pthread_key_t` so it is actually faster to use it directly.
static pthread_key_t jl_tls_key;

__attribute__((constructor)) void jl_mac_init_tls(void)
{
    pthread_key_create(&jl_tls_key, NULL);
}

JL_DLLEXPORT JL_CONST_FUNC jl_ptls_t (jl_get_ptls_states)(void)
{
    void *ptls = pthread_getspecific(jl_tls_key);
    if (__unlikely(!ptls)) {
        ptls = calloc(1, sizeof(jl_tls_states_t));
        pthread_setspecific(jl_tls_key, ptls);
    }
    return (jl_ptls_t)ptls;
}

// This is only used after the tls is already initialized on the thread
static JL_CONST_FUNC jl_ptls_t jl_get_ptls_states_fast(void)
{
    return (jl_ptls_t)pthread_getspecific(jl_tls_key);
}

jl_get_ptls_states_func jl_get_ptls_states_getter(void)
{
    // for codegen
    return &jl_get_ptls_states_fast;
}
#  elif defined(_OS_WINDOWS_)
// Apparently windows doesn't have a static TLS model (or one that can be
// reliably used from a shared library) either..... Use `TLSAlloc` instead.

static DWORD jl_tls_key;

// Put this here for now. We can move this out later if we find more use for it.
BOOLEAN WINAPI DllMain(IN HINSTANCE hDllHandle, IN DWORD nReason,
                       IN LPVOID Reserved)
{
    switch (nReason) {
    case DLL_PROCESS_ATTACH:
        jl_tls_key = TlsAlloc();
        assert(jl_tls_key != TLS_OUT_OF_INDEXES);
        // Fall through
    case DLL_THREAD_ATTACH:
        TlsSetValue(jl_tls_key, calloc(1, sizeof(jl_tls_states_t)));
        break;
    case DLL_THREAD_DETACH:
        free(TlsGetValue(jl_tls_key));
        TlsSetValue(jl_tls_key, NULL);
        break;
    case DLL_PROCESS_DETACH:
        free(TlsGetValue(jl_tls_key));
        TlsFree(jl_tls_key);
        break;
    }
    return 1; // success
}

JL_DLLEXPORT JL_CONST_FUNC jl_ptls_t (jl_get_ptls_states)(void)
{
    return (jl_ptls_t)TlsGetValue(jl_tls_key);
}

jl_get_ptls_states_func jl_get_ptls_states_getter(void)
{
    // for codegen
    return &jl_get_ptls_states;
}
#  else
// We use the faster static version in the main executable to replace
// the slower version in the shared object. The code in different libraries
// or executables, however, have to agree on which version to use.
// The general solution is to add one more indirection in the C entry point
// (see `jl_get_ptls_states_wrapper`).
//
// When `ifunc` is available, we can use it to trick the linker to use the
// real address (`jl_get_ptls_states_static`) directly as the symbol address.
// (see `jl_get_ptls_states_resolve`).
//
// However, since the detection of the static version in `ifunc`
// is not guaranteed to be reliable, we still need to fallback to the wrapper
// version as the symbol address if we didn't find the static version in `ifunc`.

// fallback provided for embedding
static JL_CONST_FUNC jl_ptls_t jl_get_ptls_states_fallback(void)
{
    static __thread jl_tls_states_t tls_states;
    return &tls_states;
}
#if JL_USE_IFUNC
JL_DLLEXPORT JL_CONST_FUNC __attribute__((weak))
jl_ptls_t jl_get_ptls_states_static(void);
#endif
static jl_ptls_t jl_get_ptls_states_init(void);
static jl_get_ptls_states_func jl_tls_states_cb = jl_get_ptls_states_init;
static jl_ptls_t jl_get_ptls_states_init(void)
{
    // This 2-step initialization is used to detect calling
    // `jl_set_ptls_states_getter` after the address of the TLS variables
    // are used. Since the address of TLS variables should be constant,
    // changing the getter address can result in wierd crashes.

    // This is clearly not thread safe but should be fine since we
    // make sure the tls states callback is finalized before adding
    // multiple threads
    jl_get_ptls_states_func cb = jl_get_ptls_states_fallback;
#if JL_USE_IFUNC
    if (jl_get_ptls_states_static)
        cb = jl_get_ptls_states_static;
#endif
    jl_tls_states_cb = cb;
    return cb();
}

static JL_CONST_FUNC jl_ptls_t jl_get_ptls_states_wrapper(void)
{
    return (*jl_tls_states_cb)();
}

JL_DLLEXPORT void jl_set_ptls_states_getter(jl_get_ptls_states_func f)
{
    if (f == jl_tls_states_cb || !f)
        return;
    // only allow setting this once
    if (jl_tls_states_cb == jl_get_ptls_states_init) {
        jl_tls_states_cb = f;
    }
    else {
        jl_safe_printf("ERROR: Attempt to change TLS address.\n");
        exit(1);
    }
}

#if JL_USE_IFUNC
static jl_get_ptls_states_func jl_get_ptls_states_resolve(void)
{
    if (jl_tls_states_cb != jl_get_ptls_states_init)
        return jl_tls_states_cb;
    // If we can't find the static version, return the wrapper instead
    // of the slow version so that we won't resolve to the slow version
    // due to issues in the relocation order.
    // This may not be necessary once `ifunc` support in glibc is more mature.
    if (!jl_get_ptls_states_static)
        return jl_get_ptls_states_wrapper;
    jl_tls_states_cb = jl_get_ptls_states_static;
    return jl_tls_states_cb;
}

JL_DLLEXPORT JL_CONST_FUNC jl_ptls_t (jl_get_ptls_states)(void)
    __attribute__((ifunc ("jl_get_ptls_states_resolve")));
#else // JL_TLS_USE_IFUNC
JL_DLLEXPORT JL_CONST_FUNC jl_ptls_t (jl_get_ptls_states)(void)
{
    return jl_get_ptls_states_wrapper();
}
#endif // JL_TLS_USE_IFUNC
jl_get_ptls_states_func jl_get_ptls_states_getter(void)
{
    if (jl_tls_states_cb == jl_get_ptls_states_init)
        jl_get_ptls_states_init();
    // for codegen
    return jl_tls_states_cb;
}
#  endif
#else
JL_DLLEXPORT jl_tls_states_t jl_tls_states;
JL_DLLEXPORT JL_CONST_FUNC jl_ptls_t (jl_get_ptls_states)(void)
{
    return &jl_tls_states;
}
#endif

JL_DLLEXPORT int jl_n_threads;
jl_ptls_t *jl_all_tls_states;

// return calling thread's ID
// Also update the suspended_threads list in signals-mach when changing the
// type of the thread id.
JL_DLLEXPORT int16_t jl_threadid(void)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    return ptls->tid;
}

void jl_init_threadtls(int16_t tid)
{
    jl_ptls_t ptls = jl_get_ptls_states();
#ifdef _OS_WINDOWS_
    if (tid == 0) {
        if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                             GetCurrentProcess(), &hMainThread, 0,
                             FALSE, DUPLICATE_SAME_ACCESS)) {
            jl_printf(JL_STDERR, "WARNING: failed to access handle to main thread\n");
            hMainThread = INVALID_HANDLE_VALUE;
        }
    }
#else
    ptls->system_id = pthread_self();
#endif
    ptls->tid = tid;
    ptls->pgcstack = NULL;
    ptls->gc_state = 0; // GC unsafe
    // Conditionally initialize the safepoint address. See comment in
    // `safepoint.c`
    if (tid == 0) {
        ptls->safepoint = (size_t*)(jl_safepoint_pages + jl_page_size);
    }
    else {
        ptls->safepoint = (size_t*)(jl_safepoint_pages + jl_page_size * 2 +
                                    sizeof(size_t));
    }
    ptls->defer_signal = 0;
    ptls->current_module = NULL;
    void *bt_data = malloc(sizeof(uintptr_t) * (JL_MAX_BT_SIZE + 1));
    if (bt_data == NULL) {
        jl_printf(JL_STDERR, "could not allocate backtrace buffer\n");
        gc_debug_critical_error();
        abort();
    }
    ptls->bt_data = (uintptr_t*)bt_data;
    jl_init_thread_heap(ptls);
    jl_install_thread_signal_handler(ptls);

    jl_all_tls_states[tid] = ptls;
}

// all threads call this function to run user code
jl_value_t *jl_thread_run_fun(const jl_generic_fptr_t *fptr, jl_method_instance_t *mfunc,
                              jl_value_t **args, uint32_t nargs)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    JL_TRY {
        (void)jl_assume(fptr->jlcall_api != 2);
        jl_call_fptr_internal(fptr, mfunc, args, nargs);
    }
    JL_CATCH {
        // Lock this output since we know it'll likely happen on multiple threads
        static jl_mutex_t lock;
        JL_LOCK_NOGC(&lock);
        jl_jmp_buf *old_buf = ptls->safe_restore;
        jl_jmp_buf buf;
        if (!jl_setjmp(buf, 0)) {
            // Set up the safe_restore context so that the printing uses the thread safe version
            ptls->safe_restore = &buf;
            jl_printf(JL_STDERR, "\nError thrown in threaded loop on thread %d: ",
                      (int)ptls->tid);
            jl_static_show(JL_STDERR, ptls->exception_in_transit);
        }
        ptls->safe_restore = old_buf;
        JL_UNLOCK_NOGC(&lock);
    }
    return jl_nothing;
}

// lock for code generation
jl_mutex_t codegen_lock;
jl_mutex_t typecache_lock;

#ifdef JULIA_ENABLE_THREADING

ssize_t jl_tls_offset = -1;

#ifdef JL_ELF_TLS_VARIANT
const int jl_tls_elf_support = 1;
// Optimize TLS access in codegen if the TLS buffer is using a IE or LE model.
// To detect such case, we find the size of the TLS segment in the main
// executable and the thread pointer (TP) and then see if the TLS pointer on the
// current thread is in the right range.
// This can in principle be extended to the case where the TLS buffer is
// in the shared library but is part of the static buffer but that seems harder
// to detect.
#  if JL_ELF_TLS_VARIANT == 1
// In Variant 1, the static TLS buffer comes after a fixed size TCB.
// The alignment needs to be applied to the original size.
static inline size_t jl_add_tls_size(size_t orig_size, size_t size, size_t align)
{
    return LLT_ALIGN(orig_size, align) + size;
}
static inline ssize_t jl_check_tls_bound(void *tp, void *ptls, size_t tls_size)
{
    ssize_t offset = (char*)ptls - (char*)tp;
    if (offset < JL_ELF_TLS_INIT_SIZE ||
        (size_t)offset + sizeof(jl_tls_states_t) > tls_size)
        return -1;
    return offset;
}
#  elif JL_ELF_TLS_VARIANT == 2
// In Variant 2, the static TLS buffer comes before a unknown size TCB.
// The alignment needs to be applied to the new size.
static inline size_t jl_add_tls_size(size_t orig_size, size_t size, size_t align)
{
    return LLT_ALIGN(orig_size + size, align);
}
static inline ssize_t jl_check_tls_bound(void *tp, void *ptls, size_t tls_size)
{
    ssize_t offset = (char*)tp - (char*)ptls;
    if (offset < sizeof(jl_tls_states_t) || offset > tls_size)
        return -1;
    return -offset;
}
#  else
#    error "Unknown static TLS variant"
#  endif

// Find the size of the TLS segment in the main executable
typedef struct {
    size_t total_size;
} check_tls_cb_t;

static int check_tls_cb(struct dl_phdr_info *info, size_t size, void *_data)
{
    check_tls_cb_t *data = (check_tls_cb_t*)_data;
    const ElfW(Phdr) *phdr = info->dlpi_phdr;
    unsigned phnum = info->dlpi_phnum;
    size_t total_size = JL_ELF_TLS_INIT_SIZE;

    for (unsigned i = 0; i < phnum; i++) {
        const ElfW(Phdr) *seg = &phdr[i];
        if (seg->p_type != PT_TLS)
            continue;
        // There should be only one TLS segment
        // Variant II
        total_size = jl_add_tls_size(total_size, seg->p_memsz, seg->p_align);
    }
    data->total_size = total_size;
    // only run once (on the main executable)
    return 1;
}

static void jl_check_tls(void)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    check_tls_cb_t data = {0};
    dl_iterate_phdr(check_tls_cb, &data);
    if (data.total_size == 0)
        return;
    void *tp; // Thread pointer
#if defined(_CPU_X86_64_)
    asm("movq %%fs:0, %0" : "=r"(tp));
#elif defined(_CPU_X86_)
    asm("movl %%gs:0, %0" : "=r"(tp));
#elif defined(_CPU_AARCH64_)
    asm("mrs %0, tpidr_el0" : "=r"(tp));
#elif defined(__ARM_ARCH) && __ARM_ARCH >= 7
    asm("mrc p15, 0, %0, c13, c0, 3" : "=r"(tp));
#else
#  error "Cannot emit thread pointer for this architecture."
#endif
    ssize_t offset = jl_check_tls_bound(tp, ptls, data.total_size);
    if (offset == -1)
        return;
    jl_tls_offset = offset;
}
#else
const int jl_tls_elf_support = 0;
#endif

// interface to Julia; sets up to make the runtime thread-safe
void jl_init_threading(void)
{
    char *cp;

#ifdef JL_ELF_TLS_VARIANT
    jl_check_tls();
#endif

    // how many threads available, usable
    int max_threads = jl_cpu_cores();
    jl_n_threads = JULIA_NUM_THREADS;
    cp = getenv(NUM_THREADS_NAME);
    if (cp)
        jl_n_threads = (uint64_t)strtol(cp, NULL, 10);
    if (jl_n_threads > max_threads)
        jl_n_threads = max_threads;
    if (jl_n_threads <= 0)
        jl_n_threads = 1;

    jl_all_tls_states = (jl_ptls_t*)malloc(jl_n_threads * sizeof(void*));

    // initialize this thread (set tid, create heap, etc.)
    jl_init_threadtls(0);
}

static uv_barrier_t thread_init_done;

void jl_start_threads(void)
{
    char *cp, mask[UV_CPU_SETSIZE];
    int i, exclusive;
    uv_thread_t uvtid;
    jl_threadarg_t **targs;

    // do we have exclusive use of the machine? default is no
    exclusive = DEFAULT_MACHINE_EXCLUSIVE;
    cp = getenv(MACHINE_EXCLUSIVE_NAME);
    if (cp)
        exclusive = strtol(cp, NULL, 10);

    // exclusive use: affinitize threads, master thread on proc 0, rest
    // according to a 'compact' policy
    // non-exclusive: no affinity settings; let the kernel move threads about
    if (exclusive) {
        memset(mask, 0, UV_CPU_SETSIZE);
        mask[0] = 1;
        uvtid = (uv_thread_t)uv_thread_self();
        uv_thread_setaffinity(&uvtid, mask, NULL, UV_CPU_SETSIZE);
    }

    // initialize threading infrastructure
    jl_init_threadinginfra();

    // create threads
    targs = (jl_threadarg_t **)malloc((jl_n_threads - 1) * sizeof (jl_threadarg_t *));

    uv_barrier_init(&thread_init_done, jl_n_threads);

    for (i = 0;  i < jl_n_threads - 1;  ++i) {
        targs[i] = (jl_threadarg_t *)malloc(sizeof (jl_threadarg_t));
        targs[i]->tid = i + 1;
        targs[i]->barrier = &thread_init_done;
        jl_init_threadarg(targs[i]);
        uv_thread_create(&uvtid, jl_threadfun, targs[i]);
        if (exclusive) {
            memset(mask, 0, UV_CPU_SETSIZE);
            mask[i+1] = 1;
            uv_thread_setaffinity(&uvtid, mask, NULL, UV_CPU_SETSIZE);
        }
        uv_thread_detach(&uvtid);
    }

    jl_init_started_threads(targs);

    uv_barrier_wait(&thread_init_done);

    // free the argument array; the threads will free their arguments
    free(targs);
}

#else // !JULIA_ENABLE_THREADING

void jl_init_threading(void)
{
    static jl_ptls_t _jl_all_tls_states;
    jl_all_tls_states = &_jl_all_tls_states;
    jl_n_threads = 1;
    jl_init_threadtls(0);
}

void jl_start_threads(void) { }

JL_DLLEXPORT jl_value_t *jl_threading_run(jl_value_t *_args)
{
    uint32_t nargs;
    jl_value_t **args;
    if (!jl_is_svec(_args)) {
        nargs = 1;
        args = &_args;
    }
    else {
        nargs = jl_svec_len(_args);
        args = jl_svec_data(_args);
    }
    jl_method_instance_t *mfunc = jl_lookup_generic(args, nargs,
                                                    jl_int32hash_fast(jl_return_address()),
                                                    jl_get_ptls_states()->world_age);
    jl_generic_fptr_t fptr;
    if (jl_compile_method_internal(&fptr, mfunc))
        return jl_nothing;
    return jl_thread_run_fun(&fptr, mfunc, args, nargs);
}

#endif // !JULIA_ENABLE_THREADING

// Make gc alignment available for threading
// see threads.jl alignment
JL_DLLEXPORT int jl_alignment(size_t sz)
{
    return jl_gc_alignment(sz);
}

#ifdef __cplusplus
}
#endif
