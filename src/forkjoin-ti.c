// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "julia.h"
#include "julia_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "threading.h"
#include "threadgroup.h"

#ifdef JULIA_ENABLE_THREADING
#ifdef JULIA_ENABLE_FORKJOIN_TI

// thread state
enum {
    TI_THREAD_INIT,
    TI_THREAD_WORK
};

// passed to thread function
typedef struct {
    int16_t volatile state;
    ti_threadgroup_t *tg;
} ti_threadarg_t;

// work command to thread function
typedef struct {
    jl_method_instance_t *mfunc;
    jl_generic_fptr_t fptr;
    jl_value_t **args;
    uint32_t nargs;
    jl_value_t *ret;
    jl_module_t *current_module;
    size_t world_age;
} ti_threadwork_t;

// for broadcasting work to threads
static ti_threadwork_t threadwork;

// only one thread group for now
static ti_threadgroup_t *tgworld;

void jl_init_threadinginfra(void) { }

void jl_init_threadarg(jl_threadarg_t *targ)
{
    ti_threadarg_t *tiarg = (ti_threadarg_t *)malloc(sizeof (ti_threadarg_t));
    tiarg->state = TI_THREAD_INIT;
    targ->arg = (void *)tiarg;
}

void jl_init_started_threads(jl_threadarg_t **targs)
{
    // set up the world thread group
    ti_threadgroup_create(1, jl_n_threads, 1, &tgworld);
    for (int i = 0;  i < jl_n_threads;  ++i)
        ti_threadgroup_addthread(tgworld, i, NULL);

    jl_ptls_t ptls = jl_get_ptls_states();
    ti_threadgroup_initthread(tgworld, ptls->tid);

    // give the threads the world thread group; they will block waiting for fork
    for (int i = 0;  i < jl_n_threads - 1;  ++i) {
        ti_threadarg_t *tiarg = (ti_threadarg_t *)targs[i]->arg;
        tiarg->tg = tgworld;
        jl_atomic_store_release(&tiarg->state, TI_THREAD_WORK);
    }
}

// thread function: used by all except the main thread
void jl_threadfun(void *arg)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_threadarg_t *targ = (jl_threadarg_t *)arg;
    ti_threadarg_t *tiarg = (ti_threadarg_t *)targ->arg;
    ti_threadgroup_t *tg;
    ti_threadwork_t *work;

    // initialize this thread (set tid, create heap, etc.)
    jl_init_threadtls(targ->tid);
    jl_init_stack_limits(0);

    // set up tasking
    jl_init_root_task(ptls->stack_lo, ptls->stack_hi - ptls->stack_lo);
#ifdef COPY_STACKS
    jl_set_base_ctx((char*)&arg);
#endif

    // set the thread-local tid and wait for a thread group
    while (jl_atomic_load_acquire(&tiarg->state) == TI_THREAD_INIT)
        jl_cpu_pause();

    // Assuming the functions called below doesn't contain unprotected GC
    // critical region. In general, the following part of this function
    // shouldn't call any managed code without calling `jl_gc_unsafe_enter`
    // first.
    jl_gc_state_set(ptls, JL_GC_STATE_SAFE, 0);
    uv_barrier_wait(targ->barrier);

    // initialize this thread in the thread group
    tg = tiarg->tg;
    ti_threadgroup_initthread(tg, ptls->tid);

    // free the thread argument here
    free(tiarg);
    free(targ);

    int init = 1;

    // work loop
    for (; ;) {
        ti_threadgroup_fork(tg, ptls->tid, (void **)&work, init);
        init = 0;

        if (work) {
            // TODO: before we support getting return value from
            //       the work, and after we have proper GC transition
            //       support in the codegen and runtime we don't need to
            //       enter GC unsafe region when starting the work.
            int8_t gc_state = jl_gc_unsafe_enter(ptls);
            // This is probably always NULL for now
            jl_module_t *last_m = ptls->current_module;
            size_t last_age = ptls->world_age;
            JL_GC_PUSH1(&last_m);
            ptls->current_module = work->current_module;
            ptls->world_age = work->world_age;
            jl_thread_run_fun(&work->fptr, work->mfunc, work->args, work->nargs);
            ptls->current_module = last_m;
            ptls->world_age = last_age;
            JL_GC_POP();
            jl_gc_unsafe_leave(ptls, gc_state);
        }

        ti_threadgroup_join(tg, ptls->tid);
    }
}

// interface to user code: specialize and compile the user thread function
// and run it in all threads
JL_DLLEXPORT jl_value_t *jl_threading_run(jl_value_t *_args)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    // GC safe
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

    int8_t gc_state = jl_gc_unsafe_enter(ptls);

    threadwork.mfunc = jl_lookup_generic(args, nargs,
                                         jl_int32hash_fast(jl_return_address()), ptls->world_age);
    // Ignore constant return value for now.
    if (jl_compile_method_internal(&threadwork.fptr, threadwork.mfunc))
        return jl_nothing;
    threadwork.args = args;
    threadwork.nargs = nargs;
    threadwork.ret = jl_nothing;
    threadwork.current_module = ptls->current_module;
    threadwork.world_age = ptls->world_age;

    // fork the world thread group
    ti_threadwork_t *tw = &threadwork;
    ti_threadgroup_fork(tgworld, ptls->tid, (void **)&tw, 0);

    // this thread must do work too
    tw->ret = jl_thread_run_fun(&threadwork.fptr, threadwork.mfunc, args, nargs);

    // wait for completion
    ti_threadgroup_join(tgworld, ptls->tid);

    jl_gc_unsafe_leave(ptls, gc_state);

    return tw->ret;
}

#endif // JULIA_ENABLE_FORKJOIN_TI
#endif // JULIA_ENABLE_THREADING

#ifdef __cplusplus
}
#endif
