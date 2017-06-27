// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "julia.h"
#include "julia_internal.h"
#include "threading.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef JULIA_ENABLE_THREADING
#ifdef JULIA_ENABLE_PARTR

/* the `start` task */
ptask_t *start_task;

/* sticky task queues need to be visible to all threads */
ptask_t  ***all_taskqs;
int8_t   **all_taskq_locks;

/* forward declare thread function */
void *partr_thread(void *arg_);

/* internally used to indicate a yield occurred in the runtime itself */
static const int64_t yield_from_sync = 1;


// initialize the threading infrastructure
void jl_init_threadinginfra(void)
{
    /* initialize the synchronization trees pool and the multiqueue */
    synctreepool_init();
    multiq_init();

    /* allocate per-thread task queues, for sticky tasks */
    all_taskqs = (ptask_t ***)_mm_malloc(jl_n_threads * sizeof(ptask_t **), 64);
    all_taskq_locks = (int8_t **)_mm_malloc(jl_n_threads * sizeof(int8_t *), 64);
}


// initialize the thread function argument
void jl_init_threadarg(jl_threadarg_t *targ) { }


// helper for final thread initialization
static void init_started_thread()
{
    /* allocate this thread's sticky task queue pointer and initialize the lock */
    jl_ptls_t ptls = jl_get_ptls_states();
    seed_cong(&ptls->rngseed);
    ptls->sticky_taskq_lock = (int8_t *)_mm_malloc(sizeof(int8_t) + sizeof(ptask_t *), 64);
    ptls->sticky_taskq = (ptask_t **)(taskq_lock + sizeof(int8_t));
    jl_atomic_clear(taskq_lock);
    *ptls->sticky_taskq = NULL;
    all_taskqs[ptls->tid] = ptls->sticky_taskq;
    all_taskq_locks[ptls->tid] = ptls->sticky_taskq_lock;
}


// once the threads are started, perform any final initializations
void jl_init_started_threads(jl_threadarg_t **targs)
{
    // master thread final initialization
    init_started_thread();
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

static void *partr_thread(void *arg_)
{
    BARRIER_THREAD_DECL;
    lthread_arg_t *arg = (lthread_arg_t *)arg_;

    tid = arg->tid;
    seed_cong(&rngseed);

    /* set affinity if requested */
    if (arg->exclusive) {
        hwloc_set_cpubind(arg->topology, arg->cpuset, HWLOC_CPUBIND_THREAD);
        hwloc_bitmap_free(arg->cpuset);
    }
    show_affinity();

    /* allocate this thread's sticky task queue pointer and initialize the lock */
    taskq_lock = (int8_t *)_mm_malloc(sizeof(int8_t) + sizeof(ptask_t *), 64);
    taskq = (ptask_t **)(taskq_lock + sizeof(int8_t));
    __atomic_clear(taskq_lock, __ATOMIC_RELAXED);
    *taskq = NULL;
    all_taskqs[tid] = taskq;
    all_taskq_locks[tid] = taskq_lock;

    BARRIER();

    /* free the thread function argument */
    free(arg);

    /* get the highest priority task and run it */
    while (run_next() == 0)
        ;

    /* free the sticky task queue pointer (and its lock) */
    _mm_free(taskq_lock);

    return NULL;
}



// old threading interface: run specified function in all threads. partr created
// jl_n_threads tasks and enqueues them; these may not actually run in all the
// threads.
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
    tw->ret = ti_run_fun(&threadwork.fptr, threadwork.mfunc, args, nargs);

    // wait for completion
    ti_threadgroup_join(tgworld, ptls->tid);

    jl_gc_unsafe_leave(ptls, gc_state);

    return tw->ret;
}


// coroutine entry point
static void partr_coro(void *ctx)
{
    ptask_t *task = (ptask_t *)ctx; // TODO. ctx_get_user_ptr(ctx);
    task->result = task->f(task->arg, task->start, task->end);

    /* grain tasks must synchronize */
    if (task->grain_num >= 0) {
        int was_last = 0;

        /* reduce... */
        if (task->red) {
            task->result = reduce(task->arr, task->red, task->rf,
                                  task->result, task->grain_num);
            /*  if this task is last, set the result in the parent task */
            if (task->result) {
                task->parent->red_result = task->result;
                was_last = 1;
            }
        }
        /* ... or just sync */
        else {
            if (last_arriver(task->arr, task->grain_num))
                was_last = 1;
        }

        /* the last task to finish needs to finish up the loop */
        if (was_last) {
            /* a non-parent task must wake up the parent */
            if (task->grain_num > 0) {
                multiq_insert(task->parent, 0);
            }
            /* the parent task was last; it can just end */
        }
        else {
            /* the parent task needs to wait */
            if (task->grain_num == 0) {
                yield_value(task->ctx, (void *)yield_from_sync);
            }
        }
    }
}


// allocate and initialize a task
static ptask_t *setup_task(void *(*f)(void *, int64_t, int64_t), void *arg,
        int64_t start, int64_t end)
{
    ptask_t *task = NULL; // TODO. task_alloc();
    if (task == NULL)
        return NULL;

    // TODO. ctx_construct(task->ctx, task->stack, TASK_STACK_SIZE, partr_coro, task);
    task->f = f;
    task->arg = arg;
    task->start = start;
    task->end = end;
    task->sticky_tid = -1;
    task->grain_num = -1;

    return task;
}


// free a task
static void *release_task(ptask_t *task)
{
    void *result = task->result;
    // TODO. ctx_destruct(task->ctx);
    if (task->grain_num == 0  &&  task->red)
        reducer_free(task->red);
    if (task->grain_num == 0  &&  task->arr)
        arriver_free(task->arr);
    task->f = NULL;
    task->arg = task->result = task->red_result = NULL;
    task->start = task->end = 0;
    task->rf = NULL;
    task->parent = task->cq = NULL;
    task->arr = NULL;
    task->red = NULL;
    // TODO. task_free(task);
    return result;
}


// add the specified task to the sticky task queue
static void add_to_taskq(ptask_t *task)
{
    assert(task->sticky_tid != -1);

    ptask_t **q = all_taskqs[task->sticky_tid];
    int8_t *lock = all_taskq_locks[task->sticky_tid];

    while (jl_atomic_test_and_set(lock))
        cpu_pause();

    if (*q == NULL)
        *q = task;
    else {
        ptask_t *pt = *q;
        while (pt->next)
            pt = pt->next;
        pt->next = task;
    }

    jl_atomic_clear(lock);
}


// pop the first task off the sticky task queue
static ptask_t *get_from_taskq()
{
    jl_ptls_t ptls = jl_get_ptls_states();

    /* racy check for quick path */
    if (*ptls->sticky_taskq == NULL)
        return NULL;

    while (jl_atomic_test_and_set(ptls->sticky_taskq_lock))
        cpu_pause();

    ptask_t *task = *ptls->sticky_taskq;
    if (task) {
        *ptls->sticky_taskq = task->next;
        task->next = NULL;
    }

    jl_atomic_clear(ptls->sticky_taskq_lock);

    return task;
}


// get the next available task and run it
static int run_next()
{
    ptask_t *task;

    /* first check for sticky tasks */
    task = get_from_taskq();

    /* no sticky tasks, go to the multiq */
    if (task == NULL) {
        task = multiq_deletemin();
        if (task == NULL)
            return 0;

        /* a sticky task will only come out of the multiq if it has not been run */
        if (task->settings & TASK_IS_STICKY) {
            assert(task->sticky_tid == -1);
            task->sticky_tid = tid;
        }
    }

    /* run/resume the task */
    jl_ptls_t ptls = jl_get_ptls_states();
    ptls->curr_task = task;
    int64_t y = (int64_t)resume(task->ctx);
    ptls->curr_task = NULL;

    /* if the task isn't done, it is either in a CQ, or must be re-queued */
    if (!ctx_is_done(task->ctx)) {
        /* the yield value tells us if the task is in a CQ */
        if (y != yield_from_sync) {
            /* sticky tasks go to the thread's sticky queue */
            if (task->settings & TASK_IS_STICKY)
                add_to_taskq(task);
            /* all others go back into the multiq */
            else
                multiq_insert(task, task->prio);
        }
        return 0;
    }

    /* The task completed. As detached tasks cannot be synced, clean
       those up here.
     */
    if (task->settings & TASK_IS_DETACHED) {
        release_task(task);
        return 0;
    }

    /* add back all the tasks in this one's completion queue */
    while (__atomic_test_and_set(&task->cq_lock, __ATOMIC_ACQUIRE))
        cpu_pause();
    ptask_t *cqtask, *cqnext;
    cqtask = task->cq;
    task->cq = NULL;
    while (cqtask) {
        cqnext = cqtask->next;
        cqtask->next = NULL;
        if (cqtask->settings & TASK_IS_STICKY)
            add_to_taskq(cqtask);
        else
            multiq_insert(cqtask, cqtask->prio);
        cqtask = cqnext;
    }
    __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);

    return 0;
}


/*  partr_start() -- the runtime entry point

    To be called from thread 0, before creating any tasks. Wraps into
    a task and invokes `f(arg)`; tasks should only be spawned/synced
    from within tasks.
 */
int partr_start(void **ret, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end)
{
    assert(tid == 0);

    start_task = setup_task(f, arg, start, end);
    if (start_task == NULL)
        return -1;
    start_task->settings |= TASK_IS_STICKY;
    start_task->sticky_tid = tid;

    jl_ptls_t ptls = jl_get_ptls_states();
    ptls->curr_task = start_task;
    int64_t y = (int64_t)resume(start_task->ctx);
    ptls->curr_task = NULL;

    if (!ctx_is_done(start_task->ctx)) {
        if (y != yield_from_sync) {
            add_to_taskq(start_task);
        }
        while (run_next() == 0)
            if (ctx_is_done(start_task->ctx))
                break;
    }

    void *r = release_task(start_task);
    if (ret)
        *ret = r;

    return 0;
}


/*  partr_spawn() -- create a task for `f(arg)` and enqueue it for execution

    Implicitly asserts that `f(arg)` can run concurrently with everything
    else that's currently running. If `detach` is set, the spawned task
    will not be returned (and cannot be synced). Yields.
 */
int partr_spawn(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t start, int64_t end, int8_t sticky, int8_t detach)
{
    ptask_t *task = setup_task(f, arg, start, end);
    if (task == NULL)
        return -1;
    if (sticky)
        task->settings |= TASK_IS_STICKY;
    if (detach)
        task->settings |= TASK_IS_DETACHED;

    if (multiq_insert(task, tid) != 0) {
        release_task(task);
        return -2;
    }

    *t = detach ? NULL : (partr_t)task;

    /* only yield if we're running a non-sticky task */
    jl_ptls_t ptls = jl_get_ptls_states();
    if (!(ptls->curr_task->settings & TASK_IS_STICKY))
        yield(ptls->curr_task->ctx);

    return 0;
}


/*  partr_sync() -- get the return value of task `t`

    Returns only when task `t` has completed.
 */
int partr_sync(void **r, partr_t t, int done_with_task)
{
    ptask_t *task = (ptask_t *)t;

    /* if the target task has not finished, add the current task to its
       completion queue; the thread that runs the target task will add
       this task back to the ready queue
     */
    if (!ctx_is_done(task->ctx)) {
        curr_task->next = NULL;
        while (__atomic_test_and_set(&task->cq_lock, __ATOMIC_ACQUIRE))
            cpu_pause();

        /* ensure the task didn't finish before we got the lock */
        if (!ctx_is_done(task->ctx)) {
            /* add the current task to the CQ */
            if (task->cq == NULL)
                task->cq = curr_task;
            else {
                ptask_t *pt = task->cq;
                while (pt->next)
                    pt = pt->next;
                pt->next = curr_task;
            }

            /* unlock the CQ and yield the current task */
            __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);
            yield_value(curr_task->ctx, (void *)yield_from_sync);
        }

        /* the task finished before we could add to its CQ */
        else {
            __atomic_clear(&task->cq_lock, __ATOMIC_RELEASE);
        }
    }

    if (r)
        *r = task->grain_num >= 0 && task->red ?
                task->red_result : task->result;

    if (done_with_task)
        release_task(task);

    return 0;
}


/*  partr_parfor() -- spawn multiple tasks for a parallel loop

    Spawn tasks that invoke `f(arg, start, end)` such that the sum of `end-start`
    for all tasks is `count`. Uses `rf()`, if provided, to reduce the return
    values from the tasks, and returns the result. Yields.
 */
int partr_parfor(partr_t *t, void *(*f)(void *, int64_t, int64_t),
        void *arg, int64_t count, void *(*rf)(void *, void *))
{
    int64_t n = GRAIN_K * nthreads;
    lldiv_t each = lldiv(count, n);

    /* allocate synchronization tree(s) */
    arriver_t *arr = arriver_alloc();
    if (arr == NULL)
        return -1;
    reducer_t *red = NULL;
    if (rf != NULL) {
        red = reducer_alloc();
        if (red == NULL) {
            arriver_free(arr);
            return -2;
        }
    }

    /* allocate and enqueue (GRAIN_K * nthreads) tasks */
    *t = NULL;
    int64_t start = 0, end;
    for (int64_t i = 0;  i < n;  ++i) {
        end = start + each.quot + (i < each.rem ? 1 : 0);
        ptask_t *task = setup_task(f, arg, start, end);
        if (task == NULL)
            return -1;

        /* The first task is the parent (root) task of the parfor, thus only
           this can be synced. So, we create the remaining tasks detached.
         */
        if (*t == NULL) *t = task;
        else task->settings = TASK_IS_DETACHED;

        task->parent = *t;
        task->grain_num = i;
        task->rf = rf;
        task->arr = arr;
        task->red = red;

        if (multiq_insert(task, tid) != 0) {
            release_task(task);
            return -3;
        }

        start = end;
    }

    /* only yield if we're running a non-sticky task */
    if (!(curr_task->settings & TASK_IS_STICKY))
        yield(curr_task->ctx);

    return 0;
}


#endif // JULIA_ENABLE_PARTR
#endif // JULIA_ENABLE_THREADING

#ifdef __cplusplus
}
#endif
