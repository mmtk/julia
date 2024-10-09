// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifdef MMTK_GC

#include "gc.h"
#include "mmtk_julia.h"
#include "julia_gcext.h"

// callbacks
// ---

typedef void (*jl_gc_cb_func_t)(void);

JL_DLLEXPORT void jl_gc_set_cb_root_scanner(jl_gc_cb_root_scanner_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_task_scanner(jl_gc_cb_task_scanner_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_pre_gc(jl_gc_cb_pre_gc_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_post_gc(jl_gc_cb_post_gc_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_notify_external_alloc(jl_gc_cb_notify_external_alloc_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_notify_external_free(jl_gc_cb_notify_external_free_t cb, int enable)
{
}
JL_DLLEXPORT void jl_gc_set_cb_notify_gc_pressure(jl_gc_cb_notify_gc_pressure_t cb, int enable)
{
}
JL_DLLEXPORT int64_t jl_gc_pool_live_bytes(void)
{
    mmtk_unreachable();
    return 0;
}

inline void maybe_collect(jl_ptls_t ptls)
{
    // Just do a safe point for general maybe_collect
    jl_gc_safepoint_(ptls);
}

// This is only used for malloc. We need to know if we need to do GC. However, keeping checking with MMTk (mmtk_gc_poll),
// is expensive. So we only check for every few allocations.
static inline void malloc_maybe_collect(jl_ptls_t ptls, size_t sz)
{
    // We do not need to carefully maintain malloc_sz_since_last_poll. We just need to
    // avoid using mmtk_gc_poll too frequently, and try to be precise on our heap usage
    // as much as we can.
    if (ptls->malloc_sz_since_last_poll > 4096) {
        jl_atomic_store_relaxed(&ptls->malloc_sz_since_last_poll, 0);
        mmtk_gc_poll(ptls);
    } else {
        jl_atomic_fetch_add_relaxed(&ptls->malloc_sz_since_last_poll, sz);
        jl_gc_safepoint_(ptls);
    }
}

// allocation
int jl_gc_classify_pools(size_t sz, int *osize)
{
    if (sz > GC_MAX_SZCLASS)
        return -1; // call big alloc function
    size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    *osize = LLT_ALIGN(allocsz, 16);
    return 0; // use MMTk's fastpath logic
}

// malloc wrappers, aligned allocation
// We currently just duplicate what Julia GC does. We will in the future replace the malloc calls with MMTK's malloc.
#if defined(_OS_WINDOWS_)
inline void *jl_malloc_aligned(size_t sz, size_t align)
{
    return _aligned_malloc(sz ? sz : 1, align);
}
inline void *jl_realloc_aligned(void *p, size_t sz, size_t oldsz,
                                       size_t align)
{
    (void)oldsz;
    return _aligned_realloc(p, sz ? sz : 1, align);
}
inline void jl_free_aligned(void *p) JL_NOTSAFEPOINT
{
    _aligned_free(p);
}
#else
inline void *jl_malloc_aligned(size_t sz, size_t align)
{
#if defined(_P64) || defined(__APPLE__)
    if (align <= 16)
        return malloc(sz);
#endif
    void *ptr;
    if (posix_memalign(&ptr, align, sz))
        return NULL;
    return ptr;
}
inline void *jl_realloc_aligned(void *d, size_t sz, size_t oldsz,
                                       size_t align)
{
#if defined(_P64) || defined(__APPLE__)
    if (align <= 16)
        return realloc(d, sz);
#endif
    void *b = jl_malloc_aligned(sz, align);
    if (b != NULL) {
        memcpy(b, d, oldsz > sz ? sz : oldsz);
        free(d);
    }
    return b;
}
inline void jl_free_aligned(void *p) JL_NOTSAFEPOINT
{
    free(p);
}
#endif

void jl_gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT
{
    if (a->flags.how == 2) {
        char *d = (char*)a->data - a->offset*a->elsize;
        if (a->flags.isaligned)
            jl_free_aligned(d);
        else
            free(d);
        gc_num.freed += jl_array_nbytes(a);
        gc_num.freecall++;
    }
}

// weak references
// ---

void clear_weak_refs(void)
{
    assert(gc_n_threads);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = gc_all_tls_states[i];
        if (ptls2 != NULL) {
            size_t n, l = ptls2->heap.weak_refs.len;
            void **lst = ptls2->heap.weak_refs.items;
            for (n = 0; n < l; n++) {
                jl_weakref_t *wr = (jl_weakref_t*)lst[n];
                if (!mmtk_is_live_object(wr->value))
                    wr->value = (jl_value_t*)jl_nothing;
            }
        }
    }
}

void sweep_weak_refs(void)
{
    assert(gc_n_threads);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = gc_all_tls_states[i];
        if (ptls2 != NULL) {
            size_t n = 0;
            size_t ndel = 0;
            size_t l = ptls2->heap.weak_refs.len;
            void **lst = ptls2->heap.weak_refs.items;
            if (l == 0)
                continue;
            while (1) {
                jl_weakref_t *wr = (jl_weakref_t*)lst[n];
                if (mmtk_is_live_object(wr))
                    n++;
                else
                    ndel++;
                if (n >= l - ndel)
                    break;
                void *tmp = lst[n];
                lst[n] = lst[n + ndel];
                lst[n + ndel] = tmp;
            }
            ptls2->heap.weak_refs.len -= ndel;
        }
    }
}


// big values
// ---

// Size includes the tag and the tag is not cleared!!
inline jl_value_t *jl_gc_big_alloc_inner(jl_ptls_t ptls, size_t sz)
{
    // TODO: assertion needed here?
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);
    // TODO: drop this okay?
    // maybe_collect(ptls);

    jl_value_t *v = jl_mmtk_gc_alloc_big(ptls, sz);
    // TODO: this is done (without atomic operations) in jl_mmtk_gc_alloc_big; enable
    // here when that's edited?
    /*
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_num.allocd) + allocsz);
    jl_atomic_store_relaxed(&ptls->gc_num.bigalloc,
        jl_atomic_load_relaxed(&ptls->gc_num.bigalloc) + 1);
    */
    // TODO: move to jl_mmtk_gc_alloc_big if needed.
/*
#ifdef MEMDEBUG
    memset(v, 0xee, allocsz);
#endif
*/
    // TODO: need to set this? have to move to jl_mmtk_gc_alloc_big then.
    // v->age = 0;
    // TODO: dropping this; confirm okay? `sweep_big` no longer needed?
    // gc_big_object_link(v, &ptls->heap.big_objects);
    return v;
}

// Size includes the tag and the tag is not cleared!!
inline jl_value_t *jl_gc_pool_alloc_inner(jl_ptls_t ptls, int pool_offset, int osize)
{
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);
#ifdef MEMDEBUG
    return jl_gc_big_alloc(ptls, osize);
#endif
    // TODO: drop this okay?
    // maybe_collect(ptls);

    jl_value_t *v = jl_mmtk_gc_alloc_default(ptls, osize, 16, NULL);
    // TODO: this is done (without atomic operations) in jl_mmtk_gc_alloc_default; enable
    // here when that's edited?
    /*
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_num.allocd) + osize);
    jl_atomic_store_relaxed(&ptls->gc_num.poolalloc,
        jl_atomic_load_relaxed(&ptls->gc_num.poolalloc) + 1);
    */
   return v;
}

// roots
// ---

JL_DLLEXPORT void jl_gc_queue_root(const jl_value_t *ptr)
{
    mmtk_unreachable();
}

// TODO: exported, but not MMTk-specific?
JL_DLLEXPORT void jl_gc_queue_multiroot(const jl_value_t *parent, const jl_value_t *ptr) JL_NOTSAFEPOINT
{
    mmtk_unreachable();
}

JL_DLLEXPORT void jl_gc_queue_binding(jl_binding_t *bnd)
{
    mmtk_unreachable();
}


// marking
// ---

JL_DLLEXPORT int jl_gc_mark_queue_obj(jl_ptls_t ptls, jl_value_t *obj)
{
    mmtk_unreachable();
    return 0;
}
JL_DLLEXPORT void jl_gc_mark_queue_objarray(jl_ptls_t ptls, jl_value_t *parent,
                                            jl_value_t **objs, size_t nobjs)
{
    mmtk_unreachable();
}


// GC control
// ---

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t collection)
{
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (jl_atomic_load_acquire(&jl_gc_disable_counter)) {
        size_t localbytes = jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval;
        jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
        static_assert(sizeof(_Atomic(uint64_t)) == sizeof(gc_num.deferred_alloc), "");
        jl_atomic_fetch_add((_Atomic(uint64_t)*)&gc_num.deferred_alloc, localbytes);
        return;
    }
    mmtk_handle_user_collection_request(ptls, collection);
}

void jl_init_thread_heap(jl_ptls_t ptls)
{
    jl_thread_heap_t *heap = &ptls->heap;
    jl_gc_pool_t *p = heap->norm_pools;
    for (int i = 0; i < JL_GC_N_POOLS; i++) {
        p[i].osize = jl_gc_sizeclasses[i];
        p[i].freelist = NULL;
        p[i].newpages = NULL;
    }
    small_arraylist_new(&heap->weak_refs, 0);
    small_arraylist_new(&heap->live_tasks, 0);
    for (int i = 0; i < JL_N_STACK_POOLS; i++)
        small_arraylist_new(&heap->free_stacks[i], 0);
    heap->mallocarrays = NULL;
    heap->mafreelist = NULL;
    heap->big_objects = NULL;
    arraylist_new(&heap->rem_bindings, 0);
    heap->remset = &heap->_remset[0];
    heap->last_remset = &heap->_remset[1];
    arraylist_new(heap->remset, 0);
    arraylist_new(heap->last_remset, 0);
    arraylist_new(&ptls->finalizers, 0);
    arraylist_new(&ptls->sweep_objs, 0);

    jl_gc_mark_cache_t *gc_cache = &ptls->gc_cache;
    gc_cache->perm_scanned_bytes = 0;
    gc_cache->scanned_bytes = 0;
    gc_cache->nbig_obj = 0;

    memset(&ptls->gc_num, 0, sizeof(ptls->gc_num));
    jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);

    // Create mutator
    MMTk_Mutator mmtk_mutator = mmtk_bind_mutator((void *)ptls, ptls->tid);
    // Copy the mutator to the thread local storage
    memcpy(&ptls->mmtk_mutator, mmtk_mutator, sizeof(MMTkMutatorContext));
    // Call post_bind to maintain a list of active mutators and to reclaim the old mutator (which is no longer needed)
    mmtk_post_bind_mutator(&ptls->mmtk_mutator, mmtk_mutator);
}

void jl_deinit_thread_heap(jl_ptls_t ptls)
{
    mmtk_destroy_mutator(&ptls->mmtk_mutator);
}

extern jl_mutex_t finalizers_lock;
extern arraylist_t to_finalize;
extern arraylist_t finalizer_list_marked;
extern arraylist_t eytzinger_image_tree;
extern arraylist_t eytzinger_idxs;

// System-wide initialization
// TODO: remove locks? remove anything else?
void jl_gc_init(void)
{
    JL_MUTEX_INIT(&heapsnapshot_lock);
    JL_MUTEX_INIT(&finalizers_lock);
    uv_mutex_init(&gc_perm_lock);

    arraylist_new(&finalizer_list_marked, 0);
    arraylist_new(&to_finalize, 0);
    arraylist_new(&eytzinger_image_tree, 0);
    arraylist_new(&eytzinger_idxs, 0);
    arraylist_push(&eytzinger_idxs, (void*)0);
    arraylist_push(&eytzinger_image_tree, (void*)1); // outside image

    gc_num.interval = default_collect_interval;
    last_long_collect_interval = default_collect_interval;
    gc_num.allocd = 0;
    gc_num.max_pause = 0;
    gc_num.max_memory = 0;

#ifdef _P64
    total_mem = uv_get_total_memory();
    uint64_t constrained_mem = uv_get_constrained_memory();
    if (constrained_mem > 0 && constrained_mem < total_mem)
        total_mem = constrained_mem;
#endif

    // We allocate with abandon until we get close to the free memory on the machine.
    uint64_t free_mem = uv_get_available_memory();
    uint64_t high_water_mark = free_mem / 10 * 7;  // 70% high water mark

    if (high_water_mark < max_total_memory)
       max_total_memory = high_water_mark;

    // MMTk-specific
    long long min_heap_size;
    long long max_heap_size;
    char* min_size_def = getenv("MMTK_MIN_HSIZE");
    char* min_size_gb = getenv("MMTK_MIN_HSIZE_G");

    char* max_size_def = getenv("MMTK_MAX_HSIZE");
    char* max_size_gb = getenv("MMTK_MAX_HSIZE_G");

    // default min heap currently set as 0
    // and default max heap currently set as 0
    // which means that by default mmtk will use stock heuristics
    if (min_size_def != NULL) {
        char *p;
        double min_size = strtod(min_size_def, &p);
        min_heap_size = (long) 1024 * 1024 * min_size;
    } else if (min_size_gb != NULL) {
        char *p;
        double min_size = strtod(min_size_gb, &p);
        min_heap_size = (long) 1024 * 1024 * 1024 * min_size;
    } else {
        min_heap_size = 0;
    }

    if (max_size_def != NULL) {
        char *p;
        double max_size = strtod(max_size_def, &p);
        max_heap_size = (long) 1024 * 1024 * max_size;
    } else if (max_size_gb != NULL) {
        char *p;
        double max_size = strtod(max_size_gb, &p);
        max_heap_size = (long) 1024 * 1024 * 1024 * max_size;
    } else {
        max_heap_size = 0;
    }

    // Assert that the number of stock GC threads is 0; MMTK uses the number of threads in jl_options.ngcthreads
    assert(jl_n_gcthreads == 0);

    // Check that the julia_copy_stack rust feature has been defined when the COPY_STACK has been defined
    int copy_stacks;

#ifdef COPY_STACKS
    copy_stacks = 1;
#else
    copy_stacks = 0;
#endif

    mmtk_julia_copy_stack_check(copy_stacks);

    // if only max size is specified initialize MMTk with a fixed size heap
    // TODO: We just assume mark threads means GC threads, and ignore the number of concurrent sweep threads.
    // If the two values are the same, we can use either. Otherwise, we need to be careful.
    uintptr_t gcthreads = jl_options.ngcthreads;
    // if only max_heap_size is set (max_size_def != 0), mmtk will run with a fixed heap size
    // if both min_heap_size and max_heap_size are set (their values different than 0), mmtk will use membalancer
    // and resize the heap within those values. If none of these variables are set, then mmtk will use stock heuristics
    // note that this will be overwritten in Rust by the env variable MMTK_GC_TRIGGER
    mmtk_gc_init(min_heap_size, max_heap_size, gcthreads, &mmtk_upcalls, (sizeof(jl_taggedvalue_t)), jl_buff_tag);
}

// allocation wrappers that track allocation and let collection run

JL_DLLEXPORT void *jl_gc_counted_malloc(size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, sz);
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, sz);
    }
    return malloc(sz);
}

JL_DLLEXPORT void *jl_gc_counted_calloc(size_t nm, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, nm * sz);
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, nm * sz);
    }
    return calloc(nm, sz);
}

JL_DLLEXPORT void jl_gc_counted_free_with_size(void *p, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    free(p);
    if (pgcstack && ct->world_age) {
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, -sz);
    }
}

JL_DLLEXPORT void *jl_gc_counted_realloc_with_old_size(void *p, size_t old, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, sz);
        if (sz < old)
            jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, old - sz);
        else
            jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, sz - old);
    }
    return realloc(p, sz);
}

jl_value_t *jl_gc_realloc_string(jl_value_t *s, size_t sz)
{
    size_t len = jl_string_len(s);
    jl_value_t *snew = jl_alloc_string(sz);
    memcpy(jl_string_data(snew), jl_string_data(s), sz <= len ? sz : len);
    if(mmtk_is_object_pinned(s)) {
        // if the source string was pinned, we also pin the new one
        mmtk_pin_object(snew);
    }
    return snew;
}

JL_DLLEXPORT int jl_gc_enable_conservative_gc_support(void)
{
    return 0;
}

JL_DLLEXPORT int jl_gc_conservative_gc_support_enabled(void)
{
    return 0;
}

// TODO: if this is needed, it can be added in MMTk
JL_DLLEXPORT jl_value_t *jl_gc_internal_obj_base_ptr(void *p)
{
    return NULL;
}


// gc-debug functions
// ---

jl_gc_pagemeta_t *jl_gc_page_metadata(void *data)
{
    return NULL;
}

JL_DLLEXPORT jl_taggedvalue_t *jl_gc_find_taggedvalue_pool(char *p, size_t *osize_p)
{
    return NULL;
}

void jl_gc_debug_critical_error(void) JL_NOTSAFEPOINT
{
}

void jl_gc_debug_print_status(void) JL_NOTSAFEPOINT
{
    // May not be accurate but should be helpful enough
    uint64_t pool_count = gc_num.poolalloc;
    uint64_t big_count = gc_num.bigalloc;
    jl_safe_printf("Allocations: %" PRIu64 " "
                   "(Pool: %" PRIu64 "; Big: %" PRIu64 "); GC: %d\n",
                   pool_count + big_count, pool_count, big_count, gc_num.pause);
}

void jl_print_gc_stats(JL_STREAM *s)
{
}

#ifdef OBJPROFILE
void objprofile_count(void *ty, int old, int sz) JL_NOTSAFEPOINT
{
}

void objprofile_printall(void)
{
}

void objprofile_reset(void)
{
}
#endif

// gc thread function
void jl_gc_threadfun(void *arg)
{
    mmtk_unreachable();
}

JL_DLLEXPORT void jl_gc_array_ptr_copy(jl_array_t *dest, void **dest_p, jl_array_t *src, void **src_p, ssize_t n) JL_NOTSAFEPOINT
{
    jl_ptls_t ptls = jl_current_task->ptls;
    mmtk_memory_region_copy(&ptls->mmtk_mutator, jl_array_owner(src), src_p, jl_array_owner(dest), dest_p, n);
}

#define jl_p_gcpreserve_stack (jl_current_task->gcpreserve_stack)

// This macro currently uses malloc instead of alloca because this function will exit
// after pushing the roots into the gc_preserve_stack, which means that the preserve_begin function's
// stack frame will be destroyed (together with its alloca variables). When we support lowering this code
// inside the same function that is doing the preserve_begin/preserve_end calls we should be able to simple use allocas.
// Note also that we use a separate stack for gc preserve roots to avoid the possibility of calling free
// on a stack that has been allocated with alloca instead of malloc, which could happen depending on the order in which
// JL_GC_POP() and jl_gc_preserve_end_hook() occurs.

#define JL_GC_PUSHARGS_PRESERVE_ROOT_OBJS(rts_var,n)                                                    \
  rts_var = ((jl_value_t**)malloc(((n)+2)*sizeof(jl_value_t*)))+2;                                      \
  ((void**)rts_var)[-2] = (void*)JL_GC_ENCODE_PUSHARGS(n);                                              \
  ((void**)rts_var)[-1] = jl_p_gcpreserve_stack;                                                        \
  memset((void*)rts_var, 0, (n)*sizeof(jl_value_t*));                                                   \
  jl_p_gcpreserve_stack = (jl_gcframe_t*)&(((void**)rts_var)[-2]);                                      \

#define JL_GC_POP_PRESERVE_ROOT_OBJS()                                                                  \
    jl_gcframe_t *curr = jl_p_gcpreserve_stack;                                                         \
    if(curr) {                                                                                          \
        (jl_p_gcpreserve_stack = jl_p_gcpreserve_stack->prev);                                          \
        free(curr);                                                                                     \
    }

// Add each argument as a tpin root object.
// However, we cannot use JL_GC_PUSH and JL_GC_POP since the slots should live
// beyond this function. Instead, we maintain a tpin stack by mallocing/freeing
// the frames for each of the preserve regions we encounter
JL_DLLEXPORT void jl_gc_preserve_begin_hook(int n, ...) JL_NOTSAFEPOINT
{
    jl_value_t** frame;
    JL_GC_PUSHARGS_PRESERVE_ROOT_OBJS(frame, n);
    if (n == 0) return;

    va_list args;
    va_start(args, n);
    for (int i = 0; i < n; i++) {
        frame[i] = va_arg(args, jl_value_t *);
    }
    va_end(args);
}

JL_DLLEXPORT void jl_gc_preserve_end_hook(void) JL_NOTSAFEPOINT
{
    JL_GC_POP_PRESERVE_ROOT_OBJS();
}

// No inline write barrier -- only used for debugging
JL_DLLEXPORT void jl_gc_wb1_noinline(const void *parent) JL_NOTSAFEPOINT
{
    jl_gc_wb_back(parent);
}

JL_DLLEXPORT void jl_gc_wb2_noinline(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    jl_gc_wb(parent, ptr);
}

JL_DLLEXPORT void jl_gc_wb_binding_noinline(const void *bnd, const void *val) JL_NOTSAFEPOINT
{
    jl_gc_wb_binding((jl_binding_t*)bnd, (void*)val);
}

JL_DLLEXPORT void jl_gc_wb1_slow(const void *parent) JL_NOTSAFEPOINT
{
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    mmtk_object_reference_write_slow(&ptls->mmtk_mutator, parent, (const void*) 0);
}

JL_DLLEXPORT void jl_gc_wb2_slow(const void *parent, const void* ptr) JL_NOTSAFEPOINT
{
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    mmtk_object_reference_write_slow(&ptls->mmtk_mutator, parent, ptr);
}

JL_DLLEXPORT void jl_gc_post_alloc_slow(void* obj, int size) JL_NOTSAFEPOINT {
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    mmtk_immix_post_alloc_slow(&ptls->mmtk_mutator, obj, size);
}

void *jl_gc_perm_alloc_nolock(size_t sz, int zero, unsigned align, unsigned offset)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    size_t allocsz = mmtk_align_alloc_sz(sz);
    void* addr = mmtk_immortal_alloc_fast(&ptls->mmtk_mutator, allocsz, align, offset);
    return addr;
}

void *jl_gc_perm_alloc(size_t sz, int zero, unsigned align, unsigned offset)
{
    return jl_gc_perm_alloc_nolock(sz, zero, align, offset);
}

void jl_gc_notify_image_load(const char* img_data, size_t len)
{
    mmtk_set_vm_space((void*)img_data, len);
}

void jl_gc_notify_image_alloc(char* img_data, size_t len)
{
    mmtk_immortal_region_post_alloc((void*)img_data, len);
}

JL_DLLEXPORT void jl_gc_pin(jl_value_t* obj)
{
    if (obj) {
        mmtk_pin_object(obj);
    }
}

#ifdef __cplusplus
}
#endif

#endif // MMTK_GC
