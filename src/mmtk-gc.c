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


inline void maybe_collect(jl_ptls_t ptls)
{
    mmtk_gc_poll(ptls);
}


// malloc wrappers, aligned allocation
// ---

inline void *jl_malloc_aligned(size_t sz, size_t align)
{
    return mmtk_malloc_aligned(sz ? sz : 1, align); // XXX sz
}
inline void *jl_realloc_aligned(void *d, size_t sz, size_t oldsz,
                                       size_t align)
{
    void *res = jl_malloc_aligned(sz, align);
    if (res != NULL) {
        memcpy(res, d, oldsz > sz ? sz : oldsz);
        mmtk_free_aligned(d);
    }
    return res;
}
inline void jl_free_aligned(void *p) JL_NOTSAFEPOINT
{
    mmtk_free_aligned(p);
}


// finalizers
// ---

JL_DLLEXPORT void jl_gc_run_pending_finalizers(jl_task_t *ct)
{
    if (ct == NULL)
        ct = jl_current_task;
    mmtk_jl_run_pending_finalizers(ct->ptls);
}

JL_DLLEXPORT void jl_gc_add_ptr_finalizer(jl_ptls_t ptls, jl_value_t *v, void *f) JL_NOTSAFEPOINT
{
    register_finalizer(v, f, 1);
}

// schedule f(v) to call at the next quiescent interval (aka after the next safepoint/region on all threads)
JL_DLLEXPORT void jl_gc_add_quiescent(jl_ptls_t ptls, void **v, void *f) JL_NOTSAFEPOINT
{
    /* TODO: unsupported? */
}

JL_DLLEXPORT void jl_gc_add_finalizer_th(jl_ptls_t ptls, jl_value_t *v, jl_function_t *f) JL_NOTSAFEPOINT
{
    if (__unlikely(jl_typeis(f, jl_voidpointer_type))) {
        jl_gc_add_ptr_finalizer(ptls, v, jl_unbox_voidpointer(f));
    }
    else {
        register_finalizer(v, f, 0);
    }
}

JL_DLLEXPORT void jl_finalize_th(jl_task_t *ct, jl_value_t *o)
{
    run_finalizers_for_obj(o);
}

void jl_gc_run_all_finalizers(jl_task_t *ct)
{
    mmtk_jl_gc_run_all_finalizers();
}

void jl_gc_add_finalizer_(jl_ptls_t ptls, void *v, void *f) JL_NOTSAFEPOINT
{
    register_finalizer(v, f, 0);
}


// weak references
// ---
JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref_th(jl_ptls_t ptls, jl_value_t *value)
{
    jl_weakref_t *wr = (jl_weakref_t*)jl_gc_alloc(ptls, sizeof(void*), jl_weakref_type);
    wr->value = value;  // NOTE: wb not needed here
    mmtk_add_weak_candidate(wr);
    return wr;
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

    jl_value_t *v = jl_mmtk_gc_alloc_default(ptls, pool_offset, osize, NULL);
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

void jl_gc_free_array(jl_array_t *a) JL_NOTSAFEPOINT
{
    if (a->flags.how == 2) {
        char *d = (char*)a->data - a->offset*a->elsize;
        if (a->flags.isaligned)
            mmtk_free_aligned(d);
        else
            mmtk_free(d);
        gc_num.freed += jl_array_nbytes(a);
        gc_num.freecall++;
    }
}


// roots
// ---

JL_DLLEXPORT void jl_gc_queue_root(const jl_value_t *ptr)
{
    /* TODO: not needed? */
}

// TODO: exported, but not MMTk-specific?
JL_DLLEXPORT void jl_gc_queue_multiroot(const jl_value_t *parent, const jl_value_t *ptr) JL_NOTSAFEPOINT
{
    /* TODO: confirm not needed? */
}


// marking
// ---

JL_DLLEXPORT int jl_gc_mark_queue_obj(jl_ptls_t ptls, jl_value_t *obj)
{
    return 0;
}
JL_DLLEXPORT void jl_gc_mark_queue_objarray(jl_ptls_t ptls, jl_value_t *parent,
                                            jl_value_t **objs, size_t nobjs)
{
}


// GC control
// ---

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t collection)
{
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (jl_atomic_load_relaxed(&jl_gc_disable_counter)) {
        size_t localbytes = jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval;
        jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
        static_assert(sizeof(_Atomic(uint64_t)) == sizeof(gc_num.deferred_alloc), "");
        jl_atomic_fetch_add((_Atomic(uint64_t)*)&gc_num.deferred_alloc, localbytes);
        return;
    }
    handle_user_collection_request(ptls);
}

// Per-thread initialization
// TODO: remove `norm_pools`, `weak_refs`, etc. from `heap`?
// TODO: remove `gc_cache`?
void jl_init_thread_heap(jl_ptls_t ptls)
{
    jl_thread_heap_t *heap = &ptls->heap;
    jl_gc_pool_t *p = heap->norm_pools;
    for (int i = 0; i < JL_GC_N_POOLS; i++) {
        p[i].osize = jl_gc_sizeclasses[i];
        p[i].freelist = NULL;
        p[i].newpages = NULL;
    }
    arraylist_new(&heap->weak_refs, 0);
    arraylist_new(&heap->live_tasks, 0);
    heap->mallocarrays = NULL;
    heap->mafreelist = NULL;
    heap->big_objects = NULL;
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

    MMTk_Mutator mmtk_mutator = bind_mutator((void *)ptls, ptls->tid);
    ptls->mmtk_mutator_ptr = ((MMTkMutatorContext*)mmtk_mutator);
}

// System-wide initialization
// TODO: remove locks? remove anything else?
void jl_gc_init(void)
{
    if (jl_options.heap_size_hint)
        jl_gc_set_max_memory(jl_options.heap_size_hint);

    JL_MUTEX_INIT(&heapsnapshot_lock);
    uv_mutex_init(&gc_perm_lock);

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

    // default min heap currently set as Julia's default_collect_interval
    if (min_size_def != NULL) {
        char *p;
        double min_size = strtod(min_size_def, &p);
        min_heap_size = (long) 1024 * 1024 * min_size;
    } else if (min_size_gb != NULL) {
        char *p;
        double min_size = strtod(min_size_gb, &p);
        min_heap_size = (long) 1024 * 1024 * 1024 * min_size;
    } else {
        min_heap_size = default_collect_interval;
    }

    // default max heap currently set as 70% the free memory in the system
    if (max_size_def != NULL) {
        char *p;
        double max_size = strtod(max_size_def, &p);
        max_heap_size = (long) 1024 * 1024 * max_size;
    } else if (max_size_gb != NULL) {
        char *p;
        double max_size = strtod(max_size_gb, &p);
        max_heap_size = (long) 1024 * 1024 * 1024 * max_size;
    } else {
        max_heap_size = uv_get_free_memory() * 70 / 100;
    }

    // if only max size is specified initialize MMTk with a fixed size heap
    if (max_size_def != NULL || (max_size_gb != NULL && (min_size_def == NULL && min_size_gb == NULL))) {
        gc_init(0, max_heap_size, &mmtk_upcalls, (sizeof(jl_taggedvalue_t)));
    } else {
        gc_init(min_heap_size, max_heap_size, &mmtk_upcalls, (sizeof(jl_taggedvalue_t)));
    }
}

// allocation wrappers that track allocation and let collection run

JL_DLLEXPORT void *jl_gc_counted_malloc(size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + sz);
        jl_atomic_store_relaxed(&ptls->gc_num.malloc,
            jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
        return mmtk_counted_malloc(sz);
    }
    return malloc(sz);
}

JL_DLLEXPORT void *jl_gc_counted_calloc(size_t nm, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + nm*sz);
        jl_atomic_store_relaxed(&ptls->gc_num.malloc,
            jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
        return mmtk_counted_calloc(nm, sz);
    }
    return calloc(nm, sz);
}

JL_DLLEXPORT void jl_gc_counted_free_with_size(void *p, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        jl_atomic_store_relaxed(&ptls->gc_num.freed,
            jl_atomic_load_relaxed(&ptls->gc_num.freed) + sz);
        jl_atomic_store_relaxed(&ptls->gc_num.freecall,
            jl_atomic_load_relaxed(&ptls->gc_num.freecall) + 1);
        mmtk_free_with_size(p, sz);
        return;
    }
    free(p);
}

JL_DLLEXPORT void *jl_gc_counted_realloc_with_old_size(void *p, size_t old, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        maybe_collect(ptls);
        if (sz < old)
            jl_atomic_store_relaxed(&ptls->gc_num.freed,
                jl_atomic_load_relaxed(&ptls->gc_num.freed) + (old - sz));
        else
            jl_atomic_store_relaxed(&ptls->gc_num.allocd,
                jl_atomic_load_relaxed(&ptls->gc_num.allocd) + (sz - old));
        jl_atomic_store_relaxed(&ptls->gc_num.realloc,
            jl_atomic_load_relaxed(&ptls->gc_num.realloc) + 1);
        return mmtk_realloc_with_old_size(p, sz, old);
    }
    // TODO: correct?
    return realloc(p, sz);
}

jl_value_t *jl_gc_realloc_string(jl_value_t *s, size_t sz)
{
    size_t len = jl_string_len(s);
    jl_value_t *snew = jl_alloc_string(sz);
    memcpy(jl_string_data(snew), jl_string_data(s), sz <= len ? sz : len);
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

void objprofile_count(void *ty, int old, int sz) JL_NOTSAFEPOINT
{
}

void objprofile_printall(void)
{
}

void objprofile_reset(void)
{
}

#ifdef __cplusplus
}
#endif

#endif // MMTK_GC
