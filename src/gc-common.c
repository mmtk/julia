// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc.h"

jl_gc_num_t gc_num = {0};
size_t last_long_collect_interval;
int gc_n_threads;
jl_ptls_t* gc_all_tls_states;

const uint64_t _jl_buff_tag[3] = {0x4eadc0004eadc000ull, 0x4eadc0004eadc000ull, 0x4eadc0004eadc000ull}; // aka 0xHEADER00

JL_DLLEXPORT uintptr_t jl_get_buff_tag(void)
{
    return jl_buff_tag;
}

// `tid` of first GC thread
int gc_first_tid;

// These should be moved to gc.c
int64_t live_bytes = 0;

JL_DLLEXPORT _Atomic(int) jl_gc_have_pending_finalizers = 0;

// mutex for gc-heap-snapshot.
jl_mutex_t heapsnapshot_lock;


// GC knobs and self-measurement variables
int64_t last_gc_total_bytes = 0;

// max_total_memory is a suggestion.  We try very hard to stay
// under this limit, but we will go above it rather than halting.
#ifdef _P64
typedef uint64_t memsize_t;
const size_t default_collect_interval = 5600 * 1024 * sizeof(void*);
const size_t max_collect_interval = 1250000000UL;
size_t total_mem;
// We expose this to the user/ci as jl_gc_set_max_memory
memsize_t max_total_memory = (memsize_t) 2 * 1024 * 1024 * 1024 * 1024 * 1024;
#else
typedef uint32_t memsize_t;
const size_t default_collect_interval = 3200 * 1024 * sizeof(void*);
const size_t max_collect_interval =  500000000UL;
// Work really hard to stay within 2GB
// Alternative is to risk running out of address space
// on 32 bit architectures.
memsize_t max_total_memory = (memsize_t) 2 * 1024 * 1024 * 1024;
#endif


// finalizers
// ---
uint64_t finalizer_rngState[4];
jl_mutex_t finalizers_lock;
// `ptls->finalizers` and `finalizer_list_marked` might have tagged pointers.
// If an object pointer has the lowest bit set, the next pointer is an unboxed c function pointer.
// If an object pointer has the second lowest bit set, the current pointer is a c object pointer.
//   It must be aligned at least 4, and it finalized immediately (at "quiescence").
// `to_finalize` should not have tagged pointers.
arraylist_t finalizer_list_marked;
arraylist_t to_finalize;

void jl_rng_split(uint64_t dst[4], uint64_t src[4]);

JL_DLLEXPORT void jl_gc_init_finalizer_rng_state(void)
{
    jl_rng_split(finalizer_rngState, jl_current_task->rngState);
}

// The first two entries are assumed to be empty and the rest are assumed to
// be pointers to `jl_value_t` objects
STATIC_INLINE void jl_gc_push_arraylist(jl_task_t *ct, arraylist_t *list) JL_NOTSAFEPOINT
{
    void **items = list->items;
    items[0] = (void*)JL_GC_ENCODE_PUSHARGS(list->len - 2);
    items[1] = ct->gcstack;
    ct->gcstack = (jl_gcframe_t*)items;
}

STATIC_INLINE void schedule_finalization(void *o, void *f) JL_NOTSAFEPOINT
{
    arraylist_push(&to_finalize, o);
    arraylist_push(&to_finalize, f);
    // doesn't need release, since we'll keep checking (on the reader) until we see the work and
    // release our lock, and that will have a release barrier by then
    jl_atomic_store_relaxed(&jl_gc_have_pending_finalizers, 1);
}

void run_finalizer(jl_task_t *ct, void *o, void *ff)
{
    int ptr_finalizer = gc_ptr_tag(o, 1);
    o = gc_ptr_clear_tag(o, 3);
    if (ptr_finalizer) {
        ((void (*)(void*))ff)((void*)o);
        return;
    }
    JL_TRY {
        size_t last_age = ct->world_age;
        ct->world_age = jl_atomic_load_acquire(&jl_world_counter);
        jl_apply_generic((jl_value_t*)ff, (jl_value_t**)&o, 1);
        ct->world_age = last_age;
    }
    JL_CATCH {
        jl_printf((JL_STREAM*)STDERR_FILENO, "error in running finalizer: ");
        jl_static_show((JL_STREAM*)STDERR_FILENO, jl_current_exception());
        jl_printf((JL_STREAM*)STDERR_FILENO, "\n");
        jlbacktrace(); // written to STDERR_FILENO
    }
}

void jl_gc_add_finalizer_(jl_ptls_t ptls, void *v, void *f) JL_NOTSAFEPOINT
{
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);
    arraylist_t *a = &ptls->finalizers;
    // This acquire load and the release store at the end are used to
    // synchronize with `finalize_object` on another thread. Apart from the GC,
    // which is blocked by entering a unsafe region, there might be only
    // one other thread accessing our list in `finalize_object`
    // (only one thread since it needs to acquire the finalizer lock).
    // Similar to `finalize_object`, all content mutation has to be done
    // between the acquire and the release of the length.
    size_t oldlen = jl_atomic_load_acquire((_Atomic(size_t)*)&a->len);
    if (__unlikely(oldlen + 2 > a->max)) {
        JL_LOCK_NOGC(&finalizers_lock);
        // `a->len` might have been modified.
        // Another possibility is to always grow the array to `oldlen + 2` but
        // it's simpler this way and uses slightly less memory =)
        oldlen = a->len;
        arraylist_grow(a, 2);
        a->len = oldlen;
        JL_UNLOCK_NOGC(&finalizers_lock);
    }
    void **items = a->items;
    items[oldlen] = v;
    items[oldlen + 1] = f;
    jl_atomic_store_release((_Atomic(size_t)*)&a->len, oldlen + 2);
}

// Same assumption as `jl_gc_push_arraylist`. Requires the finalizers lock
// to be hold for the current thread and will release the lock when the
// function returns.
void jl_gc_run_finalizers_in_list(jl_task_t *ct, arraylist_t *list)
{
    // Avoid marking `ct` as non-migratable via an `@async` task (as noted in the docstring
    // of `finalizer`) in a finalizer:
    uint8_t sticky = ct->sticky;
    // empty out the first two entries for the GC frame
    arraylist_push(list, list->items[0]);
    arraylist_push(list, list->items[1]);
    jl_gc_push_arraylist(ct, list);
    void **items = list->items;
    size_t len = list->len;
    JL_UNLOCK_NOGC(&finalizers_lock);
    // run finalizers in reverse order they were added, so lower-level finalizers run last
    for (size_t i = len-4; i >= 2; i -= 2)
        run_finalizer(ct, items[i], items[i + 1]);
    // first entries were moved last to make room for GC frame metadata
    run_finalizer(ct, items[len-2], items[len-1]);
    // matches the jl_gc_push_arraylist above
    JL_GC_POP();
    ct->sticky = sticky;
}

void run_finalizers(jl_task_t *ct)
{
    // Racy fast path:
    // The race here should be OK since the race can only happen if
    // another thread is writing to it with the lock held. In such case,
    // we don't need to run pending finalizers since the writer thread
    // will flush it.
    if (to_finalize.len == 0)
        return;
    JL_LOCK_NOGC(&finalizers_lock);
    if (to_finalize.len == 0) {
        JL_UNLOCK_NOGC(&finalizers_lock);
        return;
    }
    arraylist_t copied_list;
    memcpy(&copied_list, &to_finalize, sizeof(copied_list));
    if (to_finalize.items == to_finalize._space) {
        copied_list.items = copied_list._space;
    }
    jl_atomic_store_relaxed(&jl_gc_have_pending_finalizers, 0);
    arraylist_new(&to_finalize, 0);

    uint64_t save_rngState[4];
    memcpy(&save_rngState[0], &ct->rngState[0], sizeof(save_rngState));
    jl_rng_split(ct->rngState, finalizer_rngState);

    // This releases the finalizers lock.
    int8_t was_in_finalizer = ct->ptls->in_finalizer;
    ct->ptls->in_finalizer = 1;
    jl_gc_run_finalizers_in_list(ct, &copied_list);
    ct->ptls->in_finalizer = was_in_finalizer;
    arraylist_free(&copied_list);

    memcpy(&ct->rngState[0], &save_rngState[0], sizeof(save_rngState));
}

// if `need_sync` is true, the `list` is the `finalizers` list of another
// thread and we need additional synchronizations
void finalize_object(arraylist_t *list, jl_value_t *o,
                            arraylist_t *copied_list, int need_sync) JL_NOTSAFEPOINT
{
    // The acquire load makes sure that the first `len` objects are valid.
    // If `need_sync` is true, all mutations of the content should be limited
    // to the first `oldlen` elements and no mutation is allowed after the
    // new length is published with the `cmpxchg` at the end of the function.
    // This way, the mutation should not conflict with the owning thread,
    // which only writes to locations later than `len`
    // and will not resize the buffer without acquiring the lock.
    size_t len = need_sync ? jl_atomic_load_acquire((_Atomic(size_t)*)&list->len) : list->len;
    size_t oldlen = len;
    void **items = list->items;
    size_t j = 0;
    for (size_t i = 0; i < len; i += 2) {
        void *v = items[i];
        int move = 0;
        if (o == (jl_value_t*)gc_ptr_clear_tag(v, 1)) {
            void *f = items[i + 1];
            move = 1;
            arraylist_push(copied_list, v);
            arraylist_push(copied_list, f);
        }
        if (move || __unlikely(!v)) {
            // remove item
        }
        else {
            if (j < i) {
                items[j] = items[i];
                items[j+1] = items[i+1];
            }
            j += 2;
        }
    }
    len = j;
    if (oldlen == len)
        return;
    if (need_sync) {
        // The memset needs to be unconditional since the thread might have
        // already read the length.
        // The `memset` (like any other content mutation) has to be done
        // **before** the `cmpxchg` which publishes the length.
        memset(&items[len], 0, (oldlen - len) * sizeof(void*));
        jl_atomic_cmpswap((_Atomic(size_t)*)&list->len, &oldlen, len);
    }
    else {
        list->len = len;
    }
}

JL_DLLEXPORT void jl_gc_add_ptr_finalizer(jl_ptls_t ptls, jl_value_t *v, void *f) JL_NOTSAFEPOINT
{
    jl_gc_add_finalizer_(ptls, (void*)(((uintptr_t)v) | 1), f);
}

// schedule f(v) to call at the next quiescent interval (aka after the next safepoint/region on all threads)
JL_DLLEXPORT void jl_gc_add_quiescent(jl_ptls_t ptls, void **v, void *f) JL_NOTSAFEPOINT
{
    assert(!gc_ptr_tag(v, 3));
    jl_gc_add_finalizer_(ptls, (void*)(((uintptr_t)v) | 3), f);
}

JL_DLLEXPORT void jl_gc_add_finalizer_th(jl_ptls_t ptls, jl_value_t *v, jl_function_t *f) JL_NOTSAFEPOINT
{
    if (__unlikely(jl_typeis(f, jl_voidpointer_type))) {
        jl_gc_add_ptr_finalizer(ptls, v, jl_unbox_voidpointer(f));
    }
    else {
        jl_gc_add_finalizer_(ptls, v, f);
    }
}

JL_DLLEXPORT void jl_gc_run_pending_finalizers(jl_task_t *ct)
{
    if (ct == NULL)
        ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (!ptls->in_finalizer && ptls->locks.len == 0 && ptls->finalizers_inhibited == 0) {
        run_finalizers(ct);
    }
}

JL_DLLEXPORT void jl_finalize_th(jl_task_t *ct, jl_value_t *o)
{
    JL_LOCK_NOGC(&finalizers_lock);
    // Copy the finalizers into a temporary list so that code in the finalizer
    // won't change the list as we loop through them.
    // This list is also used as the GC frame when we are running the finalizers
    arraylist_t copied_list;
    arraylist_new(&copied_list, 0);
    // No need to check the to_finalize list since the user is apparently
    // still holding a reference to the object
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = gc_all_tls_states[i];
        if (ptls2 != NULL)
            finalize_object(&ptls2->finalizers, o, &copied_list, jl_atomic_load_relaxed(&ct->tid) != i);
    }
    finalize_object(&finalizer_list_marked, o, &copied_list, 0);
    gc_n_threads = 0;
    gc_all_tls_states = NULL;
    if (copied_list.len > 0) {
        // This releases the finalizers lock.
        jl_gc_run_finalizers_in_list(ct, &copied_list);
    }
    else {
        JL_UNLOCK_NOGC(&finalizers_lock);
    }
    arraylist_free(&copied_list);
}

void schedule_all_finalizers(arraylist_t *flist) JL_NOTSAFEPOINT
{
    void **items = flist->items;
    size_t len = flist->len;
    for(size_t i = 0; i < len; i+=2) {
        void *v = items[i];
        void *f = items[i + 1];
        if (__unlikely(!v))
            continue;
        schedule_finalization(v, f);
    }
    flist->len = 0;
}

void jl_gc_run_all_finalizers(jl_task_t *ct)
{
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    // this is called from `jl_atexit_hook`; threads could still be running
    // so we have to guard the finalizers' lists
    JL_LOCK_NOGC(&finalizers_lock);
    schedule_all_finalizers(&finalizer_list_marked);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = gc_all_tls_states[i];
        if (ptls2 != NULL)
            schedule_all_finalizers(&ptls2->finalizers);
    }
    // unlock here because `run_finalizers` locks this
    JL_UNLOCK_NOGC(&finalizers_lock);
    gc_n_threads = 0;
    gc_all_tls_states = NULL;
    run_finalizers(ct);
}

JL_DLLEXPORT int jl_gc_get_finalizers_inhibited(jl_ptls_t ptls)
{
    if (ptls == NULL)
        ptls = jl_current_task->ptls;
    return ptls->finalizers_inhibited;
}

JL_DLLEXPORT void jl_gc_disable_finalizers_internal(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    ptls->finalizers_inhibited++;
}

JL_DLLEXPORT void jl_gc_enable_finalizers_internal(void)
{
    jl_task_t *ct = jl_current_task;
#ifdef NDEBUG
    ct->ptls->finalizers_inhibited--;
#else
    jl_gc_enable_finalizers(ct, 1);
#endif
}

JL_DLLEXPORT void jl_gc_enable_finalizers(jl_task_t *ct, int on)
{
    if (ct == NULL)
        ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    int old_val = ptls->finalizers_inhibited;
    int new_val = old_val + (on ? -1 : 1);
    if (new_val < 0) {
        JL_TRY {
            jl_error(""); // get a backtrace
        }
        JL_CATCH {
            jl_printf((JL_STREAM*)STDERR_FILENO, "WARNING: GC finalizers already enabled on this thread.\n");
            // Only print the backtrace once, to avoid spamming the logs
            static int backtrace_printed = 0;
            if (backtrace_printed == 0) {
                backtrace_printed = 1;
                jlbacktrace(); // written to STDERR_FILENO
            }
        }
        return;
    }
    ptls->finalizers_inhibited = new_val;
    if (jl_atomic_load_relaxed(&jl_gc_have_pending_finalizers)) {
        jl_gc_run_pending_finalizers(ct);
    }
}

// allocation
// ---

JL_DLLEXPORT jl_value_t *(jl_gc_alloc)(jl_ptls_t ptls, size_t sz, void *ty)
{
    return jl_gc_alloc_(ptls, sz, ty);
}

// Instrumented version of jl_gc_big_alloc_inner, called into by
// LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_big_alloc(jl_ptls_t ptls, size_t sz)
{
    jl_value_t *val = jl_gc_big_alloc_inner(ptls, sz);

    maybe_record_alloc_to_profile(val, sz, jl_gc_unknown_type_tag);
    return val;
}

// Instrumented version of jl_gc_big_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_big_alloc_instrumented(jl_ptls_t ptls, size_t sz, jl_value_t *type)
{
    jl_value_t *val = jl_gc_big_alloc_inner(ptls, sz);
    maybe_record_alloc_to_profile(val, sz, (jl_datatype_t*)type);
    return val;
}


// This wrapper exists only to prevent `jl_gc_big_alloc_inner` from being
// inlined into its callers. We provide an external-facing interface for
// callers, and inline `jl_gc_big_alloc_inner` into this. (See
// https://github.com/JuliaLang/julia/pull/43868 for more details.)
jl_value_t *jl_gc_big_alloc_noinline(jl_ptls_t ptls, size_t sz)
{
    return jl_gc_big_alloc_inner(ptls, sz);
}

// Instrumented version of jl_gc_pool_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_pool_alloc_instrumented(jl_ptls_t ptls, int pool_offset,
                                        int osize, jl_value_t* type)
{
    jl_value_t *val = jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
    maybe_record_alloc_to_profile(val, osize, (jl_datatype_t*)type);
    return val;
}

JL_DLLEXPORT jl_value_t *jl_gc_pool_alloc(jl_ptls_t ptls, int pool_offset, int osize)
{
    jl_value_t *val = jl_gc_pool_alloc_inner(ptls, pool_offset, osize);

    maybe_record_alloc_to_profile(val, osize, jl_gc_unknown_type_tag);
    return val;
}

// This wrapper exists only to prevent `jl_gc_pool_alloc_inner` from being inlined into
// its callers. We provide an external-facing interface for callers, and inline `jl_gc_pool_alloc_inner`
// into this. (See https://github.com/JuliaLang/julia/pull/43868 for more details.)
jl_value_t *jl_gc_pool_alloc_noinline(jl_ptls_t ptls, int pool_offset, int osize)
{
    return jl_gc_pool_alloc_inner(ptls, pool_offset, osize);
}

// TODO: jl_gc_track_malloced_array needed? Eliminate heap.mallocarrays,
// heap.mafreelist, mallocarray_t?
void jl_gc_track_malloced_array(jl_ptls_t ptls, jl_array_t *a) JL_NOTSAFEPOINT
{
    // This is **NOT** a GC safe point.
    mallocarray_t *ma;
    if (ptls->heap.mafreelist == NULL) {
        ma = (mallocarray_t*)malloc_s(sizeof(mallocarray_t));
    }
    else {
        ma = ptls->heap.mafreelist;
        ptls->heap.mafreelist = ma->next;
    }
    ma->a = a;
    ma->next = ptls->heap.mallocarrays;
    ptls->heap.mallocarrays = ma;
}

void jl_gc_count_allocd(size_t sz) JL_NOTSAFEPOINT
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_num.allocd) + sz);
}


// GCNum, statistics manipulation
// ---
void combine_thread_gc_counts(jl_gc_num_t *dest) JL_NOTSAFEPOINT
{
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls = gc_all_tls_states[i];
        if (ptls) {
            dest->allocd += (jl_atomic_load_relaxed(&ptls->gc_num.allocd) + gc_num.interval);
            dest->freed += jl_atomic_load_relaxed(&ptls->gc_num.freed);
            dest->malloc += jl_atomic_load_relaxed(&ptls->gc_num.malloc);
            dest->realloc += jl_atomic_load_relaxed(&ptls->gc_num.realloc);
            dest->poolalloc += jl_atomic_load_relaxed(&ptls->gc_num.poolalloc);
            dest->bigalloc += jl_atomic_load_relaxed(&ptls->gc_num.bigalloc);
            dest->freecall += jl_atomic_load_relaxed(&ptls->gc_num.freecall);
        }
    }
}

void reset_thread_gc_counts(void) JL_NOTSAFEPOINT
{
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls = gc_all_tls_states[i];
        if (ptls != NULL) {
            // don't reset `pool_live_bytes` here
            jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
            jl_atomic_store_relaxed(&ptls->gc_num.freed, 0);
            jl_atomic_store_relaxed(&ptls->gc_num.malloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_num.realloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_num.poolalloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_num.bigalloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_num.freecall, 0);
        }
    }
}

void jl_gc_reset_alloc_count(void) JL_NOTSAFEPOINT
{
    combine_thread_gc_counts(&gc_num);
    live_bytes += (gc_num.deferred_alloc + gc_num.allocd);
    gc_num.allocd = 0;
    gc_num.deferred_alloc = 0;
    reset_thread_gc_counts();
}

size_t jl_array_nbytes(jl_array_t *a) JL_NOTSAFEPOINT
{
    size_t sz = 0;
    int isbitsunion = jl_array_isbitsunion(a);
    if (jl_array_ndims(a) == 1)
        sz = a->elsize * a->maxsize + ((a->elsize == 1 && !isbitsunion) ? 1 : 0);
    else
        sz = a->elsize * jl_array_len(a);
    if (isbitsunion)
        // account for isbits Union array selector bytes
        sz += jl_array_len(a);
    return sz;
}


void gc_premark(jl_ptls_t ptls2)
{
    arraylist_t *remset = ptls2->heap.remset;
    ptls2->heap.remset = ptls2->heap.last_remset;
    ptls2->heap.last_remset = remset;
    ptls2->heap.remset->len = 0;
    ptls2->heap.remset_nptr = 0;
    // avoid counting remembered objects & bindings twice
    // in `perm_scanned_bytes`
    size_t len = remset->len;
    void **items = remset->items;
    for (size_t i = 0; i < len; i++) {
        jl_value_t *item = (jl_value_t*)items[i];
        objprofile_count(jl_typeof(item), 2, 0);
        jl_astaggedvalue(item)->bits.gc = GC_OLD_MARKED;
    }
    len = ptls2->heap.rem_bindings.len;
    items = ptls2->heap.rem_bindings.items;
    for (size_t i = 0; i < len; i++) {
        void *ptr = items[i];
        jl_astaggedvalue(ptr)->bits.gc = GC_OLD_MARKED;
    }
}



// GC control
// ---

JL_DLLEXPORT _Atomic(uint32_t) jl_gc_disable_counter = 1;

JL_DLLEXPORT int jl_gc_enable(int on)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    int prev = !ptls->disable_gc;
    ptls->disable_gc = (on == 0);
    if (on && !prev) {
        // disable -> enable
        if (jl_atomic_fetch_add(&jl_gc_disable_counter, -1) == 1) {
            gc_num.allocd += gc_num.deferred_alloc;
            gc_num.deferred_alloc = 0;
        }
    }
    else if (prev && !on) {
        // enable -> disable
        jl_atomic_fetch_add(&jl_gc_disable_counter, 1);
        // check if the GC is running and wait for it to finish
        jl_gc_safepoint_(ptls);
    }
    return prev;
}

void jl_gc_wait_for_the_world(jl_ptls_t* gc_all_tls_states, int gc_n_threads)
{
    assert(gc_n_threads);
    if (gc_n_threads > 1)
        jl_wake_libuv();
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls2 = gc_all_tls_states[i];
        if (ptls2 != NULL) {
            // This acquire load pairs with the release stores
            // in the signal handler of safepoint so we are sure that
            // all the stores on those threads are visible.
            // We're currently also using atomic store release in mutator threads
            // (in jl_gc_state_set), but we may want to use signals to flush the
            // memory operations on those threads lazily instead.
            while (!jl_atomic_load_relaxed(&ptls2->gc_state) || !jl_atomic_load_acquire(&ptls2->gc_state))
                jl_cpu_pause(); // yield?
        }
    }
}

JL_DLLEXPORT int jl_gc_is_enabled(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return !ptls->disable_gc;
}

JL_DLLEXPORT void jl_gc_get_total_bytes(int64_t *bytes) JL_NOTSAFEPOINT
{
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num);
    // Sync this logic with `base/util.jl:GC_Diff`
    *bytes = (num.total_allocd + num.deferred_alloc + num.allocd);
}

JL_DLLEXPORT uint64_t jl_gc_total_hrtime(void)
{
    return gc_num.total_time;
}

JL_DLLEXPORT jl_gc_num_t jl_gc_num(void)
{
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num);
    return num;
}

JL_DLLEXPORT void jl_gc_reset_stats(void)
{
    gc_num.max_pause = 0;
    gc_num.max_memory = 0;
    gc_num.max_time_to_safepoint = 0;
}

// TODO: these were supposed to be thread local
JL_DLLEXPORT int64_t jl_gc_diff_total_bytes(void) JL_NOTSAFEPOINT
{
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_sync_total_bytes(int64_t offset) JL_NOTSAFEPOINT
{
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb - offset;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_live_bytes(void)
{
    return live_bytes;
}

void jl_gc_set_max_memory(uint64_t max_mem) {
    if (max_mem > 0
        && max_mem < (uint64_t)1 << (sizeof(memsize_t) * 8 - 1)) {
        max_total_memory = max_mem;
    }
}

JL_DLLEXPORT uint64_t jl_gc_get_max_memory(void)
{
    return max_total_memory;
}

// callback for passing OOM errors from gmp
JL_DLLEXPORT void jl_throw_out_of_memory_error(void)
{
    jl_throw(jl_memory_exception);
}

// allocation wrappers that save the size of allocations, to allow using
// jl_gc_counted_* functions with a libc-compatible API.

JL_DLLEXPORT void *jl_malloc(size_t sz)
{
    int64_t *p = (int64_t *)jl_gc_counted_malloc(sz + JL_SMALL_BYTE_ALIGNMENT);
    if (p == NULL)
        return NULL;
    p[0] = sz;
    return (void *)(p + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}

//_unchecked_calloc does not check for potential overflow of nm*sz
STATIC_INLINE void *_unchecked_calloc(size_t nm, size_t sz) {
    size_t nmsz = nm*sz;
    int64_t *p = (int64_t *)jl_gc_counted_calloc(nmsz + JL_SMALL_BYTE_ALIGNMENT, 1);
    if (p == NULL)
        return NULL;
    p[0] = nmsz;
    return (void *)(p + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}

JL_DLLEXPORT void *jl_calloc(size_t nm, size_t sz)
{
    if (nm > SSIZE_MAX/sz - JL_SMALL_BYTE_ALIGNMENT)
        return NULL;
    return _unchecked_calloc(nm, sz);
}

JL_DLLEXPORT void jl_free(void *p)
{
    if (p != NULL) {
        int64_t *pp = (int64_t *)p - 2;
        size_t sz = pp[0];
        jl_gc_counted_free_with_size(pp, sz + JL_SMALL_BYTE_ALIGNMENT);
    }
}

JL_DLLEXPORT void *jl_realloc(void *p, size_t sz)
{
    int64_t *pp;
    size_t szold;
    if (p == NULL) {
        pp = NULL;
        szold = 0;
    }
    else {
        pp = (int64_t *)p - 2;
        szold = pp[0] + JL_SMALL_BYTE_ALIGNMENT;
    }
    int64_t *pnew = (int64_t *)jl_gc_counted_realloc_with_old_size(pp, szold, sz + JL_SMALL_BYTE_ALIGNMENT);
    if (pnew == NULL)
        return NULL;
    pnew[0] = sz;
    return (void *)(pnew + 2); // assumes JL_SMALL_BYTE_ALIGNMENT == 16
}

// allocating blocks for Arrays and Strings

JL_DLLEXPORT void *jl_gc_managed_malloc(size_t sz)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    maybe_collect(ptls);
    size_t allocsz = LLT_ALIGN(sz, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz)  // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);
    jl_atomic_store_relaxed(&ptls->gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_num.allocd) + allocsz);
    jl_atomic_store_relaxed(&ptls->gc_num.malloc,
        jl_atomic_load_relaxed(&ptls->gc_num.malloc) + 1);
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *b = malloc_cache_align(allocsz);
    if (b == NULL)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    // jl_gc_managed_malloc is currently always used for allocating array buffers.
    maybe_record_alloc_to_profile((jl_value_t*)b, sz, (jl_datatype_t*)jl_buff_tag);
    return b;
}

void *gc_managed_realloc_(jl_ptls_t ptls, void *d, size_t sz, size_t oldsz,
                                 int isaligned, jl_value_t *owner, int8_t can_collect)
{
    if (can_collect)
        maybe_collect(ptls);

    size_t allocsz = LLT_ALIGN(sz, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz)  // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);

    if (jl_astaggedvalue(owner)->bits.gc == GC_OLD_MARKED) {
        ptls->gc_cache.perm_scanned_bytes += allocsz - oldsz;
        live_bytes += allocsz - oldsz;
    }
    else if (allocsz < oldsz)
        jl_atomic_store_relaxed(&ptls->gc_num.freed,
            jl_atomic_load_relaxed(&ptls->gc_num.freed) + (oldsz - allocsz));
    else
        jl_atomic_store_relaxed(&ptls->gc_num.allocd,
            jl_atomic_load_relaxed(&ptls->gc_num.allocd) + (allocsz - oldsz));
    jl_atomic_store_relaxed(&ptls->gc_num.realloc,
        jl_atomic_load_relaxed(&ptls->gc_num.realloc) + 1);

    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *b;
    if (isaligned)
        b = realloc_cache_align(d, allocsz, oldsz);
    else
        b = realloc(d, allocsz);
    if (b == NULL)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    // gc_managed_realloc_ is currently used exclusively for resizing array buffers.
    if (allocsz > oldsz) {
        maybe_record_alloc_to_profile((jl_value_t*)b, allocsz - oldsz, (jl_datatype_t*)jl_buff_tag);
    }
    return b;
}

JL_DLLEXPORT void *jl_gc_managed_realloc(void *d, size_t sz, size_t oldsz,
                                         int isaligned, jl_value_t *owner)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return gc_managed_realloc_(ptls, d, sz, oldsz, isaligned, owner, 1);
}

uv_mutex_t gc_perm_lock;

JL_DLLEXPORT void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_add_finalizer_th(ptls, v, f);
}

JL_DLLEXPORT void jl_finalize(jl_value_t *o)
{
    jl_finalize_th(jl_current_task, o);
}

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref_th(jl_ptls_t ptls,
                                                jl_value_t *value)
{
    jl_weakref_t *wr = (jl_weakref_t*)jl_gc_alloc(ptls, sizeof(void*),
                                                  jl_weakref_type);
    wr->value = value;  // NOTE: wb not needed here
    // We need this pin. The weak refs are stored in an arraylist.
    OBJ_PIN(wr);
    // This pin is not necessary. We can deal with object forwarding in weak ref processing.
    // Pinning is simpler.
    OBJ_PIN(value);
    small_arraylist_push(&ptls->heap.weak_refs, wr);
    return wr;
}

JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref(jl_value_t *value)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_new_weakref_th(ptls, value);
}

JL_DLLEXPORT jl_value_t *jl_gc_allocobj(size_t sz)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sz, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_0w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, 0, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_1w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void*), NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_2w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void*) * 2, NULL);
}

JL_DLLEXPORT jl_value_t *jl_gc_alloc_3w(void)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return jl_gc_alloc(ptls, sizeof(void*) * 3, NULL);
}

JL_DLLEXPORT size_t jl_gc_max_internal_obj_size(void)
{
    // TODO: meaningful for MMTk?
    return GC_MAX_SZCLASS;
}

JL_DLLEXPORT size_t jl_gc_external_obj_hdr_size(void)
{
    return sizeof(bigval_t);
}


JL_DLLEXPORT void * jl_gc_alloc_typed(jl_ptls_t ptls, size_t sz, void *ty)
{
    return jl_gc_alloc(ptls, sz, ty);
}

JL_DLLEXPORT void jl_gc_schedule_foreign_sweepfunc(jl_ptls_t ptls, jl_value_t *obj)
{
    // TODO: correct for MMTk?
    arraylist_push(&ptls->sweep_objs, obj);
}


// gc-debug common functions
// ---

int gc_slot_to_fieldidx(void *obj, void *slot, jl_datatype_t *vt) JL_NOTSAFEPOINT
{
    int nf = (int)jl_datatype_nfields(vt);
    for (int i = 1; i < nf; i++) {
        if (slot < (void*)((char*)obj + jl_field_offset(vt, i)))
            return i - 1;
    }
    return nf - 1;
}

int gc_slot_to_arrayidx(void *obj, void *_slot) JL_NOTSAFEPOINT
{
    char *slot = (char*)_slot;
    jl_datatype_t *vt = (jl_datatype_t*)jl_typeof(obj);
    char *start = NULL;
    size_t len = 0;
    size_t elsize = sizeof(void*);
    if (vt == jl_module_type) {
        jl_module_t *m = (jl_module_t*)obj;
        start = (char*)m->usings.items;
        len = m->usings.len;
    }
    else if (vt == jl_simplevector_type) {
        start = (char*)jl_svec_data(obj);
        len = jl_svec_len(obj);
    }
    else if (vt->name == jl_array_typename) {
        jl_array_t *a = (jl_array_t*)obj;
        start = (char*)a->data;
        len = jl_array_len(a);
        elsize = a->elsize;
    }
    if (slot < start || slot >= start + elsize * len)
        return -1;
    return (slot - start) / elsize;
}

static int gc_logging_enabled = 0;

JL_DLLEXPORT void jl_enable_gc_logging(int enable) {
    gc_logging_enabled = enable;
}

JL_DLLEXPORT void jl_save_context_for_conservative_scanning(jl_ptls_t ptls, void *ctx)
{
#ifdef GC_SAVE_CONTEXT_FOR_CONSERVATIVE_SCANNING
    if (ctx == NULL) {
        // Save the context for the thread as it was running at the time of the call
        int r = getcontext(&ptls->ctx_at_the_time_gc_started);
        if (r == -1) {
            jl_safe_printf("Failed to save context for conservative scanning\n");
            abort();
        }
        return;
    }
    memcpy(&ptls->ctx_at_the_time_gc_started, ctx, sizeof(ucontext_t));
#endif
}
