// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "gc.h"

jl_gc_num_t gc_num = {0};
size_t last_long_collect_interval;
int gc_n_threads;
jl_ptls_t* gc_all_tls_states;

int64_t live_bytes = 0;

JL_DLLEXPORT _Atomic(int) jl_gc_have_pending_finalizers = 0;

// mutex for gc-heap-snapshot.
jl_mutex_t heapsnapshot_lock;

const uint64_t _jl_buff_tag[3] = {0x4eadc0004eadc000ull, 0x4eadc0004eadc000ull, 0x4eadc0004eadc000ull}; // aka 0xHEADER00
JL_DLLEXPORT uintptr_t jl_get_buff_tag(void)
{
    return jl_buff_tag;
}

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

JL_DLLEXPORT void jl_gc_init_finalizer_rng_state(void)
{
    jl_rng_split(finalizer_rngState, jl_current_task->rngState);
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

// This wrapper exists only to prevent `jl_gc_big_alloc_inner` from being
// inlined into its callers. We provide an external-facing interface for
// callers, and inline `jl_gc_big_alloc_inner` into this. (See
// https://github.com/JuliaLang/julia/pull/43868 for more details.)
jl_value_t *jl_gc_big_alloc_noinline(jl_ptls_t ptls, size_t sz)
{
    return jl_gc_big_alloc_inner(ptls, sz);
}

// Instrumented version of jl_gc_pool_alloc_inner, called into by LLVM-generated code.
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

int jl_gc_classify_pools(size_t sz, int *osize)
{
    if (sz > GC_MAX_SZCLASS)
        return -1;
    size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    int klass = jl_gc_szclass(allocsz);
    *osize = jl_gc_sizeclasses[klass];
    return (int)(intptr_t)(&((jl_ptls_t)0)->heap.norm_pools[klass]);
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
        if (ptls) {
            memset(&ptls->gc_num, 0, sizeof(ptls->gc_num));
            jl_atomic_store_relaxed(&ptls->gc_num.allocd, -(int64_t)gc_num.interval);
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
    // avoid counting remembered objects
    // in `perm_scanned_bytes`
    size_t len = remset->len;
    void **items = remset->items;
    for (size_t i = 0; i < len; i++) {
        jl_value_t *item = (jl_value_t *)items[i];
        objprofile_count(jl_typeof(item), 2, 0);
        jl_astaggedvalue(item)->bits.gc = GC_OLD_MARKED;
    }
}



// GC control
// ---

_Atomic(uint32_t) jl_gc_disable_counter = 1;

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
            enable_collection();
        }
    }
    else if (prev && !on) {
        disable_collection();
        // enable -> disable
        jl_atomic_fetch_add(&jl_gc_disable_counter, 1);
        // check if the GC is running and wait for it to finish
        jl_gc_safepoint_(ptls);
    }
    return prev;
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

JL_DLLEXPORT void jl_gc_set_max_memory(uint64_t max_mem)
{
    if (max_mem > 0 && max_mem < (uint64_t)1 << (sizeof(memsize_t) * 8 - 1))
        max_total_memory = max_mem;
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

    // TODO: not needed? gc_cache.*?
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
    maybe_record_alloc_to_profile((jl_value_t*)b, sz, jl_gc_unknown_type_tag);
    return b;
}

JL_DLLEXPORT void *jl_gc_managed_realloc(void *d, size_t sz, size_t oldsz,
                                         int isaligned, jl_value_t *owner)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    return gc_managed_realloc_(ptls, d, sz, oldsz, isaligned, owner, 1);
}

// Perm gen allocator
// 2M pool
#define GC_PERM_POOL_SIZE (2 * 1024 * 1024)
// 20k limit for pool allocation. At most 1% fragmentation
#define GC_PERM_POOL_LIMIT (20 * 1024)
uv_mutex_t gc_perm_lock;
static uintptr_t gc_perm_pool = 0;
static uintptr_t gc_perm_end = 0;

static void *gc_perm_alloc_large(size_t sz, int zero, unsigned align, unsigned offset) JL_NOTSAFEPOINT
{
    // `align` must be power of two
    assert(offset == 0 || offset < align);
    const size_t malloc_align = sizeof(void*) == 8 ? 16 : 4;
    if (align > 1 && (offset != 0 || align > malloc_align))
        sz += align - 1;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *base = zero ? calloc(1, sz) : malloc(sz);
    if (base == NULL)
        jl_throw(jl_memory_exception);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    jl_may_leak(base);
    assert(align > 0);
    unsigned diff = (offset - (uintptr_t)base) % align;
    return (void*)((char*)base + diff);
}

STATIC_INLINE void *gc_try_perm_alloc_pool(size_t sz, unsigned align, unsigned offset) JL_NOTSAFEPOINT
{
    uintptr_t pool = LLT_ALIGN(gc_perm_pool + offset, (uintptr_t)align) - offset;
    uintptr_t end = pool + sz;
    if (end > gc_perm_end)
        return NULL;
    gc_perm_pool = end;
    return (void*)jl_assume(pool);
}

// **NOT** a safepoint
void *jl_gc_perm_alloc_nolock(size_t sz, int zero, unsigned align, unsigned offset)
{
    // The caller should have acquired `gc_perm_lock`
    assert(align < GC_PERM_POOL_LIMIT);
#ifndef MEMDEBUG
    if (__unlikely(sz > GC_PERM_POOL_LIMIT))
#endif
        return gc_perm_alloc_large(sz, zero, align, offset);
    void *ptr = gc_try_perm_alloc_pool(sz, align, offset);
    if (__likely(ptr))
        return ptr;
    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
    void *pool = VirtualAlloc(NULL, GC_PERM_POOL_SIZE, MEM_COMMIT, PAGE_READWRITE);
    SetLastError(last_error);
    errno = last_errno;
    if (__unlikely(pool == NULL))
        return NULL;
#else
    void *pool = mmap(0, GC_PERM_POOL_SIZE, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    errno = last_errno;
    if (__unlikely(pool == MAP_FAILED))
        return NULL;
#endif
    gc_perm_pool = (uintptr_t)pool;
    gc_perm_end = gc_perm_pool + GC_PERM_POOL_SIZE;
    return gc_try_perm_alloc_pool(sz, align, offset);
}

// **NOT** a safepoint
void *jl_gc_perm_alloc(size_t sz, int zero, unsigned align, unsigned offset)
{
    assert(align < GC_PERM_POOL_LIMIT);
#ifndef MEMDEBUG
    if (__unlikely(sz > GC_PERM_POOL_LIMIT))
#endif
        return gc_perm_alloc_large(sz, zero, align, offset);
    uv_mutex_lock(&gc_perm_lock);
    void *p = jl_gc_perm_alloc_nolock(sz, zero, align, offset);
    uv_mutex_unlock(&gc_perm_lock);
    return p;
}

JL_DLLEXPORT void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_gc_add_finalizer_th(ptls, v, f);
}

JL_DLLEXPORT void jl_finalize(jl_value_t *o)
{
    jl_finalize_th(jl_current_task, o);
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

#ifdef __cplusplus
}
#endif
