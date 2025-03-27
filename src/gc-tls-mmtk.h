// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JL_GC_TLS_H
#define JL_GC_TLS_H

#include <assert.h>
#include "mmtkMutator.h"
#include "julia_atomics.h"

// VO bit is required to support conservative stack scanning and moving.
#define MMTK_NEEDS_VO_BIT (1)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    MMTkMutatorContext mmtk_mutator;
    _Atomic(size_t) malloc_sz_since_last_poll;
    ucontext_t ctx_at_the_time_gc_started;
} jl_gc_tls_states_t;

#ifdef __cplusplus
}
#endif

#endif // JL_GC_TLS_H
