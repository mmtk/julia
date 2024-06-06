// This file is a part of Julia. License is MIT: https://julialang.org/license

// RUN: clang -D__clang_gcanalyzer__ --analyze -Xanalyzer -analyzer-output=text -Xclang -load -Xclang libGCCheckerPlugin%shlibext -I%julia_home/src -I%julia_home/src/support -I%julia_home/usr/include ${CLANGSA_FLAGS} ${CLANGSA_CXXFLAGS} ${CPPFLAGS} ${CFLAGS} -Xclang -analyzer-checker=core,julia.GCChecker --analyzer-no-default-checks -Xclang -verify -v -x c %s

#include "julia.h"
#include "julia_internal.h"

extern void look_at_value(jl_value_t *v);
extern void process_unrooted(jl_value_t *maybe_unrooted JL_MAYBE_UNROOTED);

void unpinned_argument() {
    jl_svec_t *val = jl_svec1(NULL);
    JL_GC_PROMISE_ROOTED(val);
    look_at_value((jl_value_t*) val);
}

int unrooted_argument() {
  jl_svec_t *val = jl_svec1(NULL);
  process_unrooted((jl_value_t*)val);
}

extern void process_pinned(jl_value_t *pinned JL_REQUIRE_PIN);

void non_pinned_argument_for_require_pin() {
    jl_svec_t *val = jl_svec1(NULL); // expected-note{{Started tracking value here}}
    JL_GC_PROMISE_ROOTED(val);       // expected-note{{Value was rooted here}}
    process_pinned(val);             // expected-warning{{Passing non-pinned argument to function that requires a pin argument}}
                                     // expected-note@-1{{Passing non-pinned argument to function that requires a pin argument}}
}

void pinned_argument_for_require_pin() {
    jl_svec_t *val = jl_svec1(NULL);
    JL_GC_PROMISE_ROOTED(val);
    PTR_PIN(val);
    process_pinned(val);
    PTR_UNPIN(val);
}

void cannot_use_moved_value_for_arg() {
    jl_svec_t *val = jl_svec1(NULL);    // expected-note{{Started tracking value here}}
    JL_GC_PROMISE_ROOTED(val);          // expected-note{{Value was rooted here}}
    jl_gc_safepoint();                  // expected-note{{Value was moved here}}
    look_at_value((jl_value_t*) val);   // expected-warning{{Argument value may have been moved}} // <<< here -- the value is used, and it is moved -- it is wrong.
                                        // expected-note@-1{{Argument value may have been moved}}
}

void pin_after_safepoint() {
    jl_svec_t *val = jl_svec1(NULL);
    JL_GC_PROMISE_ROOTED(val);
    jl_gc_safepoint();
    PTR_PIN(val); // expected-warning{{Attempt to PIN a value that is already moved}}
                  // expected-note@-1{{Attempt to PIN a value that is already moved}}
    look_at_value((jl_value_t*) val);
}

void proper_pin_before_safepoint() {
    jl_svec_t *val = jl_svec1(NULL);
    JL_GC_PROMISE_ROOTED(val);
    PTR_PIN(val);
    jl_gc_safepoint();
    look_at_value((jl_value_t*) val);
    PTR_UNPIN(val);
}

extern void process_tpinned(jl_value_t *tpinned JL_REQUIRE_TPIN);

void push_tpin_value() {
    jl_svec_t *val = jl_svec1(NULL);
    JL_GC_PUSH1(&val);
    jl_gc_safepoint();
    process_tpinned((jl_value_t*) val);
    JL_GC_POP();
}

void push_no_tpin_value() {
    jl_svec_t *val = jl_svec1(NULL);    // expected-note{{Started tracking value here}}
    JL_GC_PUSH1_NO_TPIN(&val);          // expected-note{{GC frame changed here}}
                                        // expected-note@-1{{Value was rooted here}}
    jl_gc_safepoint();
    process_tpinned((jl_value_t*) val); // expected-warning{{Passing non-tpinned argument to function that requires a tpin argument}}
                                        // expected-note@-1{{Passing non-tpinned argument to function that requires a tpin argument}}
    JL_GC_POP();
}

void pointer_to_pointer(jl_value_t **v) {
    // *v is not pinned.
    look_at_value(*v);
}

void pointer_to_pointer2(jl_value_t* u, jl_value_t **v) {
    *v = u;
    look_at_value(*v);
}

extern jl_value_t *first_array_elem(jl_array_t *a JL_PROPAGATES_ROOT);

void root_propagation(jl_expr_t *expr) {
  jl_value_t *val = first_array_elem(expr->args); // expected-note{{Started tracking value here}}
  jl_gc_safepoint();  // expected-note{{Value was moved here}}
  look_at_value(val); // expected-warning{{Argument value may have been moved}}
                      // expected-note@-1{{Argument value may have been moved}}
}

void derive_ptr_alias(jl_method_instance_t *mi) {
  jl_value_t* a = mi->specTypes;
  jl_value_t* b = mi->specTypes;
  look_at_value(b);
}

void derive_ptr_alias2(jl_method_instance_t *mi) {
  look_at_value(mi->specTypes);
}

// Ignore this case for now. The checker conjures new syms for function return values.
// It pins the first return value, but cannot see the second return value is an alias of the first.
// However, we could rewrite the code so the checker can check it.
// void mtable(jl_value_t *f) {
//   PTR_PIN((jl_value_t*)jl_gf_mtable(f));
//   look_at_value((jl_value_t*)jl_gf_mtable(f));
// }

void mtable(jl_value_t *f) {
    jl_value_t* mtable = (jl_value_t*)jl_gf_mtable(f);
    look_at_value(mtable);
}

void pass_arg_to_non_safepoint(jl_tupletype_t *sigt) {
    jl_value_t *ati = jl_tparam(sigt, 0);
}

// Though the code loads the pointer after the safepoint, we don't know if the compiler would hoist the load before the safepoint.
// So it is fine that the checker reports this as an error.
void load_new_pointer_after_safepoint(jl_tupletype_t *t) {
    jl_value_t *a0 = jl_svecref(((jl_datatype_t*)(t))->parameters, 0);//expected-note{{Started tracking value here}}
    jl_safepoint();                                                   //expected-note{{Value was moved here}}
    jl_value_t *a1 = jl_svecref(((jl_datatype_t*)(t))->parameters, 1);//expected-warning{{Argument value may have been moved}}
                                                                      //expected-note@-1{{Argument value may have been moved}}
}

void hoist_load_before_safepoint(jl_tupletype_t *t) {
    jl_svec_t* params = ((jl_datatype_t*)(t))->parameters; //expected-note{{Started tracking value here}}
    jl_value_t *a0 = jl_svecref(params, 0);
    jl_safepoint();                         //expected-note{{Value was moved here}}
    jl_value_t *a1 = jl_svecref(params, 1); //expected-warning{{Argument value may have been moved}}
                                            //expected-note@-1{{Argument value may have been moved}}
}

// We tpin a local var, and later rebind a value to the local val. The value should be considered as pinned.
void rebind_tpin(jl_method_instance_t *mi, size_t world) {
    jl_code_info_t *src = NULL;
    JL_GC_PUSH1(&src);
    PTR_PIN(mi);
    jl_value_t *ci = jl_rettype_inferred(mi, world, world);
    PTR_UNPIN(mi);
    jl_code_instance_t *codeinst = (ci == jl_nothing ? NULL : (jl_code_instance_t*)ci);
    if (codeinst) {
        src = (jl_code_info_t*)jl_atomic_load_relaxed(&codeinst->inferred);
        src = jl_uncompress_ir(mi->def.method, codeinst, (jl_array_t*)src);
    }
    JL_GC_POP();
}

void rebind_tpin_simple1() {
    jl_value_t *t = NULL;
    JL_GC_PUSH1(&t);
    jl_svec_t *v = jl_svec1(NULL);
    t = (jl_value_t*)v;
    process_tpinned(t);
    JL_GC_POP();
}

void rebind_tpin_simple2() {
    jl_value_t *t = NULL;
    JL_GC_PUSH1(&t);
    jl_svec_t *v = jl_svec1(NULL);
    t = (jl_value_t*)v;
    process_tpinned(v);
    JL_GC_POP();
}

int transitive_closure(jl_value_t *v JL_REQUIRE_TPIN) {
    if (jl_is_unionall(v)) {
        jl_unionall_t *ua = (jl_unionall_t*)v;
        return transitive_closure(ua->body);
    }
    return 0;
}

int properly_tpin_arg(jl_value_t *v) {
    JL_GC_PUSH1(&v);
    process_tpinned(v);
    JL_GC_POP();
}

int no_tpin_arg(jl_value_t *v) {
    process_tpinned(v); // expected-warning{{Passing non-tpinned argument to function that requires a tpin argument}}
                           // expected-note@-1{{Passing non-tpinned argument to function that requires a tpin argument}}
                           // expected-note@+1{{Started tracking value here (root was inherited)}}
}

jl_value_t *return_value_propagate(jl_value_t *t JL_PROPAGATES_ROOT);
void return_value_should_not_be_moved_propagated(jl_value_t *t)
{
    jl_value_t *ret = return_value_propagate(t);
    // ret should not be moved at this point
    PTR_PIN(ret);
}

jl_value_t *return_value();
void return_value_should_be_not_be_moved() {
    jl_value_t *ret = return_value();
    // ret should not be moved at this point
    PTR_PIN(ret);
}
