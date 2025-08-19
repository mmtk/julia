# This file is a part of Julia. License is MIT: https://julialang.org/license

export increment_pin_count!, decrement_pin_count!,
       increment_tpin_count!, decrement_tpin_count!,
       get_pin_count, get_tpin_count

"""
    increment_pin_count!(obj)

Increment the pin count of `obj` to preserve it beyond a lexical scope.

This ensures that `obj` is not moved or collected by the garbage collector.
It is crucial for safely passing references to foreign code.

Each call increments the pin count by one. The object remains pinned and
alive until the count is decremented to zero via `decrement_pin_count!`.

# Examples

```julia
x = SomeObject()
increment_pin_count!(x)   # x is now pinned (count = 1)

increment_pin_count!(x)   # pin count is now 2
"""
function increment_pin_count!(obj)
    ccall(:jl_increment_pin_count, Cvoid, (Any,), obj)
end

"""
    decrement_pin_count!(obj)

Decrement the pin count of `obj`.

When the count drops to zero, the object is no longer pinned and may be
moved (or collected by the garbage collector, if no other references
exist).

This is necessary to release objects that were previously preserved for
foreign code.

# Examples

```julia
x = SomeObject()
increment_pin_count!(x)   # x is now pinned (count = 1)

increment_pin_count!(x)   # pin count is now 2

decrement_pin_count!(x)   # reduces pin count to 1

decrement_pin_count!(x)   # count is 0; x may now be collected
"""
function decrement_pin_count!(obj)
    ccall(:jl_decrement_pin_count, Cvoid, (Any,), obj)
end

"""
    increment_tpin_count!(obj)

Increment the transitive pin count of `obj` to preserve it beyond a
lexical scope.

This ensures that `obj` and any other objects reachable from it are not
moved or collected by the garbage collector. This is crucial for safely
passing references to foreign code.

Each call increments the transitive pin count by one. The object remains
transitively pinned and alive until the count is decremented to zero via
`decrement_tpin_count!`.

# Examples

```julia
x = SomeObject()
increment_tpin_count!(x)   # x is now transitively pinned (count = 1)

increment_tpin_count!(x)   # transitive pin count is now 2
"""
function increment_tpin_count!(obj)
    ccall(:jl_increment_tpin_count, Cvoid, (Any,), obj)
end

"""
    decrement_tpin_count!(obj)

Decrement the transitive pin count of `obj`.

When the count drops to zero, `obj` and any objects reachable from it are
no longer pinned and may be moved (if no other pins exist) or collected by
the garbage collector (if no other references exist).

This is necessary to release object graphs that were previously preserved
for foreign code.

# Examples

```julia
x = SomeObject()
increment_tpin_count!(x)     # pins x and reachable objects (count = 1)

decrement_tpin_count!(x)     # reduces transitive pin count to 0
                             # objects may now be collected
"""
function decrement_tpin_count!(obj)
    ccall(:jl_decrement_tpin_count, Cvoid, (Any,), obj)
end

"""
    get_pin_count(obj)

Return the current pin count of `obj`.

This indicates how many times `obj` has been explicitly pinned via
`increment_pin_count!`. A nonzero count means the object is currently
pinned and will not be moved or collected by the garbage collector (GC).

# Examples

```julia
x = SomeObject()
increment_pin_count!(x)
get_pin_count(x)   # returns 1

increment_pin_count!(x)
get_pin_count(x)   # returns 2

decrement_pin_count!(x)
get_pin_count(x)   # returns 1
"""
function get_pin_count(obj)
    c = ccall(:jl_get_pin_count, Csize_t, (Any,), obj)
    return Int64(c)
end

"""
    get_tpin_count(obj)

Return the current transitive pin count of `obj`.

This indicates how many times `obj` has been explicitly transitively pinned
via `increment_tpin_count!`. A nonzero count means `obj` and all objects
reachable from it are currently pinned and will not be moved or collected
by the garbage collector (GC).

# Examples

```julia
x = SomeObject()
increment_tpin_count!(x)
get_tpin_count(x)   # returns 1

increment_tpin_count!(x)
get_tpin_count(x)   # returns 2

decrement_tpin_count!(x)
get_tpin_count(x)   # returns 1
"""
function get_tpin_count(obj)
    c = ccall(:jl_get_tpin_count, Csize_t, (Any,), obj)
    return Int64(c)
end
