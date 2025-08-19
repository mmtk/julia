// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <map>
#include <mutex>

#include "julia.h"

// Let's just use global maps in this first implementation
// They could cause contention in multi-threaded code, so we might need to optimize them later

// Pinning
std::map<void *, size_t> pin_count_map;
std::mutex pin_count_map_lock;
// Transitive Pinning
std::map<void *, size_t> tpin_count_map;
std::mutex tpin_count_map_lock;

#ifdef __cplusplus
extern "C" {
#endif

// Pinning
JL_DLLEXPORT void jl_increment_pin_count(void *obj) {
    pin_count_map_lock.lock();
    if (pin_count_map.find(obj) == pin_count_map.end()) {
        pin_count_map[obj] = 0;
    }
    pin_count_map[obj]++;
    pin_count_map_lock.unlock();
}
JL_DLLEXPORT void jl_decrement_pin_count(void *obj) {
    pin_count_map_lock.lock();
    auto it = pin_count_map.find(obj);
    if (it != pin_count_map.end()) {
        if (it->second == 1) {
            pin_count_map.erase(it);
        } else {
            it->second--;
        }
    }
    pin_count_map_lock.unlock();
}

// Transitive Pinning
JL_DLLEXPORT void jl_increment_tpin_count(void *obj) {
    tpin_count_map_lock.lock();
    if (tpin_count_map.find(obj) == tpin_count_map.end()) {
        tpin_count_map[obj] = 0;
    }
    tpin_count_map[obj]++;
    tpin_count_map_lock.unlock();
}
JL_DLLEXPORT void jl_decrement_tpin_count(void *obj) {
    tpin_count_map_lock.lock();
    auto it = tpin_count_map.find(obj);
    if (it != tpin_count_map.end()) {
        if (it->second == 1) {
            tpin_count_map.erase(it);
        } else {
            it->second--;
        }
    }
    tpin_count_map_lock.unlock();
}

// Retrieve Pinning and Transitive Pinning counts for a given object
JL_DLLEXPORT size_t jl_get_pin_count(void *obj) {
    pin_count_map_lock.lock();
    auto it = pin_count_map.find(obj);
    size_t count = (it != pin_count_map.end()) ? it->second : 0;
    pin_count_map_lock.unlock();
    return count;
}
JL_DLLEXPORT size_t jl_get_tpin_count(void *obj) {
    tpin_count_map_lock.lock();
    auto it = tpin_count_map.find(obj);
    size_t count = (it != tpin_count_map.end()) ? it->second : 0;
    tpin_count_map_lock.unlock();
    return count;
}

// Returns all pinned and transitively pinned objects
// Argument should have been initialized by the caller
// TODO: add a few assertions to check this?
JL_DLLEXPORT void jl_dump_all_pinned_objects(arraylist_t *objects) {
    pin_count_map_lock.lock();
    for (const auto &pair : pin_count_map) {
        arraylist_push(objects, pair.first);
    }
    pin_count_map_lock.unlock();
}
JL_DLLEXPORT void jl_dump_all_tpinned_objects(arraylist_t *objects) {
    tpin_count_map_lock.lock();
    for (const auto &pair : tpin_count_map) {
        arraylist_push(objects, pair.first);
    }
    tpin_count_map_lock.unlock();
}

#ifdef __cplusplus
}
#endif
