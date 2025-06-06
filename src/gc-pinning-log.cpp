// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <cassert>
#include <iostream>
#include <map>
#include <mutex>

#include "julia.h"

struct pinning_site_t {
    int lineno;
    const char *filename;
    pinning_site_t() : lineno(0), filename(nullptr) {}
    pinning_site_t(int line, const char *file) : lineno(line), filename(file) {}
    bool operator<(const pinning_site_t &other) const {
        if (lineno != other.lineno) {
            return lineno < other.lineno;
        }
        return filename < other.filename;
    }
    bool operator==(const pinning_site_t &other) const {
        return lineno == other.lineno && filename == other.filename;
    }
};

struct pinning_log_entry_t {
    void *pinned_object;
    pinning_site_t site;
    pinning_log_entry_t() : pinned_object(nullptr), site(pinning_site_t{}) {}
};

struct linear_pinning_log_t {
#define BUFFER_CAPACITY (1ULL << 20)
    size_t idx;
    pinning_log_entry_t buffer[BUFFER_CAPACITY];
    linear_pinning_log_t() : idx(0) {
        for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
            buffer[i] = pinning_log_entry_t{};
        }
    }
    pinning_log_entry_t *bump_alloc_log_entry() {
        if (idx >= BUFFER_CAPACITY) {
            assert(0 && "Exceeded buffer capacity");
        }
        auto e = &buffer[idx++];
        return e;
    }
    void reset_log_buffer() {
        idx = 0;
        for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
            buffer[i] = pinning_log_entry_t{};
        }
    }
};

struct coalesced_pinning_log_t {
    std::map<void *, std::map<pinning_site_t, size_t>> objects_to_pinning_sites;
    coalesced_pinning_log_t() = default;
    void add_pinning_event(void *pinned_object, const pinning_site_t &site) {
        if (objects_to_pinning_sites.find(pinned_object) == objects_to_pinning_sites.end()) {
            std::map<pinning_site_t, size_t> site_map{};
            objects_to_pinning_sites[pinned_object] = std::map<pinning_site_t, size_t>{};
        }
        auto &site_map = objects_to_pinning_sites[pinned_object];
        if (site_map.find(site) == site_map.end()) {
            site_map[site] = 0;
        }
        site_map[site]++;
    }
};

struct pinning_log_t {
    linear_pinning_log_t linear_log;
    coalesced_pinning_log_t coalesced_log;
    check_alive_fn is_alive;
    std::mutex mu;
    pinning_log_entry_t *alloc_pinning_log_entry(void *pinned_object) {
        pinning_log_entry_t *e;
        mu.lock();
        e = linear_log.bump_alloc_log_entry();
        mu.unlock();
        return e;
    }
    void coalesce_linear_pinning_log() {
        mu.lock();
        for (size_t i = 0; i < linear_log.idx; ++i) {
            auto &entry = linear_log.buffer[i];
            if (entry.pinned_object != nullptr) {
                coalesced_log.add_pinning_event(entry.pinned_object, entry.site);
            }
        }
        linear_log.reset_log_buffer();
        mu.unlock();
    }
    void set_check_alive_fn(check_alive_fn fn) {
        mu.lock();
        is_alive = fn;
        mu.unlock();
    }
    void gc_log(void) {
        coalesce_linear_pinning_log();
        mu.lock();
        for (auto it = coalesced_log.objects_to_pinning_sites.begin(); it != coalesced_log.objects_to_pinning_sites.end();) {
            if (!is_alive(reinterpret_cast<jl_value_t *>(it->first))) {
                it = coalesced_log.objects_to_pinning_sites.erase(it);
            } else {
                ++it;
            }
        }
        mu.unlock();
    }
    void print_pinning_log_as_json() {
        mu.lock();
        std::cerr << "[\n";
        bool first = true;
        for (const auto &object_entry : coalesced_log.objects_to_pinning_sites) {
            if (!first) {
                std::cerr << ",\n";
            }
            first = false;
            std::cerr << "  {\n";
            std::cerr << "    \"pinned_object\": \"" << object_entry.first << "\",\n";
            const char *type;
            if (is_alive(reinterpret_cast<jl_value_t *>(object_entry.first))) {
                type = jl_typeof_str(reinterpret_cast<jl_value_t *>(object_entry.first));
            } else {
                type = "unknown";
            }
            std::cerr << "    \"type\": \"" << type << "\",\n";
            std::cerr << "    \"pinning_sites\": [\n";
            bool first_site = true;
            for (const auto &site_entry : object_entry.second) {
                if (!first_site) {
                    std::cerr << ",\n";
                }
                first_site = false;
                std::cerr << "      {\n";
                auto filename = site_entry.first.filename ? site_entry.first.filename : "unknown";
                std::cerr << "        \"filename\": \"" << filename << "\",\n";
                std::cerr << "        \"lineno\": " << site_entry.first.lineno << ",\n";
                std::cerr << "        \"count\": " << site_entry.second << "\n";
                std::cerr << "      }";
            }
            std::cerr << "\n    ]\n";
            std::cerr << "  }";
        }
        std::cerr << "\n]\n";
        mu.unlock();
    }
};

pinning_log_t pinning_log;

#ifdef __cplusplus
extern "C" {
#endif

int pinning_log_enabled;

JL_DLLEXPORT void jl_set_check_alive_fn(check_alive_fn fn) {
    pinning_log.set_check_alive_fn(fn);
}
JL_DLLEXPORT void jl_enable_pinning_log(void) {
    pinning_log_enabled = 1;
}
JL_DLLEXPORT void jl_gc_log(void) {
    if (!pinning_log_enabled) {
        return;
    }
    pinning_log.gc_log();
}
JL_DLLEXPORT void jl_log_pinning_event(void *pinned_object, const char *filename, int lineno) {
    if (!pinning_log_enabled) {
        return;
    }
    auto pe = pinning_log.alloc_pinning_log_entry(pinned_object);
    pe->pinned_object = pinned_object;
    pe->site.lineno = lineno;
    pe->site.filename = filename;
}
JL_DLLEXPORT void jl_print_pinning_log(void) {
    if (!pinning_log_enabled) {
        return;
    }
    pinning_log.coalesce_linear_pinning_log();
    pinning_log.print_pinning_log_as_json();
    jl_safe_printf("=========================\n");
}

#ifdef __cplusplus
}
#endif
