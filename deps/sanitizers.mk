# Interrogate the compiler about where it is keeping its sanitizer libraries
ifeq ($(USECLANG),1)
SANITIZER_LIB_PATH := $(shell LANG=C $(CC) -print-runtime-dir)
else
SANITIZER_LIB_PATH := $(dir $(shell LANG=C $(CC) -print-file-name=libasan.so))
endif

# Given a colon-separated list of paths in $(2), find the location of the library given in $(1)
define pathsearch
$(wildcard $(addsuffix /$(1),$(subst :, ,$(2))))
endef

define copy_sanitizer_lib
install-sanitizers: $$(addprefix $$(build_libdir)/, $$(notdir $$(call pathsearch,$(1),$$(SANITIZER_LIB_PATH)))) | $$(build_shlibdir)
$$(addprefix $$(build_shlibdir)/,$(2)): $$(addprefix $$(SANITIZER_LIB_PATH)/,$(2)) | $$(build_shlibdir)
	-cp $$< $$@
endef

ifeq ($(USECLANG),1)

## Clang libraries
$(eval $(call copy_sanitizer_lib,$(call versioned_libname,libclang_rt.asan-*),$(call versioned_libname,libclang_rt.asan-%)))

endif

get-sanitizers:
clean-sanitizers:
	-rm -f $(build_shlibdir)/libclang_rt.asan*$(SHLIB_EXT)*
distclean-sanitizers: clean-sanitizers
