#include "gc-interface.h"

#ifdef MMTK_GC
// MMTk needs the compiler to insert those hooks so we can transitively pin the references.
int need_gc_preserve_hook = 1;
#else
int need_gc_preserve_hook = 0;
#endif
