#include "llvm-gc-interface-passes.h"

bool LateLowerGCFrameCustom::runOnFunction(Function &F, bool *CFGModified) {
    need_gc_preserve_hook = 1;
    return LateLowerGCFrame::runOnFunction(F, CFGModified);
}
