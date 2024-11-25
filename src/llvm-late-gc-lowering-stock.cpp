#include "llvm-gc-interface-passes.h"

bool LateLowerGCFrameCustom::runOnFunction(Function &F, bool *CFGModified) {
    need_gc_preserve_hook = 0;
    LateLowerGCFrame::runOnFunction(F, CFGModified);
}
