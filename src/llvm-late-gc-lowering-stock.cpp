#include "llvm-gc-interface-passes.h"

void LateLowerGCFrame::CleanupGCPreserve(Function &F, CallInst *CI, Value *callee, Type *T_size) {
    // Do nothing for the stock GC
}
