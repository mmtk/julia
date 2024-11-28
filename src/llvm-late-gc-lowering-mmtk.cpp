#include "llvm-gc-interface-passes.h"

void LateLowerGCFrame::CleanupGCPreserve(Function &F, CallInst *CI, Value *callee, Type *T_size) {
    if (callee == gc_preserve_begin_func) {
        // Initialize an IR builder.
        IRBuilder<> builder(CI);

        builder.SetCurrentDebugLocation(CI->getDebugLoc());
        size_t nargs = 0;
        State S2(F);

        std::vector<Value*> args;
        for (Use &U : CI->args()) {
            Value *V = U;
            if (isa<Constant>(V))
                continue;
            if (isa<PointerType>(V->getType())) {
                if (isSpecialPtr(V->getType())) {
                    int Num = Number(S2, V);
                    if (Num >= 0) {
                        nargs++;
                        Value *Val = GetPtrForNumber(S2, Num, CI);
                        args.push_back(Val);
                    }
                }
            } else {
                auto Nums = NumberAll(S2, V);
                for (int Num : Nums) {
                    if (Num < 0)
                        continue;
                    Value *Val = GetPtrForNumber(S2, Num, CI);
                    args.push_back(Val);
                    nargs++;
                }
            }
        }
        args.insert(args.begin(), ConstantInt::get(T_size, nargs));

        ArrayRef<Value*> args_llvm = ArrayRef<Value*>(args);
        builder.CreateCall(getOrDeclare(jl_well_known::GCPreserveBeginHook), args_llvm );
    } else if (callee == gc_preserve_end_func) {
        // Initialize an IR builder.
        IRBuilder<> builder(CI);
        builder.SetCurrentDebugLocation(CI->getDebugLoc());
        builder.CreateCall(getOrDeclare(jl_well_known::GCPreserveEndHook), {});
    }
}
