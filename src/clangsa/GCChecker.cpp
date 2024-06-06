// This file is a part of Julia. License is MIT: https://julialang.org/license

// Assumptions for pinning:
// * The checker may not consider alias for derived pointers in some cases.
//   * if f(x) returns a derived pointer from x, a = f(x); b = f(x); PTR_PIN(a); The checker will NOT find b as pinned.
//   * a = x->y; b = x->y; PTR_PIN(a); The checker will find b as pinned.
//   * Need to see if this affects correctness.
// * The checker may report some vals as moved even if there is a new load for the val after safepoint.
//   * f(x->a); jl_safepoint(); f(x->a); x->a is loaded after a safepoint, but the checker may report errors. This seems fine, as the compiler may hoist the load.
//   * a = x->a; f(a); jl_safepoint(); f(a); a may be moved in a safepoint, and the checker will report errors.

#include "clang/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Checkers/SValExplainer.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"

#include <iostream>
#include <memory>

#if defined(__GNUC__)
#define USED_FUNC __attribute__((used))
#else
#define USED_FUNC
#endif

using std::make_unique;

namespace {
using namespace clang;
using namespace ento;

#define PDP std::shared_ptr<PathDiagnosticPiece>
#define MakePDP make_unique<PathDiagnosticEventPiece>

static const Stmt *getStmtForDiagnostics(const ExplodedNode *N)
{
    return N->getStmtForDiagnostics();
}

// Turn on/off the log here
#define DEBUG_LOG 0

class GCChecker
    : public Checker<
          eval::Call,
          check::BeginFunction,
          check::EndFunction,
          check::PostCall,
          check::PreCall,
          check::PostStmt<CStyleCastExpr>,
          check::PostStmt<ArraySubscriptExpr>,
          check::PostStmt<MemberExpr>,
          check::PostStmt<UnaryOperator>,
          check::Bind,
          check::Location> {
  mutable std::unique_ptr<BugType> BT;
  template <typename callback>
  void report_error(callback f, CheckerContext &C, StringRef message) const;
  void report_error(CheckerContext &C, StringRef message) const {
    return report_error([](PathSensitiveBugReport *) {}, C, message);
  }
  void
  report_value_error(CheckerContext &C, SymbolRef Sym, const char *message,
                     clang::SourceRange range = clang::SourceRange()) const;

public:
  struct ValueState {
    enum State { Allocated, Rooted, PotentiallyFreed, Untracked } S;
    enum Pin { TransitivelyPinned, Pinned, NotPinned, Moved } P;
    const MemRegion *Root;
    int RootDepth;

    // Optional Metadata (for error messages)
    const FunctionDecl *FD;
    const ParmVarDecl *PVD;

    ValueState(State InS, Pin PinState, const MemRegion *Root, int Depth, const FunctionDecl *FD, const ParmVarDecl *PVD)
        : S(InS), P(PinState), Root(Root), RootDepth(Depth), FD(FD), PVD(PVD) {}
    ValueState(State InS, Pin PinState, const MemRegion *Root, int Depth)
        : S(InS), P(PinState), Root(Root), RootDepth(Depth), FD(nullptr), PVD(nullptr) {}
    ValueState()
        : S(Untracked), P(NotPinned), Root(nullptr), RootDepth(0), FD(nullptr), PVD(nullptr) {
    }

    USED_FUNC void dump() const {
      llvm::dbgs() << ((S == Allocated) ? "Allocated"
                     : (S == Rooted) ? "Rooted"
                     : (S == PotentiallyFreed) ? "PotentiallyFreed"
                     : (S == Untracked) ? "Untracked"
                     : "Error");
      llvm::dbgs() << ",";
      llvm::dbgs() << ((P == TransitivelyPinned) ? "TransitivelyPinned"
                     : (P == Pinned) ? "Pinned"
                     : (P == NotPinned) ? "NotPinned"
                     : (P == Moved) ? "Moved"
                     : "Error");
      llvm::dbgs() << ",";
      if (S == Rooted) {
        llvm::dbgs() << "Root(";
        if (Root) {
          Root->dump();
          llvm::dbgs() << ",";
        }
        llvm::dbgs() << RootDepth << ")";
      }
    }

    bool operator==(const ValueState &VS) const {
      return S == VS.S && P == VS.P && Root == VS.Root && RootDepth == VS.RootDepth;
    }
    bool operator!=(const ValueState &VS) const {
      return S != VS.S || P != VS.P || Root != VS.Root || RootDepth != VS.RootDepth;
    }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(S);
      ID.AddInteger(P);
      ID.AddPointer(Root);
      ID.AddInteger(RootDepth);
    }

    bool isRooted() const { return S == Rooted; }
    bool isPotentiallyFreed() const { return S == PotentiallyFreed; }
    bool isJustAllocated() const { return S == Allocated; }
    bool isUntracked() const { return S == Untracked; }

    bool isRootedBy(const MemRegion *R) const {
      assert(R != nullptr);
      return isRooted() && R == Root;
    }

    bool isPinnedByAnyway() const { return P == Pinned || P == TransitivelyPinned; }
    bool isPinned() const { return P == Pinned; }
    bool isTransitivelyPinned() const { return P == TransitivelyPinned; }
    bool isNotPinned() const { return P == NotPinned; }
    bool isMoved() const { return P == Moved; }

    static ValueState _getPinStateChanged(ValueState old, Pin P) {
      return ValueState(old.S, P, old.Root, old.RootDepth, old.FD, old.PVD);
    }
    static ValueState getMoved(ValueState old) {
      return _getPinStateChanged(old, Moved);
    }
    static ValueState getTransitivelyPinned(ValueState old) {
      return _getPinStateChanged(old, TransitivelyPinned);
    }
    static ValueState getPinned(ValueState old) {
      return _getPinStateChanged(old, Pinned);
    }
    static ValueState getNotPinned(ValueState old) {
      return _getPinStateChanged(old, NotPinned);
    }
    static ValueState::Pin pinState(bool tpin) {
      if (tpin)
        return TransitivelyPinned;
      else
        return Pinned;
    }
    // Inherit state from a parent object to its child object
    static ValueState inheritState(ValueState parent) {
      if (parent.isTransitivelyPinned()) {
        // If parent is tpinned, the child is tpinned.
        return parent;
      } else if (parent.isPinned()) {
        // If parent is pinned, the child is not pinned.
        return getNotPinned(parent);
      } else if (parent.isMoved()) {
        // If parent is moved, the child is not pinned.
        return getNotPinned(parent);
      } else {
        // For other cases, the children have the same state as the parent.
        return parent;
      }
    }

    static ValueState getAllocated() {
      return ValueState(Allocated, NotPinned, nullptr, -1);
    }
    static ValueState getFreed() {
      return ValueState(PotentiallyFreed, NotPinned, nullptr, -1);
    }
    static ValueState getUntracked() {
      return ValueState(Untracked, NotPinned, nullptr, -1);
    }
    static ValueState getRooted(const MemRegion *Root, int Depth) {
      return ValueState(Rooted, NotPinned, Root, Depth);
    }
    static ValueState getRooted(const MemRegion *Root, Pin P, int Depth) {
      return ValueState(Rooted, P, Root, Depth);
    }
    static ValueState getForArgument(const FunctionDecl *FD,
                                     const ParmVarDecl *PVD) {
      bool isFunctionSafepoint = !isFDAnnotatedNotSafepoint(FD);
      bool maybeUnrooted = declHasAnnotation(PVD, "julia_maybe_unrooted");
      if (!isFunctionSafepoint || maybeUnrooted) {
        ValueState VS = getAllocated();
        VS.PVD = PVD;
        VS.FD = FD;
        return VS;
      }
      bool require_tpin = declHasAnnotation(PVD, "julia_require_tpin");
      bool require_pin = declHasAnnotation(PVD, "julia_require_pin");
      if (require_tpin) {
        return getRooted(nullptr, ValueState::TransitivelyPinned, -1);
      } else if (require_pin) {
        return getRooted(nullptr, ValueState::Pinned, -1);
      } else {
        return getRooted(nullptr, ValueState::NotPinned, -1);
      }
    }
  };

  struct RootState {
    enum Kind { Root, RootArray } K;
    int RootedAtDepth;

    RootState(Kind InK, int Depth) : K(InK), RootedAtDepth(Depth) {}

    bool operator==(const RootState &VS) const {
      return K == VS.K && RootedAtDepth == VS.RootedAtDepth;
    }
    bool operator!=(const RootState &VS) const {
      return K != VS.K || RootedAtDepth != VS.RootedAtDepth;
    }

    bool shouldPopAtDepth(int Depth) const { return Depth == RootedAtDepth; }
    bool isRootArray() const { return K == RootArray; }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(K);
      ID.AddInteger(RootedAtDepth);
    }

    static RootState getRoot(int Depth) { return RootState(Root, Depth); }
    static RootState getRootArray(int Depth) {
      return RootState(RootArray, Depth);
    }

    void dump() const {
      llvm::dbgs() << ((K == Root) ? "Root"
                     : (K == RootArray) ? "RootArray"
                     : "Error");
      llvm::dbgs() << ",Depth=";
      llvm::dbgs() << RootedAtDepth;
    }
  };

  struct PinState {
    enum Kind { TransitivePin, Pin, NoPin } K;
    int PinnedAtDepth;

    PinState(Kind InK, int Depth) : K(InK), PinnedAtDepth(Depth) {}

    bool operator==(const PinState &VS) const {
      return K == VS.K && PinnedAtDepth == VS.PinnedAtDepth;
    }
    bool operator!=(const PinState &VS) const {
      return K != VS.K || PinnedAtDepth != VS.PinnedAtDepth;
    }

    bool shouldPopAtDepth(int Depth) const { return Depth == PinnedAtDepth; }
    bool isTransitivePin() const { return K == TransitivePin; }
    bool isPin() const { return K == Pin; }
    bool isAnyPin() const { return K == TransitivePin || K == Pin; }
    bool isNoPin() const { return K == NoPin; }

    void Profile(llvm::FoldingSetNodeID &ID) const {
      ID.AddInteger(K);
      ID.AddInteger(PinnedAtDepth);
    }

    static PinState getPin(int Depth) { return PinState(Pin, Depth); }
    static PinState getTransitivePin(int Depth) {
      return PinState(TransitivePin, Depth);
    }
    static PinState getNoPin(int Depth) {
      return PinState(NoPin, Depth);
    }

    void dump() const {
      llvm::dbgs() << ((K == TransitivePin) ? "TransitivePin"
                     : (K == Pin) ? "Pin"
                     : (K == NoPin) ? "NoPin"
                     : "Error");
      llvm::dbgs() << ",Depth=";
      llvm::dbgs() << PinnedAtDepth;
    }
  };

private:
  template <typename callback>
  static bool isJuliaType(callback f, QualType QT) {
    if (QT->isPointerType() || QT->isArrayType())
      return isJuliaType(
          f, clang::QualType(QT->getPointeeOrArrayElementType(), 0));
    const TypedefType *TT = QT->getAs<TypedefType>();
    if (TT) {
      if (f(TT->getDecl()->getName()))
        return true;
    }
    const TagDecl *TD = QT->getUnqualifiedDesugaredType()->getAsTagDecl();
    if (!TD) {
      return false;
    }
    return f(TD->getName());
  }
  template <typename callback>
  static SymbolRef walkToRoot(callback f, const ProgramStateRef &State,
                              const MemRegion *Region);

  static bool isGCTrackedType(QualType Type);
  static bool isGCTracked(const Expr *E);
  bool isGloballyRootedType(QualType Type) const;
  bool isGloballyTransitivelyPinnedType(QualType Type) const;
  bool isNonMovingType(QualType Type) const;
  static void dumpState(const ProgramStateRef &State);
  static bool declHasAnnotation(const clang::Decl *D, const char *which);
  static bool isFDAnnotatedNotSafepoint(const clang::FunctionDecl *FD);
  bool isSafepoint(const CallEvent &Call) const;
  bool processPotentialSafepoint(const CallEvent &Call, CheckerContext &C,
                                 ProgramStateRef &State) const;
  bool processAllocationOfResult(const CallEvent &Call, CheckerContext &C,
                                 ProgramStateRef &State) const;
  bool processArgumentRooting(const CallEvent &Call, CheckerContext &C,
                              ProgramStateRef &State) const;
  bool rootRegionIfGlobal(const MemRegion *R, ProgramStateRef &,
                          CheckerContext &C, ValueState *ValS = nullptr) const;
  static const ValueState *getValStateForRegion(ASTContext &AstC,
                                                const ProgramStateRef &State,
                                                const MemRegion *R,
                                                bool Debug = false);
  bool gcEnabledHere(CheckerContext &C) const;
  bool safepointEnabledHere(CheckerContext &C) const;
  bool propagateArgumentRootedness(CheckerContext &C,
                                   ProgramStateRef &State) const;
  SymbolRef getSymbolForResult(const Expr *Result, const ValueState *OldValS,
                               ProgramStateRef &State, CheckerContext &C) const;
  void validateValue(const GCChecker::ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message) const;
  void validateValueRootnessOnly(const GCChecker::ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message) const;
  void validateValue(const GCChecker::ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message, SourceRange range) const;
  void validateValueRootnessOnly(const GCChecker::ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message, SourceRange range) const;
  int validateValueInner(const GCChecker::ValueState* VS) const;
  GCChecker::ValueState getRootedFromRegion(const MemRegion *Region, const PinState *PS, int Depth) const;
  template <typename T>
  static void logWithDump(const std::string& message, const T &obj);
  static void log(const std::string& message);
  template <typename T>
  static void log(const std::string& message, const T &obj);

public:
  void checkBeginFunction(CheckerContext &Ctx) const;
  void checkEndFunction(const clang::ReturnStmt *RS, CheckerContext &Ctx) const;
  bool evalCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
  void checkPostStmt(const CStyleCastExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const ArraySubscriptExpr *CE, CheckerContext &C) const;
  void checkPostStmt(const MemberExpr *ME, CheckerContext &C) const;
  void checkPostStmt(const UnaryOperator *UO, CheckerContext &C) const;
  void checkDerivingExpr(const Expr *Result, const Expr *Parent,
                         bool ParentIsLoc, CheckerContext &C) const;
  void checkBind(SVal Loc, SVal Val, const Stmt *S, CheckerContext &) const;
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &) const;
  class GCBugVisitor : public BugReporterVisitor {
  public:
    GCBugVisitor() {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
    }

    PDP VisitNode(const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &BR) override;
  };

  class GCValueBugVisitor : public BugReporterVisitor {
  protected:
    SymbolRef Sym;

  public:
    GCValueBugVisitor(SymbolRef S) : Sym(S) {}

    void Profile(llvm::FoldingSetNodeID &ID) const override {
      static int X = 0;
      ID.AddPointer(&X);
      ID.AddPointer(Sym);
    }

    PDP ExplainNoPropagation(const ExplodedNode *N, PathDiagnosticLocation Pos,
                             BugReporterContext &BRC, PathSensitiveBugReport &BR);
    PDP ExplainNoPropagationFromExpr(const clang::Expr *FromWhere,
                                     const ExplodedNode *N,
                                     PathDiagnosticLocation Pos,
                                     BugReporterContext &BRC, PathSensitiveBugReport &BR);

    PDP VisitNode(const ExplodedNode *N, BugReporterContext &BRC, PathSensitiveBugReport &BR) override;
  }; // namespace
};

} // namespace

REGISTER_TRAIT_WITH_PROGRAMSTATE(GCDepth, unsigned)
REGISTER_TRAIT_WITH_PROGRAMSTATE(GCDisabledAt, unsigned)
REGISTER_TRAIT_WITH_PROGRAMSTATE(SafepointDisabledAt, unsigned)
REGISTER_TRAIT_WITH_PROGRAMSTATE(MayCallSafepoint, bool)
REGISTER_MAP_WITH_PROGRAMSTATE(GCValueMap, SymbolRef, GCChecker::ValueState)
REGISTER_MAP_WITH_PROGRAMSTATE(GCRootMap, const MemRegion *,
                               GCChecker::RootState)
REGISTER_MAP_WITH_PROGRAMSTATE(GCPinMap, const MemRegion *, GCChecker::PinState)

template <typename callback>
SymbolRef GCChecker::walkToRoot(callback f, const ProgramStateRef &State,
                                const MemRegion *Region) {
  if (!Region)
    return nullptr;
  logWithDump("- walkToRoot, Region", Region);
  while (true) {
    const SymbolicRegion *SR = Region->getSymbolicBase();
    if (!SR) {
      return nullptr;
    }
    SymbolRef Sym = SR->getSymbol();
    const ValueState *OldVState = State->get<GCValueMap>(Sym);
    logWithDump("- walkToRoot, Sym", Sym);
    logWithDump("- walkToRoot, OldVState", OldVState);
    if (f(Sym, OldVState)) {
      if (const SymbolRegionValue *SRV = dyn_cast<SymbolRegionValue>(Sym)) {
        Region = SRV->getRegion();
        continue;
      } else if (const SymbolDerived *SD = dyn_cast<SymbolDerived>(Sym)) {
        Region = SD->getRegion();
        continue;
      }
      return nullptr;
    }
    return Sym;
  }
}

namespace Helpers {
static const VarRegion *walk_back_to_global_VR(const MemRegion *Region) {
  if (!Region)
    return nullptr;
  while (true) {
    const VarRegion *VR = Region->getAs<VarRegion>();
    if (VR && VR->getDecl()->hasGlobalStorage()) {
      return VR;
    }
    const SymbolicRegion *SymR = Region->getAs<SymbolicRegion>();
    if (SymR) {
      const SymbolRegionValue *SymRV =
          dyn_cast<SymbolRegionValue>(SymR->getSymbol());
      if (!SymRV) {
        const SymbolDerived *SD = dyn_cast<SymbolDerived>(SymR->getSymbol());
        if (SD) {
          Region = SD->getRegion();
          continue;
        }
        break;
      }
      Region = SymRV->getRegion();
      continue;
    }
    const SubRegion *SR = Region->getAs<SubRegion>();
    if (!SR)
      break;
    Region = SR->getSuperRegion();
  }
  return nullptr;
}
} // namespace Helpers

#define VALID 0
#define FREED 1
#define MOVED 2

void GCChecker::validateValueRootnessOnly(const ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message, SourceRange range) const {
  int v = validateValueInner(VS);
  if (v == FREED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " GCed").c_str(), range);
  } else if (v == MOVED) {
    // We don't care if it is moved
  }
}

void GCChecker::validateValue(const ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message, SourceRange range) const {
  int v = validateValueInner(VS);
  if (v == FREED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " GCed").c_str(), range);
  } else if (v == MOVED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " moved").c_str(), range);
  }
}

void GCChecker::validateValueRootnessOnly(const ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message) const {
  int v = validateValueInner(VS);
  if (v == FREED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " GCed").c_str());
  } else if (v == MOVED) {
    // We don't care if it is moved
  }
}

void GCChecker::validateValue(const ValueState* VS, CheckerContext &C, SymbolRef Sym, const char *message) const {
  int v = validateValueInner(VS);
  if (v == FREED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " GCed").c_str());
  } else if (v == MOVED) {
    GCChecker::report_value_error(C, Sym, (std::string(message) + " moved").c_str());
  }
}

int GCChecker::validateValueInner(const ValueState* VS) const {
  if (!VS)
    return VALID;

  if (VS->isPotentiallyFreed()) {
    return FREED;
  }

  if (VS->isMoved()) {
    return MOVED;
  }

  return VALID;
}

GCChecker::ValueState GCChecker::getRootedFromRegion(const MemRegion *Region, const PinState *PS, int Depth) const {
  ValueState Ret = ValueState::getRooted(Region, Depth);

  if (PS) {
    if (PS->isTransitivePin()) {
      Ret = ValueState::getTransitivelyPinned(Ret);
    } else if (PS->isPin()) {
      Ret = ValueState::getPinned(Ret);
    } else if (PS->isNoPin()) {
      Ret = ValueState::getNotPinned(Ret);
    } else {
      printf("Invalid PinState\n");
      exit(1);
    }
  }

  return Ret;
}

template <typename T>
void dumpInner(const T *obj) {
  if (obj) {
    obj->dump();
  } else {
    llvm::errs() << "null";
  }
}
template <typename T>
void dumpInner(const T &obj) {
  obj.dump();
}

template <typename T>
void GCChecker::logWithDump(const std::string& message, const T &obj) {
  if (!DEBUG_LOG)
    return;

  llvm::errs() << message;
  llvm::errs() << ": ";
  dumpInner(obj);
  llvm::errs() << "\n";
}

void GCChecker::log(const std::string& message) {
  if (!DEBUG_LOG)
    return;

  llvm::errs() << message;
  llvm::errs() << "\n";
}

template <typename T>
void GCChecker::log(const std::string& message, const T &obj) {
  if (!DEBUG_LOG)
    return;

  llvm::errs() << message;
  llvm::errs() << ": ";
  llvm::errs() << obj;
  llvm::errs() << "\n";
}

PDP GCChecker::GCBugVisitor::VisitNode(const ExplodedNode *N,
                                       BugReporterContext &BRC, PathSensitiveBugReport &BR) {
  const ExplodedNode *PrevN = N->getFirstPred();
  unsigned NewGCDepth = N->getState()->get<GCDepth>();
  unsigned OldGCDepth = PrevN->getState()->get<GCDepth>();
  if (NewGCDepth != OldGCDepth) {
    PathDiagnosticLocation Pos(getStmtForDiagnostics(N),
                               BRC.getSourceManager(), N->getLocationContext());
    return MakePDP(Pos, "GC frame changed here.");
  }
  unsigned NewGCState = N->getState()->get<GCDisabledAt>();
  unsigned OldGCState = PrevN->getState()->get<GCDisabledAt>();
  if (false /*NewGCState != OldGCState*/) {
    PathDiagnosticLocation Pos(getStmtForDiagnostics(N),
                               BRC.getSourceManager(), N->getLocationContext());
    return MakePDP(Pos, "GC enabledness changed here.");
  }
  return nullptr;
}

PDP GCChecker::GCValueBugVisitor::ExplainNoPropagationFromExpr(
    const clang::Expr *FromWhere, const ExplodedNode *N,
    PathDiagnosticLocation Pos, BugReporterContext &BRC, PathSensitiveBugReport &BR) {
  const MemRegion *Region =
      N->getState()->getSVal(FromWhere, N->getLocationContext()).getAsRegion();
  SymbolRef Parent = walkToRoot(
      [&](SymbolRef Sym, const ValueState *OldVState) { return !OldVState; },
      N->getState(), Region);
  if (!Parent && Region) {
    Parent = walkToRoot(
        [&](SymbolRef Sym, const ValueState *OldVState) { return !OldVState; },
        N->getState(), N->getState()->getSVal(Region).getAsRegion());
  }
  if (!Parent) {
    // May have been derived from a global. Check that
    const VarRegion *VR = Helpers::walk_back_to_global_VR(Region);
    if (VR) {
      BR.addNote("Derivation root was here",
                 PathDiagnosticLocation::create(VR->getDecl(),
                                                BRC.getSourceManager()));
      const VarDecl *VD = VR->getDecl();
      if (VD) {
        if (!declHasAnnotation(VD, "julia_globally_rooted")) {
          return MakePDP(Pos, "Argument value was derived from unrooted "
                              "global. May need GLOBALLY_ROOTED annotation.");
        } else if (!isGCTrackedType(VD->getType())) {
          return MakePDP(
              Pos, "Argument value was derived global with untracked type. You "
                   "may want to update the checker's type list");
        }
      }
      return MakePDP(Pos,
                     "Argument value was derived from global, but the checker "
                     "did not propagate the root. This may be a bug");
    }
    return MakePDP(Pos,
                   "Could not propagate root. Argument value was untracked.");
  }
  const ValueState *ValS = N->getState()->get<GCValueMap>(Parent);
  assert(ValS);
  if (ValS->isPotentiallyFreed()) {
    BR.addVisitor(make_unique<GCValueBugVisitor>(Parent));
    return MakePDP(
        Pos, "Root not propagated because it may have been freed. Tracking.");
  } else if (ValS->isRooted()) {
    BR.addVisitor(make_unique<GCValueBugVisitor>(Parent));
    return MakePDP(
        Pos, "Root was not propagated due to a bug. Tracking base value.");
  } else {
    BR.addVisitor(make_unique<GCValueBugVisitor>(Parent));
    return MakePDP(Pos, "No Root to propagate. Tracking.");
  }
}

PDP GCChecker::GCValueBugVisitor::ExplainNoPropagation(
    const ExplodedNode *N, PathDiagnosticLocation Pos, BugReporterContext &BRC,
    PathSensitiveBugReport &BR) {
  if (N->getLocation().getAs<StmtPoint>()) {
    const clang::Stmt *TheS = N->getLocation().castAs<StmtPoint>().getStmt();
    const clang::CallExpr *CE = dyn_cast<CallExpr>(TheS);
    const clang::MemberExpr *ME = dyn_cast<MemberExpr>(TheS);
    if (ME)
      return ExplainNoPropagationFromExpr(ME->getBase(), N, Pos, BRC, BR);
    const clang::ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(TheS);
    if (ASE)
      return ExplainNoPropagationFromExpr(ASE->getLHS(), N, Pos, BRC, BR);
    if (!CE)
      return nullptr;
    const clang::FunctionDecl *FD = CE->getDirectCallee();
    if (!FD)
      return nullptr;
    for (unsigned i = 0; i < FD->getNumParams(); ++i) {
      if (!declHasAnnotation(FD->getParamDecl(i), "julia_propagates_root"))
        continue;
      return ExplainNoPropagationFromExpr(CE->getArg(i), N, Pos, BRC, BR);
    }
    return nullptr;
  }
  return nullptr;
}

PDP GCChecker::GCValueBugVisitor::VisitNode(const ExplodedNode *N,
                                            BugReporterContext &BRC, PathSensitiveBugReport &BR) {
  const ExplodedNode *PrevN = N->getFirstPred();
  const ValueState *NewValueState = N->getState()->get<GCValueMap>(Sym);
  const ValueState *OldValueState = PrevN->getState()->get<GCValueMap>(Sym);
  const Stmt *Stmt = getStmtForDiagnostics(N);

  PathDiagnosticLocation Pos;
  if (Stmt)
    Pos = PathDiagnosticLocation{Stmt, BRC.getSourceManager(),
                                 N->getLocationContext()};
  else
    Pos = PathDiagnosticLocation::createDeclEnd(N->getLocationContext(),
                                                BRC.getSourceManager());
  if (!NewValueState)
    return nullptr;
  if (!OldValueState) {
    if (NewValueState->isRooted()) {
      return MakePDP(Pos, "Started tracking value here (root was inherited).");
    } else {
      if (NewValueState->FD) {
        bool isFunctionSafepoint =
            !isFDAnnotatedNotSafepoint(NewValueState->FD);
        bool maybeUnrooted =
            declHasAnnotation(NewValueState->PVD, "julia_maybe_unrooted");
        assert(isFunctionSafepoint || maybeUnrooted);
        (void)maybeUnrooted;
        Pos =
            PathDiagnosticLocation{NewValueState->PVD, BRC.getSourceManager()};
        if (!isFunctionSafepoint)
          return MakePDP(Pos, "Argument not rooted, because function was "
                              "annotated as not a safepoint");
        else
          return MakePDP(Pos, "Argument was annotated as MAYBE_UNROOTED.");
      } else {
        PDP Diag = ExplainNoPropagation(N, Pos, BRC, BR);
        if (Diag)
          return Diag;
        return MakePDP(Pos, "Started tracking value here.");
      }
    }
  }
  if (!OldValueState->isUntracked() && NewValueState->isUntracked()) {
    PDP Diag = ExplainNoPropagation(N, Pos, BRC, BR);
    if (Diag)
      return Diag;
    return MakePDP(Pos, "Created untracked derivative.");
  } else if (NewValueState->isPotentiallyFreed() &&
             OldValueState->isJustAllocated()) {
    // std::make_shared< in later LLVM
    return MakePDP(Pos, "Value may have been GCed here.");
  } else if (NewValueState->isPotentiallyFreed() &&
             !OldValueState->isPotentiallyFreed()) {
    // std::make_shared< in later LLVM
    return MakePDP(Pos,
                   "Value may have been GCed here (though I don't know why).");
  } else if (NewValueState->isRooted() && OldValueState->isJustAllocated()) {
    return MakePDP(Pos, "Value was rooted here.");
  } else if (!NewValueState->isRooted() && OldValueState->isRooted()) {
    return MakePDP(Pos, "Root was released here.");
  } else if (NewValueState->RootDepth != OldValueState->RootDepth) {
    return MakePDP(Pos, "Rooting Depth changed here.");
  } else if (NewValueState->isMoved() && !OldValueState->isMoved()) {
    return MakePDP(Pos, "Value was moved here.");
  }
  return nullptr;
}

template <typename callback>
void GCChecker::report_error(callback f, CheckerContext &C,
                             StringRef message) const {
  // Generate an error node.
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;

  if (!BT)
    BT.reset(new BugType(this, "Invalid GC thingy", categories::LogicError));
  auto Report = make_unique<PathSensitiveBugReport>(*BT, message, N);
  Report->addVisitor(make_unique<GCBugVisitor>());
  f(Report.get());
  C.emitReport(std::move(Report));
}

void GCChecker::report_value_error(CheckerContext &C, SymbolRef Sym,
                                   const char *message,
                                   SourceRange range) const {
  // Generate an error node.
  ExplodedNode *N = C.generateErrorNode();
  if (!N)
    return;

  if (!BT)
    BT.reset(new BugType(this, "Invalid GC thingy", categories::LogicError));
  auto Report = make_unique<PathSensitiveBugReport>(*BT, message, N);
  Report->addVisitor(make_unique<GCValueBugVisitor>(Sym));
  Report->addVisitor(make_unique<GCBugVisitor>());
  Report->addVisitor(make_unique<ConditionBRVisitor>());
  if (!range.isInvalid()) {
    Report->addRange(range);
  }
  C.emitReport(std::move(Report));
}

bool GCChecker::gcEnabledHere(CheckerContext &C) const {
  unsigned disabledAt = C.getState()->get<GCDisabledAt>();
  return disabledAt == (unsigned)-1;
}

bool GCChecker::safepointEnabledHere(CheckerContext &C) const {
  unsigned disabledAt = C.getState()->get<SafepointDisabledAt>();
  return disabledAt == (unsigned)-1;
}

bool GCChecker::propagateArgumentRootedness(CheckerContext &C,
                                            ProgramStateRef &State) const {
  const auto *LCtx = C.getLocationContext();

  const auto *Site = cast<StackFrameContext>(LCtx)->getCallSite();
  if (!Site)
    return false;

  const auto *FD = dyn_cast<FunctionDecl>(LCtx->getDecl());
  if (!FD)
    return false;

  const auto *CE = dyn_cast<CallExpr>(Site);
  if (!CE)
    return false;

  // FD->dump();

  bool Change = false;
  int idx = 0;
  for (const auto P : FD->parameters()) {
    if (!isGCTrackedType(P->getType())) {
      continue;
    }
    auto Arg = State->getSVal(CE->getArg(idx++), LCtx->getParent());
    SymbolRef ArgSym = walkToRoot(
        [](SymbolRef Sym, const ValueState *OldVState) { return !OldVState; },
        State, Arg.getAsRegion());
    if (!ArgSym) {
      continue;
    }
    const ValueState *ValS = State->get<GCValueMap>(ArgSym);
    if (!ValS) {
      report_error(
          [&](PathSensitiveBugReport *Report) {
            Report->addNote(
                "Tried to find root for this parameter in inlined call",
                PathDiagnosticLocation::create(P, C.getSourceManager()));
          },
          C, "Missed allocation of parameter");
      continue;
    }
    auto Param = State->getLValue(P, LCtx);
    SymbolRef ParamSym = State->getSVal(Param).getAsSymbol();
    if (!ParamSym) {
      continue;
    }
    if (isGloballyRootedType(P->getType())) {
      State =
          State->set<GCValueMap>(ParamSym, ValueState::getRooted(nullptr, ValueState::pinState(isGloballyTransitivelyPinnedType(P->getType())), -1));
      Change = true;
      continue;
    }
    State = State->set<GCValueMap>(ParamSym, *ValS);
    Change = true;
  }
  return Change;
}

void GCChecker::checkBeginFunction(CheckerContext &C) const {
  // Consider top-level argument values rooted, unless an annotation says
  // otherwise
  const auto *LCtx = C.getLocationContext();
  const auto *FD = dyn_cast<FunctionDecl>(LCtx->getDecl());
  if (!FD)
    return;
  logWithDump("checkBeginFunction", FD);
  ProgramStateRef State = C.getState();
  bool Change = false;
  if (C.inTopFrame()) {
    State = State->set<GCDisabledAt>((unsigned)-1);
    State = State->set<SafepointDisabledAt>((unsigned)-1);
    Change = true;
  }
  if (State->get<GCDisabledAt>() == (unsigned)-1) {
    if (declHasAnnotation(FD, "julia_gc_disabled")) {
      State = State->set<GCDisabledAt>(C.getStackFrame()->getIndex());
      Change = true;
    }
  }
  if (State->get<SafepointDisabledAt>() == (unsigned)-1 &&
      isFDAnnotatedNotSafepoint(FD)) {
    State = State->set<SafepointDisabledAt>(C.getStackFrame()->getIndex());
    Change = true;
  }
  if (!C.inTopFrame()) {
    if (propagateArgumentRootedness(C, State) || Change)
      C.addTransition(State);
    return;
  }
  for (const auto P : FD->parameters()) {
    if (declHasAnnotation(P, "julia_require_rooted_slot")) {
      auto Param = State->getLValue(P, LCtx);
      const MemRegion *Root = State->getSVal(Param).getAsRegion();
      State = State->set<GCRootMap>(Root, RootState::getRoot(-1));
      if (declHasAnnotation(P, "julia_require_tpin"))
        State = State->set<GCPinMap>(Root, PinState::getTransitivePin(-1));
      else
        State = State->set<GCPinMap>(Root, PinState::getTransitivePin(-1));
    } else if (isGCTrackedType(P->getType())) {
      auto Param = State->getLValue(P, LCtx);
      SymbolRef AssignedSym = State->getSVal(Param).getAsSymbol();
      if (!AssignedSym)
        continue;
      assert(AssignedSym);
      State = State->set<GCValueMap>(AssignedSym,
                                     ValueState::getForArgument(FD, P));
      Change = true;
    }
  }
  if (Change) {
    C.addTransition(State);
  }
}

void GCChecker::checkEndFunction(const clang::ReturnStmt *RS,
                                 CheckerContext &C) const {
  log("checkEndFunction");
  ProgramStateRef State = C.getState();

  if (RS && gcEnabledHere(C) && RS->getRetValue() && isGCTracked(RS->getRetValue())) {
    auto ResultVal = C.getSVal(RS->getRetValue());
    SymbolRef Sym = ResultVal.getAsSymbol(true);
    const ValueState *ValS = Sym ? State->get<GCValueMap>(Sym) : nullptr;
    validateValue(ValS, C, Sym, "Return value may have been", RS->getSourceRange());
  }

  bool Changed = false;
  if (State->get<GCDisabledAt>() == C.getStackFrame()->getIndex()) {
    State = State->set<GCDisabledAt>((unsigned)-1);
    Changed = true;
  }
  if (State->get<SafepointDisabledAt>() == C.getStackFrame()->getIndex()) {
    State = State->set<SafepointDisabledAt>((unsigned)-1);
    Changed = true;
  }
  if (Changed)
    C.addTransition(State);
  if (!C.inTopFrame())
    return;
  if (C.getState()->get<GCDepth>() > 0)
    report_error(C, "Non-popped GC frame present at end of function");
}

bool GCChecker::declHasAnnotation(const clang::Decl *D, const char *which) {
  for (const auto *Ann : D->specific_attrs<AnnotateAttr>()) {
    if (Ann->getAnnotation() == which)
      return true;
  }
  return false;
}

bool GCChecker::isFDAnnotatedNotSafepoint(const clang::FunctionDecl *FD) {
  return declHasAnnotation(FD, "julia_not_safepoint");
}

#if LLVM_VERSION_MAJOR >= 13
#define endswith_lower endswith_insensitive
#endif

bool GCChecker::isGCTrackedType(QualType QT) {
  return isJuliaType(
             [](StringRef Name) {
               if (Name.endswith_lower("jl_value_t") ||
                   Name.endswith_lower("jl_svec_t") ||
                   Name.endswith_lower("jl_sym_t") ||
                   Name.endswith_lower("jl_expr_t") ||
                   Name.endswith_lower("jl_code_info_t") ||
                   Name.endswith_lower("jl_array_t") ||
                   Name.endswith_lower("jl_method_t") ||
                   Name.endswith_lower("jl_method_instance_t") ||
                   Name.endswith_lower("jl_tupletype_t") ||
                   Name.endswith_lower("jl_datatype_t") ||
                   Name.endswith_lower("jl_typemap_entry_t") ||
                   Name.endswith_lower("jl_typemap_level_t") ||
                   Name.endswith_lower("jl_typename_t") ||
                   Name.endswith_lower("jl_module_t") ||
                   Name.endswith_lower("jl_tupletype_t") ||
                   Name.endswith_lower("jl_gc_tracked_buffer_t") ||
                   Name.endswith_lower("jl_binding_t") ||
                   Name.endswith_lower("jl_ordereddict_t") ||
                   Name.endswith_lower("jl_tvar_t") ||
                   Name.endswith_lower("jl_typemap_t") ||
                   Name.endswith_lower("jl_unionall_t") ||
                   Name.endswith_lower("jl_methtable_t") ||
                   Name.endswith_lower("jl_cgval_t") ||
                   Name.endswith_lower("jl_codectx_t") ||
                   Name.endswith_lower("jl_ast_context_t") ||
                   Name.endswith_lower("jl_code_instance_t") ||
                   Name.endswith_lower("jl_excstack_t") ||
                   Name.endswith_lower("jl_task_t") ||
                   Name.endswith_lower("jl_uniontype_t") ||
                   Name.endswith_lower("jl_method_match_t") ||
                   Name.endswith_lower("jl_vararg_t") ||
                   Name.endswith_lower("jl_opaque_closure_t") ||
                   Name.endswith_lower("jl_globalref_t") ||
                   // Probably not technically true for these, but let's allow it
                   Name.endswith_lower("typemap_intersection_env") ||
                   Name.endswith_lower("interpreter_state") ||
                   Name.endswith_lower("jl_typeenv_t") ||
                   Name.endswith_lower("jl_stenv_t") ||
                   Name.endswith_lower("jl_varbinding_t") ||
                   Name.endswith_lower("set_world") ||
                   Name.endswith_lower("jl_codectx_t")) {
                 return true;
               }
               return false;
             },
             QT);
}

bool GCChecker::isGCTracked(const Expr *E) {
  while (1) {
    if (isGCTrackedType(E->getType()))
      return true;
    if (auto ICE = dyn_cast<ImplicitCastExpr>(E))
      E = ICE->getSubExpr();
    else if (auto CE = dyn_cast<CastExpr>(E))
      E = CE->getSubExpr();
    else
      return false;
  }
}

bool GCChecker::isGloballyRootedType(QualType QT) const {
  return isJuliaType(
      [](StringRef Name) { return Name.endswith("jl_sym_t"); }, QT);
}

bool GCChecker::isGloballyTransitivelyPinnedType(QualType QT) const {
  return false;
}

bool GCChecker::isNonMovingType(QualType QT) const {
  return false;
}

bool GCChecker::isSafepoint(const CallEvent &Call) const {
  bool isCalleeSafepoint = true;
  if (Call.isInSystemHeader()) {
    // defined by -isystem per
    // https://clang.llvm.org/docs/UsersManual.html#controlling-diagnostics-in-system-headers
    isCalleeSafepoint = false;
  } else {
    auto *Decl = Call.getDecl();
    const DeclContext *DC = Decl ? Decl->getDeclContext() : nullptr;
    while (DC) {
      // Anything in llvm or std is not a safepoint
      if (const NamespaceDecl *NDC = dyn_cast<NamespaceDecl>(DC))
        if (NDC->getName() == "llvm" || NDC->getName() == "std")
          return false;
      DC = DC->getParent();
    }
    const FunctionDecl *FD = Decl ? Decl->getAsFunction() : nullptr;
    if (!Decl || !FD) {
      const clang::Expr *Callee =
          dyn_cast<CallExpr>(Call.getOriginExpr())->getCallee();
      if (const TypedefType *TDT = dyn_cast<TypedefType>(Callee->getType())) {
        isCalleeSafepoint =
            !declHasAnnotation(TDT->getDecl(), "julia_not_safepoint");
      } else if (const CXXPseudoDestructorExpr *PDE =
                     dyn_cast<CXXPseudoDestructorExpr>(Callee)) {
        // A pseudo-destructor is an expression that looks like a member
        // access to a destructor of a scalar type. A pseudo-destructor
        // expression has no run-time semantics beyond evaluating the base
        // expression (which would have it's own CallEvent, if applicable).
        isCalleeSafepoint = false;
      }
    } else if (FD) {
      if (FD->getBuiltinID() != 0 || FD->isTrivial())
        isCalleeSafepoint = false;
      else if (FD->getDeclName().isIdentifier() &&
               (FD->getName().startswith("uv_") ||
                FD->getName().startswith("unw_") ||
                FD->getName().startswith("_U")) &&
               FD->getName() != "uv_run")
        isCalleeSafepoint = false;
      else
        isCalleeSafepoint = !isFDAnnotatedNotSafepoint(FD);
    }
  }
  return isCalleeSafepoint;
}

bool GCChecker::processPotentialSafepoint(const CallEvent &Call,
                                          CheckerContext &C,
                                          ProgramStateRef &State) const {
  if (!isSafepoint(Call))
    return false;
  bool DidChange = false;
  if (!gcEnabledHere(C))
    return false;
  const Decl *D = Call.getDecl();
  const FunctionDecl *FD = D ? D->getAsFunction() : nullptr;
  SymbolRef SpeciallyRootedSymbol = nullptr;
  if (FD) {
    for (unsigned i = 0; i < FD->getNumParams(); ++i) {
      QualType ParmType = FD->getParamDecl(i)->getType();
      if (declHasAnnotation(FD->getParamDecl(i), "julia_temporarily_roots")) {
        if (ParmType->isPointerType() &&
            ParmType->getPointeeType()->isPointerType() &&
            isGCTrackedType(ParmType->getPointeeType())) {
          // This is probably an out parameter. Find the value it refers to now.
          SVal Loaded =
              State->getSVal(Call.getArgSVal(i).getAs<Loc>().getValue());
          SpeciallyRootedSymbol = Loaded.getAsSymbol();
          continue;
        }
        SVal Test = Call.getArgSVal(i);
        // Walk backwards to find the symbol that we're tracking for this
        // value
        const MemRegion *Region = Test.getAsRegion();
        SpeciallyRootedSymbol =
            walkToRoot([&](SymbolRef Sym,
                           const ValueState *OldVState) { return !OldVState; },
                       State, Region);
        break;
      }
    }
  }

  // Don't free the return value
  SymbolRef RetSym = Call.getReturnValue().getAsSymbol();

  // Symbolically free all unrooted values.
  GCValueMapTy AMap = State->get<GCValueMap>();
  for (auto I = AMap.begin(), E = AMap.end(); I != E; ++I) {
    if (I.getData().isJustAllocated()) {
      if (SpeciallyRootedSymbol == I.getKey())
        continue;
      if (RetSym == I.getKey())
        continue;
      State = State->set<GCValueMap>(I.getKey(), ValueState::getFreed());
      DidChange = true;
    }
  }
  // Symbolically move all unpinned values.
  GCValueMapTy AMap2 = State->get<GCValueMap>();
  for (auto I = AMap2.begin(), E = AMap2.end(); I != E; ++I) {
    logWithDump("- check Sym", I.getKey());
    if (RetSym == I.getKey())
      continue;
    if (I.getData().isNotPinned()) {
      logWithDump("- move unpinned values, VS", I.getData());
      auto NewVS = ValueState::getMoved(I.getData());
      State = State->set<GCValueMap>(I.getKey(), NewVS);
      logWithDump("- move unpinned values, NewVS", NewVS);
      DidChange = true;
    }
  }
  return DidChange;
}

const GCChecker::ValueState *
GCChecker::getValStateForRegion(ASTContext &AstC, const ProgramStateRef &State,
                                const MemRegion *Region, bool Debug) {
  if (!Region)
    return nullptr;
  SymbolRef Sym = walkToRoot(
      [&](SymbolRef Sym, const ValueState *OldVState) {
        return !OldVState || !OldVState->isRooted();
      },
      State, Region);
  if (!Sym)
    return nullptr;
  return State->get<GCValueMap>(Sym);
}

bool GCChecker::processArgumentRooting(const CallEvent &Call, CheckerContext &C,
                                       ProgramStateRef &State) const {
  auto *Decl = Call.getDecl();
  const FunctionDecl *FD = Decl ? Decl->getAsFunction() : nullptr;
  if (!FD)
    return false;
  const MemRegion *RootingRegion = nullptr;
  SymbolRef RootedSymbol = nullptr;
  for (unsigned i = 0; i < FD->getNumParams(); ++i) {
    if (declHasAnnotation(FD->getParamDecl(i), "julia_rooting_argument")) {
      logWithDump("- Rooting arg", Call.getArgSVal(i));
      RootingRegion = Call.getArgSVal(i).getAsRegion();
    } else if (declHasAnnotation(FD->getParamDecl(i),
                                 "julia_rooted_argument")) {
      logWithDump("- Rooted arg", Call.getArgSVal(i));
      RootedSymbol = Call.getArgSVal(i).getAsSymbol();
    }
  }
  if (!RootingRegion || !RootedSymbol)
    return false;
  const ValueState *OldVState =
      getValStateForRegion(C.getASTContext(), State, RootingRegion);
  if (!OldVState)
    return false;
  const ValueState *CurrentVState = State->get<GCValueMap>(RootedSymbol);
  ValueState NewVState = *OldVState;
  // If the old state is pinned, the new state is not pinned.
  // if (OldVState->isPinned() && ((CurrentVState && !CurrentVState->isPinnedByAnyway()) || !CurrentVState)) {
  //   NewVState = ValueState::getNotPinned(*OldVState);
  // }
  logWithDump("- Rooted set to", NewVState);
  State = State->set<GCValueMap>(RootedSymbol, NewVState);
  return true;
}

bool GCChecker::processAllocationOfResult(const CallEvent &Call,
                                          CheckerContext &C,
                                          ProgramStateRef &State) const {
  QualType QT = Call.getResultType();
  if (!isGCTrackedType(QT))
    return false;
  if (!Call.getOriginExpr()) {
    return false;
  }
  SymbolRef Sym = Call.getReturnValue().getAsSymbol();
  if (!Sym) {
    SVal S = C.getSValBuilder().conjureSymbolVal(
        Call.getOriginExpr(), C.getLocationContext(), QT, C.blockCount());
    State = State->BindExpr(Call.getOriginExpr(), C.getLocationContext(), S);
    Sym = S.getAsSymbol();
    logWithDump("- conjureSymbolVal, S", S);
    logWithDump("- conjureSymbolVal, Sym", Sym);
  }
  if (isGloballyRootedType(QT))
    State = State->set<GCValueMap>(Sym, ValueState::getRooted(nullptr, ValueState::pinState(isGloballyTransitivelyPinnedType(QT)), -1));
  else {
    const ValueState *ValS = State->get<GCValueMap>(Sym);
    ValueState NewVState = ValS ? *ValS : ValueState::getAllocated();
    auto *Decl = Call.getDecl();
    const FunctionDecl *FD = Decl ? Decl->getAsFunction() : nullptr;
    if (FD) {
      if (declHasAnnotation(FD, "julia_globally_rooted")) {
        if (declHasAnnotation(FD, "julia_globally_pinned")) {
          NewVState = ValueState::getRooted(nullptr, ValueState::Pinned, -1);
        } else if (declHasAnnotation(FD, "julia_globally_tpinned")) {
          NewVState = ValueState::getRooted(nullptr, ValueState::TransitivelyPinned, -1);
        } else {
          // Not pinned
          NewVState = ValueState::getRooted(nullptr, -1);
        }
      } else {
        // Special case for jl_box_ functions which have value-dependent
        // global roots.
        // See jl_as_global_root().
        StringRef FDName =
            FD->getDeclName().isIdentifier() ? FD->getName() : "";
        if (FDName.startswith("jl_box_") || FDName.startswith("ijl_box_")) {
          SVal Arg = Call.getArgSVal(0);
          if (auto CI = Arg.getAs<nonloc::ConcreteInt>()) {
            const llvm::APSInt &Value = CI->getValue();
            bool GloballyRooted = false;
            const int64_t NBOX_C = 1024;
            if (FDName.startswith("jl_box_u") || FDName.startswith("ijl_box_u")) {
              if (Value < NBOX_C) {
                GloballyRooted = true;
              }
            } else {
              if (-NBOX_C / 2 < Value && Value < (NBOX_C - NBOX_C / 2)) {
                GloballyRooted = true;
              }
            }
            if (GloballyRooted) {
              // These are perm allocated, thus pinned.
              NewVState = ValueState::getRooted(nullptr, ValueState::Pinned, -1);
            }
          }
        } else {
          for (unsigned i = 0; i < FD->getNumParams(); ++i) {
            if (declHasAnnotation(FD->getParamDecl(i),
                                  "julia_propagates_root")) {
              SVal Test = Call.getArgSVal(i);
              // Walk backwards to find the region that roots this value
              const MemRegion *Region = Test.getAsRegion();
              const ValueState *OldVState =
                  getValStateForRegion(C.getASTContext(), State, Region);
              if (OldVState) {
                NewVState = ValueState::inheritState(*OldVState);
                logWithDump("- jl_propagates_root, OldVState", *OldVState);
                logWithDump("- jl_propagates_root, NewVState", NewVState);
              }
              break;
            }
          }
        }
      }
    }
    State = State->set<GCValueMap>(Sym, NewVState);
  }
  return true;
}

void GCChecker::checkPostCall(const CallEvent &Call, CheckerContext &C) const {
  logWithDump("checkPostCall", Call);
  ProgramStateRef State = C.getState();
  log("- processArgmentRooting");
  bool didChange = processArgumentRooting(Call, C, State);
  log("- processPotentialsafepoint");
  didChange |= processPotentialSafepoint(Call, C, State);
  log("- processAllocationOfResult");
  didChange |= processAllocationOfResult(Call, C, State);
  if (didChange)
    C.addTransition(State);
}

// Implicitly root values that were casted to globally rooted values
void GCChecker::checkPostStmt(const CStyleCastExpr *CE,
                              CheckerContext &C) const {
  logWithDump("checkpostStmt(CStyleCastExpr)", CE);
  if (!isGloballyRootedType(CE->getTypeAsWritten()))
    return;
  SymbolRef Sym = C.getSVal(CE).getAsSymbol();
  if (!Sym)
    return;
  C.addTransition(
      C.getState()->set<GCValueMap>(Sym, ValueState::getRooted(nullptr, ValueState::pinState(isGloballyTransitivelyPinnedType(CE->getTypeAsWritten())), -1)));
}

SymbolRef GCChecker::getSymbolForResult(const Expr *Result,
                                        const ValueState *OldValS,
                                        ProgramStateRef &State,
                                        CheckerContext &C) const {
  QualType QT = Result->getType();
  if (!QT->isPointerType() || QT->getPointeeType()->isVoidType())
    return nullptr;
  auto ValLoc = State->getSVal(Result, C.getLocationContext()).getAs<Loc>();
  if (!ValLoc) {
    return nullptr;
  }
  SVal Loaded = State->getSVal(*ValLoc);
  if (Loaded.isUnknown() || !Loaded.getAsSymbol()) {
    if (OldValS || GCChecker::isGCTracked(Result)) {
      Loaded = C.getSValBuilder().conjureSymbolVal(
          nullptr, Result, C.getLocationContext(), Result->getType(),
          C.blockCount());
      State = State->bindLoc(*ValLoc, Loaded, C.getLocationContext());
      // State = State->BindExpr(Result, C.getLocationContext(),
      // State->getSVal(*ValLoc));
    }
  }
  return Loaded.getAsSymbol();
}

void GCChecker::checkDerivingExpr(const Expr *Result, const Expr *Parent,
                                  bool ParentIsLoc, CheckerContext &C) const {
  log("checkDerivingExpr");
  if (auto PE = dyn_cast<ParenExpr>(Parent)) {
    Parent = PE->getSubExpr();
  }
  if (auto UO = dyn_cast<UnaryOperator>(Parent)) {
    if (UO->getOpcode() == UO_AddrOf) {
      Parent = UO->getSubExpr();
    }
  }
  bool ResultTracked = true;
  ProgramStateRef State = C.getState();
  if (isGloballyRootedType(Result->getType())) {
    log("- Globally rooted");
    SymbolRef NewSym = getSymbolForResult(Result, nullptr, State, C);
    if (!NewSym) {
      return;
    }
    const ValueState *NewValS = State->get<GCValueMap>(NewSym);
    if (NewValS && NewValS->isRooted() && NewValS->RootDepth == -1) {
      logWithDump("- NewValS is already rooted, skip", NewValS);
      return;
    }
    ValueState VS = ValueState::getRooted(nullptr, ValueState::pinState(isGloballyTransitivelyPinnedType(Result->getType())), -1);
    logWithDump("- Set VS", VS);
    C.addTransition(
        State->set<GCValueMap>(NewSym, VS));
    return;
  }
  if (!isGCTracked(Result)) {
    log("- Not GC tracked");
    // TODO: We may want to refine this. This is to track pointers through the
    // array list in jl_module_t.
    bool ParentIsModule = isJuliaType(
        [](StringRef Name) { return Name.endswith("jl_module_t"); },
        Parent->getType());
    bool ResultIsArrayList = isJuliaType(
        [](StringRef Name) { return Name.endswith("arraylist_t"); },
        Result->getType());
    if (!(ParentIsModule && ResultIsArrayList) && isGCTracked(Parent)) {
      ResultTracked = false;
    }
  }
  // This is the pointer
  auto ResultVal = C.getSVal(Result);
  if (ResultVal.isUnknown()) {
    if (!Result->getType()->isPointerType()) {
      log("- Result is not pointer type");
      return;
    }
    ResultVal = C.getSValBuilder().conjureSymbolVal(
        Result, C.getLocationContext(), Result->getType(),
        C.blockCount());
    State = State->BindExpr(Result, C.getLocationContext(), ResultVal);
  }
  auto ValLoc = ResultVal.getAs<Loc>();
  if (!ValLoc) {
    log("- Result is not a Loc");
    return;
  }
  SVal ParentVal = C.getSVal(Parent);
  SymbolRef OldSym = ParentVal.getAsSymbol(true);
  const MemRegion *Region = C.getSVal(Parent).getAsRegion();
  const ValueState *OldValS = OldSym ? State->get<GCValueMap>(OldSym) : nullptr;
  logWithDump("- Region", Region);
  logWithDump("- OldSym", OldSym);
  logWithDump("- OldValS", OldValS);
  SymbolRef NewSym = getSymbolForResult(Result, OldValS, State, C);
  if (!NewSym) {
    return;
  }
  logWithDump("- NewSym", NewSym);
  // NewSym might already have a better root
  const ValueState *NewValS = State->get<GCValueMap>(NewSym);
  logWithDump("- NewValS", NewValS);
  if (Region) {
    const VarRegion *VR = Region->getAs<VarRegion>();
    bool inheritedState = false;
    ValueState Updated = getRootedFromRegion(Region, State->get<GCPinMap>(Region), -1);
    logWithDump("- getRootedFromRegion", Region);
    logWithDump("- Region VS", Updated);
    if (VR && isa<ParmVarDecl>(VR->getDecl())) {
      log("- ParamVarDecl!!!!!!!!!!!!!!!!");
      // This works around us not being able to track symbols for struct/union
      // parameters very well.
      const auto *FD =
          dyn_cast<FunctionDecl>(C.getLocationContext()->getDecl());
      if (FD) {
        inheritedState = true;
        Updated =
            ValueState::getForArgument(FD, cast<ParmVarDecl>(VR->getDecl()));
      }
    } else {
      VR = Helpers::walk_back_to_global_VR(Region);
      if (VR) {
        logWithDump("- Walk back to", VR);
        if (VR && rootRegionIfGlobal(VR, State, C, &Updated)) {
          inheritedState = true;
        }
      }
    }
    if (inheritedState && ResultTracked) {
      logWithDump("- inheritedState, Sym", NewSym);
      logWithDump("- inheritedState, VS", Updated);
      C.addTransition(State->set<GCValueMap>(NewSym, Updated));
      return;
    }
  }
  if (NewValS && NewValS->isRooted()) {
    logWithDump("- NewValS is rooted", NewValS);
    return;
  }
  if (!OldValS) {
    // This way we'll get better diagnostics
    if (isGCTracked(Result)) {
      logWithDump("- We don't have OldValS, and the result is GC tracked. Set untracked", ValueState::getUntracked());
      C.addTransition(
          State->set<GCValueMap>(NewSym, ValueState::getUntracked()));
    }
    return;
  }
  validateValue(OldValS, C, OldSym, "Creating derivative of value that may have been");
  if (!OldValS->isPotentiallyFreed() && ResultTracked) {
    logWithDump("- Set as OldValS, Sym", NewSym);
    auto InheritVS = ValueState::inheritState(*OldValS);
    logWithDump("- Set as OldValS, InheritVS", InheritVS);
    C.addTransition(State->set<GCValueMap>(NewSym, InheritVS));
    return;
  }
}

// Propagate rootedness through subscript
void GCChecker::checkPostStmt(const ArraySubscriptExpr *ASE,
                              CheckerContext &C) const {
  logWithDump("checkPostStmt(ArraySubscriptExpr)", ASE);
  // Could be a root array, in which case this should be considered rooted
  // by that array.
  const MemRegion *Region = C.getSVal(ASE->getLHS()).getAsRegion();
  ProgramStateRef State = C.getState();
  logWithDump("- Region", Region);
  log("- isGCTracked", isGCTracked(ASE));
  if (Region && Region->getAs<ElementRegion>() && isGCTracked(ASE)) {
    auto SuperRegion = Region->getAs<ElementRegion>()->getSuperRegion();
    const RootState *RS = State->get<GCRootMap>(SuperRegion);
    const PinState *PS = State->get<GCPinMap>(SuperRegion);
    logWithDump("- RootState", RS);
    logWithDump("- PinState", PS);
    if (RS) {
      ValueState ValS = getRootedFromRegion(Region, PS, State->get<GCDepth>());
      SymbolRef NewSym = getSymbolForResult(ASE, &ValS, State, C);
      if (!NewSym) {
        return;
      }
      const ValueState *ExistingValS = State->get<GCValueMap>(NewSym);
      logWithDump("- Find ExistingValS", ExistingValS);
      if (ExistingValS && ExistingValS->isRooted() &&
          ExistingValS->RootDepth < ValS.RootDepth)
        return;
      logWithDump("- Set value state, Sym", NewSym);
      logWithDump("- Set value state, VS", ValS);
      C.addTransition(State->set<GCValueMap>(NewSym, ValS));
      return;
    }
  }
  checkDerivingExpr(ASE, ASE->getLHS(), true, C);
}

void GCChecker::checkPostStmt(const MemberExpr *ME, CheckerContext &C) const {
  logWithDump("checkPostStmt(MemberExpr)", ME);
  // It is possible for the member itself to be gcrooted, so check that first
  const MemRegion *Region = C.getSVal(ME).getAsRegion();
  ProgramStateRef State = C.getState();
  if (Region && isGCTracked(ME)) {
    if (const RootState *RS = State->get<GCRootMap>(Region)) {
      ValueState ValS = getRootedFromRegion(Region, State->get<GCPinMap>(Region), RS->RootedAtDepth);
      SymbolRef NewSym = getSymbolForResult(ME, &ValS, State, C);
      if (!NewSym)
        return;
      const ValueState *ExistingValS = State->get<GCValueMap>(NewSym);
      if (ExistingValS && ExistingValS->isRooted() &&
          ExistingValS->RootDepth < ValS.RootDepth)
        return;
      C.addTransition(C.getState()->set<GCValueMap>(NewSym, ValS));
      return;
    }
  }
  if (!ME->getType()->isPointerType())
    return;
  clang::Expr *Base = ME->getBase();
  checkDerivingExpr(ME, Base, true, C);
}

void GCChecker::checkPostStmt(const UnaryOperator *UO,
                              CheckerContext &C) const {
  logWithDump("checkPostStmt(UnaryOperator)", UO);
  if (UO->getOpcode() == UO_Deref) {
    checkDerivingExpr(UO, UO->getSubExpr(), true, C);
  }
}

USED_FUNC void GCChecker::dumpState(const ProgramStateRef &State) {
  GCValueMapTy AMap = State->get<GCValueMap>();
  llvm::raw_ostream &Out = llvm::outs();
  Out << "State: "
      << "\n";
  for (auto I = AMap.begin(), E = AMap.end(); I != E; ++I) {
    I.getKey()->dumpToStream(Out);
  }
}

void GCChecker::checkPreCall(const CallEvent &Call, CheckerContext &C) const {
  if (!gcEnabledHere(C))
    return;
  logWithDump("checkPreCall", Call);
  unsigned NumArgs = Call.getNumArgs();
  ProgramStateRef State = C.getState();
  bool isCalleeSafepoint = isSafepoint(Call);
  auto *Decl = Call.getDecl();
  const FunctionDecl *FD = Decl ? Decl->getAsFunction() : nullptr;
  if (!safepointEnabledHere(C) && isCalleeSafepoint) {
    // Suppress this warning if the function is noreturn.
    // We could separate out "not safepoint, except for noreturn functions",
    // but that seems like a lot of effort with little benefit.
    if (!FD || !FD->isNoReturn()) {
      report_error(
          [&](PathSensitiveBugReport *Report) {
            if (FD)
              Report->addNote(
                  "Tried to call method defined here",
                  PathDiagnosticLocation::create(FD, C.getSourceManager()));
          },
          C, ("Calling potential safepoint as " +
              Call.getKindAsString() + " from function annotated JL_NOTSAFEPOINT").str());
      return;
    }
  }
  if (FD && FD->getDeclName().isIdentifier() &&
      FD->getName() == "JL_GC_PROMISE_ROOTED")
    return;
  if (FD && FD->getDeclName().isIdentifier() &&
      (FD->getName() == "PTR_PIN" || FD->getName() == "PTR_UNPIN"))
    return;
  for (unsigned idx = 0; idx < NumArgs; ++idx) {
    SVal Arg = Call.getArgSVal(idx);
    logWithDump("- Argument SVal", Arg);
    SymbolRef Sym = Arg.getAsSymbol();
    // Hack to work around passing unions/structs by value.
    if (auto LCV = Arg.getAs<nonloc::LazyCompoundVal>()) {
      const MemRegion *R = LCV->getRegion();
      if (R) {
        if (const SubRegion *SR = R->getAs<SubRegion>()) {
          if (const SymbolicRegion *SSR =
                  SR->getSuperRegion()->getAs<SymbolicRegion>()) {
            Sym = SSR->getSymbol();
          }
        }
      }
    }
    if (!Sym) {
      log("- No Sym");
      continue;
    }
    auto *ValState = State->get<GCValueMap>(Sym);
    if (!ValState) {
      log("- No ValState");
      continue;
    }
    SourceRange range;
    if (const Expr *E = Call.getArgExpr(idx)) {
      range = E->getSourceRange();
      if (!isGCTracked(E)) {
        log("- Not GCTracked");
        continue;
      }
    }
    logWithDump("- ValState", ValState);
    validateValue(ValState, C, Sym, "Argument value may have been", range);
    if (!ValState->isRooted()) {
      bool MaybeUnrooted = false;
      if (FD) {
        if (idx < FD->getNumParams()) {
          MaybeUnrooted =
              declHasAnnotation(FD->getParamDecl(idx), "julia_maybe_unrooted");
        }
      }
      if (!MaybeUnrooted && isCalleeSafepoint) {
        report_value_error(
            C, Sym,
            "Passing non-rooted value as argument to function that may GC",
            range);
      }
    }
    if (FD && idx < FD->getNumParams() && declHasAnnotation(FD->getParamDecl(idx), "julia_require_tpin")) {
      if (!ValState->isTransitivelyPinned()) {
        report_value_error(C, Sym, "Passing non-tpinned argument to function that requires a tpin argument.");
      }
    }
    if (FD && idx < FD->getNumParams() && declHasAnnotation(FD->getParamDecl(idx), "julia_require_pin")) {
      if (!ValState->isPinnedByAnyway()) {
        report_value_error(C, Sym, "Passing non-pinned argument to function that requires a pin argument.");
      }
    }
  }
}

bool GCChecker::evalCall(const CallEvent &Call, CheckerContext &C) const {
  // These checks should have no effect on the surrounding environment
  // (globals should not be invalidated, etc), hence the use of evalCall.
  const CallExpr *CE = dyn_cast<CallExpr>(Call.getOriginExpr());
  if (!CE)
    return false;
  unsigned CurrentDepth = C.getState()->get<GCDepth>();
  logWithDump("evalCall", Call);
  auto name = C.getCalleeName(CE);
  if (name == "JL_GC_POP") {
    if (CurrentDepth == 0) {
      report_error(C, "JL_GC_POP without corresponding push");
      return true;
    }
    CurrentDepth -= 1;
    // Go through all roots, see which ones are no longer with us.
    // The go through the values and unroot those for which those were our
    // roots.
    ProgramStateRef State = C.getState()->set<GCDepth>(CurrentDepth);
    GCRootMapTy AMap = State->get<GCRootMap>();
    SmallVector<const MemRegion *, 5> PoppedRoots;
    for (auto I = AMap.begin(), E = AMap.end(); I != E; ++I) {
      if (I.getData().shouldPopAtDepth(CurrentDepth)) {
        PoppedRoots.push_back(I.getKey());
        State = State->remove<GCRootMap>(I.getKey());
        State = State->remove<GCPinMap>(I.getKey());
        logWithDump("- pop root", I.getKey());
      }
    }
    log("- Iterate value map");
    GCValueMapTy VMap = State->get<GCValueMap>();
    for (const MemRegion *R : PoppedRoots) {
      logWithDump("-- check popped root", R);
      for (auto I = VMap.begin(), E = VMap.end(); I != E; ++I) {
        logWithDump("--- check value", I.getKey());
        logWithDump("--- check state", I.getData());
        // FIXME: If this is a pop for TPin frame, we should remove TPin as well.
        // For any region that is reachable from R, its pinning state should be reset.
        if (I.getData().isRootedBy(R)) {
          logWithDump("--- no longer rooted", ValueState::getAllocated());
          State =
              State->set<GCValueMap>(I.getKey(), ValueState::getAllocated());
        }
      }
    }
    C.addTransition(State);
    return true;
  } else if (name == "JL_GC_PUSH1" || name == "JL_GC_PUSH2" ||
             name == "JL_GC_PUSH3" || name == "JL_GC_PUSH4" ||
             name == "JL_GC_PUSH5" || name == "JL_GC_PUSH6" ||
             name == "JL_GC_PUSH7" || name == "JL_GC_PUSH8" ||
             name == "JL_GC_PUSH1_NO_TPIN" || name == "JL_GC_PUSH2_NO_TPIN" ||
             name == "JL_GC_PUSH3_NO_TPIN" || name == "JL_GC_PUSH4_NO_TPIN" ||
             name == "JL_GC_PUSH5_NO_TPIN" || name == "JL_GC_PUSH6_NO_TPIN" ||
             name == "JL_GC_PUSH7_NO_TPIN" || name == "JL_GC_PUSH8_NO_TPIN" ) {
    // transitivelypin or pin
    bool tpin = !name.endswith("_NO_TPIN");
    ProgramStateRef State = C.getState();
    // Transform slots to roots, transform values to rooted
    unsigned NumArgs = CE->getNumArgs();
    for (unsigned i = 0; i < NumArgs; ++i) {
      SVal V = C.getSVal(CE->getArg(i));
      auto MRV = V.getAs<loc::MemRegionVal>();
      if (!MRV) {
        report_error(C, "JL_GC_PUSH with something other than a local variable");
        return true;
      }
      const MemRegion *Region = MRV->getRegion();
      RootState RS = RootState::getRoot(CurrentDepth);
      State = State->set<GCRootMap>(Region, RS);
      logWithDump("- JL_GC_PUSH, Region", Region);
      logWithDump("- JL_GC_PUSH, RS", RS);
      PinState PS = PinState::getNoPin(-1);
      if (tpin)
        PS = PinState::getTransitivePin(CurrentDepth);
      else
        PS = PinState::getPin(CurrentDepth);
      State = State->set<GCPinMap>(Region, PS);
      logWithDump("- JL_GC_PUSH, PS", PS);
      // Now for the value
      SVal Value = State->getSVal(Region);
      SymbolRef Sym = Value.getAsSymbol();
      if (!Sym)
        continue;
      const ValueState *ValState = State->get<GCValueMap>(Sym);
      if (!ValState)
        continue;
      validateValue(ValState, C, Sym, "Trying to root value which may have been");
      ValueState VS = *ValState;
      if (!ValState->isRooted()) {
        VS = ValueState::getRooted(Region, CurrentDepth);
      }
      if (tpin)
        VS = ValueState::getTransitivelyPinned(VS);
      else
        VS = ValueState::getPinned(VS);
      State = State->set<GCValueMap>(Sym, VS);
      logWithDump("- JL_GC_PUSH, Sym", Sym);
      logWithDump("- JL_GC_PUSH, VS", VS);
    }
    CurrentDepth += 1;
    State = State->set<GCDepth>(CurrentDepth);
    C.addTransition(State);
    return true;
  } else if (name == "_JL_GC_PUSHARGS" || name == "_JL_GC_PUSHARGS_NO_TPIN") {
    // transitivelypin or pin
    bool tpin = !name.endswith("_NO_TPIN");
    ProgramStateRef State = C.getState();
    SVal ArgArray = C.getSVal(CE->getArg(0));
    auto MRV = ArgArray.getAs<loc::MemRegionVal>();
    if (!MRV) {
      report_error(C, "JL_GC_PUSH with something other than an args array");
      return true;
    }
    const MemRegion *Region = MRV->getRegion()->StripCasts();
    State =
        State->set<GCRootMap>(Region, RootState::getRootArray(CurrentDepth));
    if (tpin) {
      logWithDump("- Root and transitive pin", Region);
      State = State->set<GCPinMap>(Region, PinState::getTransitivePin(CurrentDepth));
    } else {
      logWithDump("- Root and pin", Region);
      State = State->set<GCPinMap>(Region, PinState::getPin(CurrentDepth));
    }
    // The Argument array may also be used as a value, so make it rooted
    // SymbolRef ArgArraySym = ArgArray.getAsSymbol();
    // assert(ArgArraySym);
    // ValueState VS = ValueState::getRooted(Region, CurrentDepth);
    // if (tpin) {
    //   VS = ValueState::getTransitivelyPinned(VS);
    // } else {
    //   VS = ValueState::getPinned(VS);
    // }
    // State = State->set<GCValueMap>(ArgArraySym, VS);
    CurrentDepth += 1;
    State = State->set<GCDepth>(CurrentDepth);
    C.addTransition(State);
    return true;
  } else if (name == "JL_GC_PROMISE_ROOTED") {
    SVal Arg = C.getSVal(CE->getArg(0));
    SymbolRef Sym = Arg.getAsSymbol();
    if (!Sym) {
      report_error(C, "Can not understand this promise.");
      return true;
    }
    C.addTransition(
        C.getState()->set<GCValueMap>(Sym, ValueState::getRooted(nullptr, ValueState::NotPinned -1)));
    return true;
  } else if (name == "PTR_PIN" || name == "PTRHASH_PIN") {
    SVal Arg = C.getSVal(CE->getArg(0));
    SymbolRef Sym = Arg.getAsSymbol();
    if (!Sym) {
      report_error(C, "Can not understand this pin.");
      return true;
    }

    const ValueState *OldVS = C.getState()->get<GCValueMap>(Sym);
    logWithDump("- PTR_PIN OldVS", OldVS);
    if (OldVS && OldVS->isMoved()) {
      report_error(C, "Attempt to PIN a value that is already moved.");
      return true;
    }

    auto MRV = Arg.getAs<loc::MemRegionVal>();
    if (!MRV) {
      report_error(C, "PTR_PIN with something other than a local variable");
      return true;
    }
    const MemRegion *Region = MRV->getRegion();
    auto State = C.getState()->set<GCPinMap>(Region, PinState::getPin(CurrentDepth));
    logWithDump("- Pin region", Region);
    State = State->set<GCValueMap>(Sym, ValueState::getPinned(*OldVS));
    logWithDump("- Pin value", Sym);
    C.addTransition(State);
    return true;
  } else if (name == "PTR_UNPIN" || name == "PTRHASH_UNPIN") {
    SVal Arg = C.getSVal(CE->getArg(0));
    SymbolRef Sym = Arg.getAsSymbol();
    if (!Sym) {
      report_error(C, "Can not understand this unpin.");
      return true;
    }
    auto MRV = Arg.getAs<loc::MemRegionVal>();
    if (!MRV) {
      report_error(C, "PTR_UNPIN with something other than a local variable");
      return true;
    }
    const MemRegion *Region = MRV->getRegion();
    auto State = C.getState();
    auto OldPinState = C.getState()->get<GCPinMap>(Region);
    logWithDump("- Old pin state", OldPinState);
    if (!OldPinState || !OldPinState->isPin()) {
      report_error(C, "PTR_UNPIN with a region that is not pinned");
      return true;
    }
    const ValueState *OldVS = State->get<GCValueMap>(Sym);
    logWithDump("- Old value state", OldVS);
    if (!OldVS || !OldVS->isPinned()) {
      report_error(C, "PTR_UNPIN with a value that is not pinned");
      return true;
    }
    State = State->set<GCPinMap>(Region, PinState::getNoPin(CurrentDepth));
    logWithDump("- Unpin region", Region);
    State = State->set<GCValueMap>(Sym, ValueState::getNotPinned(*OldVS));
    logWithDump("- Unpin value", Sym);
    C.addTransition(State);
    return true;
  } else if (name == "jl_gc_push_arraylist") {
    CurrentDepth += 1;
    ProgramStateRef State = C.getState()->set<GCDepth>(CurrentDepth);
    SVal ArrayList = C.getSVal(CE->getArg(1));
    // Try to find the items field
    FieldDecl *FD = NULL;
    RecordDecl *RD = dyn_cast_or_null<RecordDecl>(
        CE->getArg(1)->getType()->getPointeeType()->getAsTagDecl());
    if (RD) {
      for (FieldDecl *X : RD->fields()) {
        if (X->getName() == "items") {
          FD = X;
          break;
        }
      }
    }
    if (FD) {
      Loc ItemsLoc = State->getLValue(FD, ArrayList).getAs<Loc>().getValue();
      SVal Items = State->getSVal(ItemsLoc);
      if (Items.isUnknown()) {
        Items = C.getSValBuilder().conjureSymbolVal(
            CE, C.getLocationContext(), FD->getType(), C.blockCount());
        State = State->bindLoc(ItemsLoc, Items, C.getLocationContext());
      }
      assert(Items.getAsRegion());
      // The items list is now rooted
      State = State->set<GCRootMap>(Items.getAsRegion(),
                                    RootState::getRootArray(CurrentDepth));
      State = State->set<GCPinMap>(Items.getAsRegion(), PinState::getPin(CurrentDepth));
    }
    C.addTransition(State);
    return true;
  } else if (name == "jl_ast_preserve") {
    // TODO: Maybe bind the rooting to the context. For now, the second
    //       argument gets unconditionally rooted
    ProgramStateRef State = C.getState();
    SymbolRef Sym = C.getSVal(CE->getArg(1)).getAsSymbol();
    if (!Sym)
      return true;
    C.addTransition(
        State->set<GCValueMap>(Sym, ValueState::getRooted(nullptr, ValueState::Pinned, -1)));
    return true;
  } else if (name == "jl_gc_enable" || name == "ijl_gc_enable") {
    ProgramStateRef State = C.getState();
    // Check for a literal argument
    SVal Arg = C.getSVal(CE->getArg(0));
    auto CI = Arg.getAs<nonloc::ConcreteInt>();
    bool EnabledAfter = true;
    if (CI) {
      const llvm::APSInt &Val = CI->getValue();
      EnabledAfter = Val != 0;
    } else {
      cast<SymbolConjured>(Arg.getAsSymbol())->getStmt()->dump();
    }
    bool EnabledNow = State->get<GCDisabledAt>() == (unsigned)-1;
    if (!EnabledAfter) {
      State = State->set<GCDisabledAt>((unsigned)-2);
    } else {
      State = State->set<GCDisabledAt>((unsigned)-1);
    }
    // GC State is explicitly modeled, so let's make sure
    // the execution matches our model
    SVal Result = C.getSValBuilder().makeTruthVal(EnabledNow, CE->getType());
    C.addTransition(State->BindExpr(CE, C.getLocationContext(), Result));
    return true;
  }
  else if (name == "uv_mutex_lock") {
    ProgramStateRef State = C.getState();
    if (State->get<SafepointDisabledAt>() == (unsigned)-1) {
      C.addTransition(State->set<SafepointDisabledAt>(C.getStackFrame()->getIndex()));
      return true;
    }
  }
  else if (name == "uv_mutex_unlock") {
    ProgramStateRef State = C.getState();
    const auto *LCtx = C.getLocationContext();
    const auto *FD = dyn_cast<FunctionDecl>(LCtx->getDecl());
    if (State->get<SafepointDisabledAt>() == (unsigned)C.getStackFrame()->getIndex() &&
        !isFDAnnotatedNotSafepoint(FD)) {
      C.addTransition(State->set<SafepointDisabledAt>(-1));
      return true;
    }
  }
  return false;
}

void GCChecker::checkBind(SVal LVal, SVal RVal, const clang::Stmt *S,
                          CheckerContext &C) const {
  log("checkBind");
  logWithDump("- LVal", LVal);
  logWithDump("- RVal", RVal);
  auto State = C.getState();
  const MemRegion *R = LVal.getAsRegion();
  if (!R) {
    log("- LValue is not a region, return");
    return;
  }
  bool shouldBeRootArray = false;
  const ElementRegion *ER = R->getAs<ElementRegion>();
  if (ER) {
    R = R->getBaseRegion()->StripCasts();
    shouldBeRootArray = true;
  }
  SymbolRef Sym = RVal.getAsSymbol();
  if (!Sym) {
    log("- No Sym");
    return;
  }
  logWithDump("- Sym", Sym);
  const auto *RootState = State->get<GCRootMap>(R);
  logWithDump("- R", R);
  logWithDump("- RootState for R", RootState);
  if (!RootState) {
    const ValueState *ValSP = nullptr;
    ValueState ValS;
    if (rootRegionIfGlobal(R->getBaseRegion(), State, C, &ValS)) {
      logWithDump("- rootRegionIfGlobal, base", R->getBaseRegion());
      ValSP = &ValS;
      logWithDump("- rootRegionIfGlobal ValSP", ValSP);
    } else {
      logWithDump("- getValStateForRegion", R);
      ValSP = getValStateForRegion(C.getASTContext(), State, R);
      logWithDump("- getValStateForRegion", ValSP);
    }
    if (ValSP && ValSP->isRooted()) {
      logWithDump("- Found base region that is rooted", ValSP);
      const auto *RValState = State->get<GCValueMap>(Sym);
      if (RValState && RValState->isRooted() &&
          RValState->RootDepth < ValSP->RootDepth) {
        logWithDump("- No need to set ValState, current ValState", RValState);
      } else {
        auto InheritVS = ValueState::inheritState(*ValSP);
        logWithDump("- Set ValState, InheritVS", InheritVS);
        C.addTransition(State->set<GCValueMap>(Sym, InheritVS));
      }
    }
  } else {
    if (shouldBeRootArray && !RootState->isRootArray()) {
      report_error(
          C, "This assignment looks weird. Expected a root array on the LHS.");
      return;
    }
    const auto *RValState = State->get<GCValueMap>(Sym);
    if (!RValState) {
      if (rootRegionIfGlobal(Sym->getOriginRegion(), State, C)) {
        log("- Cannot find ValState for Sym, root it as global");
        C.addTransition(State);
      } else {
        Sym->dump();
        if (auto *SC = dyn_cast<SymbolConjured>(Sym)) {
          SC->getStmt()->dump();
        }
        report_value_error(C, Sym,
                          "Saw assignment to root, but missed the allocation");
      }
    } else {
      logWithDump("- Found ValState for Sym", RValState);
      validateValue(RValState, C, Sym, "Trying to root value which may have been");
      if (!RValState->isRooted() || !RValState->isPinnedByAnyway() ||
          RValState->RootDepth > RootState->RootedAtDepth) {
        auto NewVS = getRootedFromRegion(R, State->get<GCPinMap>(R), RootState->RootedAtDepth);
        logWithDump("- getRootedFromRegion", NewVS);
        C.addTransition(State->set<GCValueMap>(
            Sym, NewVS));
      }
    }
  }
}

bool GCChecker::rootRegionIfGlobal(const MemRegion *R, ProgramStateRef &State,
                                   CheckerContext &C, ValueState *ValS) const {
  if (!R)
    return false;
  const VarRegion *VR = R->getAs<VarRegion>();
  if (!VR)
    return false;
  const VarDecl *VD = VR->getDecl();
  assert(VD);
  if (!VD->hasGlobalStorage())
    return false;
  if (!isGCTrackedType(VD->getType()))
    return false;
  bool isGlobalRoot = false;
  ValueState::Pin pinState;
  if (declHasAnnotation(VD, "julia_globally_rooted") ||
      isGloballyRootedType(VD->getType())) {
    State = State->set<GCRootMap>(R, RootState::getRoot(-1));
    logWithDump("- rootRegionIfGlobal: root global", R);
    if (isGloballyTransitivelyPinnedType(VD->getType()) || declHasAnnotation(VD, "julia_globally_tpinned")) {
      State = State->set<GCPinMap>(R, PinState::getTransitivePin(-1));
      logWithDump("- rootRegionIfGlobal: transitively pin global", R);
      pinState = ValueState::TransitivelyPinned;
    } else if (declHasAnnotation(VD, "julia_globally_pinned")) {
      State = State->set<GCPinMap>(R, PinState::getPin(-1));
      logWithDump("- rootRegionIfGlobal: pin global", R);
      pinState = ValueState::Pinned;
    } else {
      logWithDump("- rootRegionIfGlobal: not pin", R);
      pinState = ValueState::NotPinned;
    }
    isGlobalRoot = true;
  }
  SVal TheVal = State->getSVal(R);
  SymbolRef Sym = TheVal.getAsSymbol();
  ValueState TheValS(isGlobalRoot ? ValueState::getRooted(R, pinState, -1)
                                  : ValueState::getAllocated());
  if (ValS) {
    *ValS = ValueState::inheritState(TheValS);
    logWithDump("- rootRegionIfGlobal: inherit state", TheValS);
  }
  if (Sym) {
    const ValueState *GVState = C.getState()->get<GCValueMap>(Sym);
    if (!GVState) {
      State = State->set<GCValueMap>(Sym, TheValS);
      logWithDump("- rootRegionIfGlobal: Sym", Sym);
      logWithDump("- rootRegionIfGlobal: VS", TheValS);
    }
  }
  return true;
}

void GCChecker::checkLocation(SVal SLoc, bool IsLoad, const Stmt *S,
                              CheckerContext &C) const {
  logWithDump("checkLocation", SLoc);
  ProgramStateRef State = C.getState();
  bool DidChange = false;
  const RootState *RS = nullptr;
  // Loading from a root produces a rooted symbol. TODO: Can we do something
  // better than this.
  if (IsLoad && (RS = State->get<GCRootMap>(SLoc.getAsRegion()))) {
    logWithDump("- IsLoad, RS", RS);
    SymbolRef LoadedSym =
        State->getSVal(SLoc.getAs<Loc>().getValue()).getAsSymbol();
    if (LoadedSym) {
      const ValueState *ValS = State->get<GCValueMap>(LoadedSym);
      logWithDump("- IsLoad, LoadedSym", LoadedSym);
      logWithDump("- IsLoad, ValS", ValS);
      if (!ValS || !ValS->isRooted() || !ValS->isPinnedByAnyway() || ValS->RootDepth > RS->RootedAtDepth) {
        auto NewVS = getRootedFromRegion(SLoc.getAsRegion(), State->get<GCPinMap>(SLoc.getAsRegion()), RS->RootedAtDepth);
        logWithDump("- IsLoad, NewVS", NewVS);
        DidChange = true;
        State = State->set<GCValueMap>(LoadedSym, NewVS);
      }
    }
  }
  logWithDump("- getAsRegion()", SLoc.getAsRegion());
  // If it's just the symbol by itself, let it be. We allow dead pointer to be
  // passed around, so long as they're not accessed. However, we do want to
  // start tracking any globals that may have been accessed.
  if (rootRegionIfGlobal(SLoc.getAsRegion(), State, C)) {
    C.addTransition(State);
    log("- rootRegionIfGlobal");
    return;
  }
  SymbolRef SymByItself = SLoc.getAsSymbol(false);
  logWithDump("- SymByItself", SymByItself);
  if (SymByItself) {
    DidChange &&C.addTransition(State);
    return;
  }
  // This will walk backwards until it finds the base symbol
  SymbolRef Sym = SLoc.getAsSymbol(true);
  logWithDump("- Sym", Sym);
  if (!Sym) {
    DidChange &&C.addTransition(State);
    return;
  }
  const ValueState *VState = State->get<GCValueMap>(Sym);
  logWithDump("- VState", VState);
  if (!VState) {
    DidChange &&C.addTransition(State);
    return;
  }
  // If this is the sym, we verify both rootness and pinning. Otherwise, it may be the parent sym and we only care about the rootness.
  if (SymByItself == Sym) {
    validateValue(VState, C, Sym, "Trying to access value which may have been");
  } else {
    validateValueRootnessOnly(VState, C, Sym, "Trying to access value which may have been");
  }
  DidChange &&C.addTransition(State);
}

namespace clang {
namespace ento {
void registerGCChecker(CheckerManager &mgr) {
  mgr.registerChecker<GCChecker>();
}
} // namespace ento
} // namespace clang

#ifdef CLANG_PLUGIN
extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
extern "C" void clang_registerCheckers(CheckerRegistry &registry) {
  registry.addChecker<GCChecker>(
      "julia.GCChecker", "Validates julia gc invariants",
      "https://docs.julialang.org/en/v1/devdocs/gc-sa/"
  );
}
#endif
