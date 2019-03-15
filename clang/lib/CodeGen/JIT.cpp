//===--- CodeGenAction.cpp - LLVM Code Generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGen/CodeGenAction.h"
#include "CodeGenModule.h"
#include "CoverageMappingGen.h"
#include "CGCXXABI.h"
#include "MacroPPCallbacks.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/MemoryBufferCache.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO/Internalize.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#include <unordered_map>
using namespace clang;
using namespace llvm;

#define DEBUG_TYPE "clang-jit"

namespace {
// FIXME: This is copied from lib/Frontend/ASTUnit.cpp

/// Gathers information from ASTReader that will be used to initialize
/// a Preprocessor.
class ASTInfoCollector : public ASTReaderListener {
  Preprocessor &PP;
  ASTContext *Context;
  HeaderSearchOptions &HSOpts;
  PreprocessorOptions &PPOpts;
  LangOptions &LangOpt;
  std::shared_ptr<clang::TargetOptions> &TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> &Target;
  unsigned &Counter;
  bool InitializedLanguage = false;

public:
  ASTInfoCollector(Preprocessor &PP, ASTContext *Context,
                   HeaderSearchOptions &HSOpts, PreprocessorOptions &PPOpts,
                   LangOptions &LangOpt,
                   std::shared_ptr<clang::TargetOptions> &TargetOpts,
                   IntrusiveRefCntPtr<TargetInfo> &Target, unsigned &Counter)
      : PP(PP), Context(Context), HSOpts(HSOpts), PPOpts(PPOpts),
        LangOpt(LangOpt), TargetOpts(TargetOpts), Target(Target),
        Counter(Counter) {}

  bool ReadLanguageOptions(const LangOptions &LangOpts, bool Complain,
                           bool AllowCompatibleDifferences) override {
    if (InitializedLanguage)
      return false;

    LangOpt = LangOpts;
    InitializedLanguage = true;

    updated();
    return false;
  }

  bool ReadHeaderSearchOptions(const HeaderSearchOptions &HSOpts,
                               StringRef SpecificModuleCachePath,
                               bool Complain) override {
    this->HSOpts = HSOpts;
    return false;
  }

  bool ReadPreprocessorOptions(const PreprocessorOptions &PPOpts, bool Complain,
                               std::string &SuggestedPredefines) override {
    this->PPOpts = PPOpts;
    return false;
  }

  bool ReadTargetOptions(const clang::TargetOptions &TargetOpts, bool Complain,
                         bool AllowCompatibleDifferences) override {
    // If we've already initialized the target, don't do it again.
    if (Target)
      return false;

    this->TargetOpts = std::make_shared<clang::TargetOptions>(TargetOpts);
    Target =
        TargetInfo::CreateTargetInfo(PP.getDiagnostics(), this->TargetOpts);

    updated();
    return false;
  }

  void ReadCounter(const serialization::ModuleFile &M,
                   unsigned Value) override {
    Counter = Value;
  }

private:
  void updated() {
    if (!Target || !InitializedLanguage)
      return;

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Target->adjust(LangOpt);

    // Initialize the preprocessor.
    PP.Initialize(*Target);

    if (!Context)
      return;

    // Initialize the ASTContext
    Context->InitBuiltinTypes(*Target);

    // Adjust printing policy based on language options.
    Context->setPrintingPolicy(PrintingPolicy(LangOpt));

    // We didn't have access to the comment options when the ASTContext was
    // constructed, so register them now.
    Context->getCommentCommandTraits().registerCommentOptions(
        LangOpt.CommentOpts);
  }
};

void fatal() {
  report_fatal_error("Clang JIT failed!");
}

// This is a variant of ORC's LegacyLookupFnResolver with a cutomized
// getResponsibilitySet behavior allowing us to claim responsibility for weak
// symbols in the loaded modules that we don't otherwise have.
// Note: We generally convert all IR level symbols to have strong linkage, but
// that won't cover everything (and especially doesn't cover the DW.ref.
// symbols created by the low-level EH logic on some platforms).
template <typename LegacyLookupFn>
class ClangLookupFnResolver final : public llvm::orc::SymbolResolver {
public:
  using ErrorReporter = std::function<void(Error)>;

  ClangLookupFnResolver(llvm::orc::ExecutionSession &ES,
                              LegacyLookupFn LegacyLookup,
                              ErrorReporter ReportError)
      : ES(ES), LegacyLookup(std::move(LegacyLookup)),
        ReportError(std::move(ReportError)) {}

  llvm::orc::SymbolNameSet
  getResponsibilitySet(const llvm::orc::SymbolNameSet &Symbols) final {
    llvm::orc::SymbolNameSet Result;

    for (auto &S : Symbols) {
      if (JITSymbol Sym = LegacyLookup(*S)) {
        // If the symbol exists elsewhere, and we have only a weak version,
        // then we're not responsible.
        continue;
      } else if (auto Err = Sym.takeError()) {
        ReportError(std::move(Err));
        return llvm::orc::SymbolNameSet();
      } else {
        Result.insert(S);
      }
    }

    return Result;
  }

  llvm::orc::SymbolNameSet
  lookup(std::shared_ptr<llvm::orc::AsynchronousSymbolQuery> Query,
                         llvm::orc::SymbolNameSet Symbols) final {
    return llvm::orc::lookupWithLegacyFn(ES, *Query, Symbols, LegacyLookup);
  }

private:
  llvm::orc::ExecutionSession &ES;
  LegacyLookupFn LegacyLookup;
  ErrorReporter ReportError;
};

template <typename LegacyLookupFn>
std::shared_ptr<ClangLookupFnResolver<LegacyLookupFn>>
createClangLookupResolver(llvm::orc::ExecutionSession &ES,
                          LegacyLookupFn LegacyLookup,
                          std::function<void(Error)> ErrorReporter) {
  return std::make_shared<ClangLookupFnResolver<LegacyLookupFn>>(
      ES, std::move(LegacyLookup), std::move(ErrorReporter));
}

class ClangJIT {
public:
  using ObjLayerT = llvm::orc::LegacyRTDyldObjectLinkingLayer;
  using CompileLayerT = llvm::orc::LegacyIRCompileLayer<ObjLayerT, llvm::orc::SimpleCompiler>;

  ClangJIT(DenseMap<StringRef, const void *> &LocalSymAddrs)
      : LocalSymAddrs(LocalSymAddrs),
        Resolver(createClangLookupResolver(
            ES,
            [this](const std::string &Name) {
              return findSymbol(Name);
            },
            [](Error Err) { cantFail(std::move(Err), "lookupFlags failed"); })),
        TM(EngineBuilder().selectTarget()), DL(TM->createDataLayout()),
        ObjectLayer(ES,
                    [this](llvm::orc::VModuleKey) {
                      return ObjLayerT::Resources{
                          std::make_shared<SectionMemoryManager>(), Resolver};
                    }),
        CompileLayer(ObjectLayer, llvm::orc::SimpleCompiler(*TM)) {
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);
  }

  llvm::TargetMachine &getTargetMachine() { return *TM; }

  llvm::orc::VModuleKey addModule(std::unique_ptr<llvm::Module> M) {
    auto K = ES.allocateVModule();
    cantFail(CompileLayer.addModule(K, std::move(M)));
    ModuleKeys.push_back(K);
    return K;
  }

  void removeModule(llvm::orc::VModuleKey K) {
    ModuleKeys.erase(find(ModuleKeys, K));
    cantFail(CompileLayer.removeModule(K));
  }

  llvm::JITSymbol findSymbol(const std::string Name) {
    return findMangledSymbol(mangle(Name));
  }

private:
  std::string mangle(const std::string &Name) {
    std::string MangledName;
    {
      llvm::raw_string_ostream MangledNameStream(MangledName);
      llvm::Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
    }
    return MangledName;
  }

  llvm::JITSymbol findMangledSymbol(const std::string &Name) {
    for (auto H : make_range(ModuleKeys.rbegin(), ModuleKeys.rend()))
      if (auto Sym = CompileLayer.findSymbolIn(H, Name,
                                               /*ExportedSymbolsOnly*/ false))
        return Sym;

    auto LSAI = LocalSymAddrs.find(Name);
    if (LSAI != LocalSymAddrs.end())
      return llvm::JITSymbol(llvm::pointerToJITTargetAddress(LSAI->second),
                             llvm::JITSymbolFlags::Exported);

    // If we can't find the symbol in the JIT, try looking in the host process.
    if (auto SymAddr = RTDyldMemoryManager::getSymbolAddressInProcess(Name))
      return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);

#ifdef _WIN32
    // For Windows retry without "_" at beginning, as RTDyldMemoryManager uses
    // GetProcAddress and standard libraries like msvcrt.dll use names
    // with and without "_" (for example "_itoa" but "sin").
    if (Name.length() > 2 && Name[0] == '_')
      if (auto SymAddr =
              RTDyldMemoryManager::getSymbolAddressInProcess(Name.substr(1)))
        return llvm::JITSymbol(SymAddr, llvm::JITSymbolFlags::Exported);
#endif

    return nullptr;
  }

  DenseMap<StringRef, const void *> &LocalSymAddrs; 
  llvm::orc::ExecutionSession ES;
  std::shared_ptr<llvm::orc::SymbolResolver> Resolver;
  std::unique_ptr<llvm::TargetMachine> TM;
  const llvm::DataLayout DL;
  ObjLayerT ObjectLayer;
  CompileLayerT CompileLayer;
  std::vector<llvm::orc::VModuleKey> ModuleKeys;
};

class BackendConsumer : public ASTConsumer {
  DiagnosticsEngine &Diags;
  BackendAction Action;
  const HeaderSearchOptions &HeaderSearchOpts;
  const CodeGenOptions &CodeGenOpts;
  const clang::TargetOptions &TargetOpts;
  const LangOptions &LangOpts;
  std::unique_ptr<raw_pwrite_stream> AsmOutStream;
  ASTContext *Context;
  std::string InFile;
  const PreprocessorOptions &PPOpts;
  LLVMContext &C;
  CoverageSourceInfo *CoverageInfo;

  std::unique_ptr<CodeGenerator> Gen;

  void replaceGenerator() {
    Gen.reset(CreateLLVMCodeGen(Diags, InFile, HeaderSearchOpts, PPOpts,
                                CodeGenOpts, C, CoverageInfo));
  }

public:
  BackendConsumer(BackendAction Action, DiagnosticsEngine &Diags,
                  const HeaderSearchOptions &HeaderSearchOpts,
                  const PreprocessorOptions &PPOpts,
                  const CodeGenOptions &CodeGenOpts,
                  const clang::TargetOptions &TargetOpts,
                  const LangOptions &LangOpts, bool TimePasses,
                  const std::string &InFile,
                  std::unique_ptr<raw_pwrite_stream> OS, LLVMContext &C,
                  CoverageSourceInfo *CoverageInfo = nullptr)
      : Diags(Diags), Action(Action), HeaderSearchOpts(HeaderSearchOpts),
        CodeGenOpts(CodeGenOpts), TargetOpts(TargetOpts), LangOpts(LangOpts),
        AsmOutStream(std::move(OS)), Context(nullptr), InFile(InFile),
        PPOpts(PPOpts), C(C), CoverageInfo(CoverageInfo) { }

  llvm::Module *getModule() const { return Gen->GetModule(); }
  std::unique_ptr<llvm::Module> takeModule() {
    return std::unique_ptr<llvm::Module>(Gen->ReleaseModule());
  }

  CodeGenerator *getCodeGenerator() { return Gen.get(); }

  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void Initialize(ASTContext &Ctx) override {
    replaceGenerator();
    Context = &Ctx;
    Gen->Initialize(Ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleInterestingDecl(DeclGroupRef D) override {
    HandleTopLevelDecl(D);
  }

  void HandleTranslationUnit(ASTContext &C) override {
      Gen->HandleTranslationUnit(C);

    // Silently ignore if we weren't initialized for some reason.
    if (!getModule())
      return;

    EmitBackendOutput(Diags, HeaderSearchOpts, CodeGenOpts, TargetOpts,
                      LangOpts, C.getTargetInfo().getDataLayout(),
                      getModule(), Action, std::move(AsmOutStream));
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void AssignInheritanceModel(CXXRecordDecl *RD) override {
    Gen->AssignInheritanceModel(RD);
  }

  void HandleVTable(CXXRecordDecl *RD) override {
    Gen->HandleVTable(RD);
  }
};

class JFIMapDeclVisitor : public RecursiveASTVisitor<JFIMapDeclVisitor> {
  DenseMap<unsigned, FunctionDecl *> &Map;

public:
  explicit JFIMapDeclVisitor(DenseMap<unsigned, FunctionDecl *> &M)
    : Map(M) { }

  bool shouldVisitTemplateInstantiations() const { return true; }

  bool VisitFunctionDecl(const FunctionDecl *D) {
    if (auto *A = D->getAttr<JITFuncInstantiationAttr>())
      Map[A->getId()] = const_cast<FunctionDecl *>(D);
    return true;
  }
};

class JFICSMapDeclVisitor : public RecursiveASTVisitor<JFICSMapDeclVisitor> {
  DenseMap<unsigned, FunctionDecl *> &Map;
  SmallVector<FunctionDecl *, 1> CurrentFD;

public:
  explicit JFICSMapDeclVisitor(DenseMap<unsigned, FunctionDecl *> &M)
    : Map(M) { }

  bool TraverseFunctionDecl(FunctionDecl *FD) {
    CurrentFD.push_back(FD);
    bool Continue =
      RecursiveASTVisitor<JFICSMapDeclVisitor>::TraverseFunctionDecl(FD);
    CurrentFD.pop_back();

    return Continue;
  }

  bool VisitDeclRefExpr(DeclRefExpr *E) {
    auto *FD = dyn_cast<FunctionDecl>(E->getDecl());
    if (!FD)
      return true;

    auto *A = FD->getAttr<JITFuncInstantiationAttr>();
    if (!A)
      return true;

    Map[A->getId()] = CurrentFD.back();

    return true;
  }
};

unsigned LastUnique = 0;
std::unique_ptr<llvm::LLVMContext> LCtx;

struct DevData {
  const char *Triple;
  const char *Arch;
  const char *ASTBuffer;
  size_t ASTBufferSize;
  const void *CmdArgs;
  size_t CmdArgsLen;
};

struct CompilerData {
  std::unique_ptr<CompilerInvocation>     Invocation;
  std::unique_ptr<llvm::opt::OptTable>    Opts;
  IntrusiveRefCntPtr<DiagnosticOptions>   DiagOpts;
  std::unique_ptr<TextDiagnosticPrinter>  DiagnosticPrinter;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  IntrusiveRefCntPtr<DiagnosticsEngine>   Diagnostics;
  IntrusiveRefCntPtr<FileManager>         FileMgr;
  IntrusiveRefCntPtr<SourceManager>       SourceMgr;
  IntrusiveRefCntPtr<MemoryBufferCache>   PCMCache;
  std::unique_ptr<HeaderSearch>           HeaderInfo;
  std::unique_ptr<PCHContainerReader>     PCHContainerRdr;
  IntrusiveRefCntPtr<TargetInfo>          Target;
  std::shared_ptr<Preprocessor>           PP;
  IntrusiveRefCntPtr<ASTContext>          Ctx;
  std::shared_ptr<clang::TargetOptions>   TargetOpts;
  std::shared_ptr<HeaderSearchOptions>    HSOpts;
  std::shared_ptr<PreprocessorOptions>    PPOpts;
  IntrusiveRefCntPtr<ASTReader>           Reader;
  std::unique_ptr<BackendConsumer>        Consumer;
  std::unique_ptr<Sema>                   S;
  TrivialModuleLoader                     ModuleLoader;
  std::unique_ptr<llvm::Module>           RunningMod;

  DenseMap<StringRef, const void *>       LocalSymAddrs;
  DenseMap<StringRef, ValueDecl *>        NewLocalSymDecls;
  std::unique_ptr<ClangJIT>               CJ;

  DenseMap<unsigned, FunctionDecl *>      FuncMap;

  // A map of each instantiation to the containing function. These might not be
  // unique, but should be unique for any place where it matters
  // (instantiations with from-string types).
  DenseMap<unsigned, FunctionDecl *>      CSFuncMap;

  CompilerData(const void *CmdArgs, unsigned CmdArgsLen,
               const void *ASTBuffer, size_t ASTBufferSize,
               const void *IRBuffer, size_t IRBufferSize,
               const void **LocalPtrs, unsigned LocalPtrsCnt,
               const void **LocalDbgPtrs, unsigned LocalDbgPtrsCnt,
               const DevData *DeviceData, unsigned DevCnt) {
    StringRef CombinedArgv((const char *) CmdArgs, CmdArgsLen);
    SmallVector<StringRef, 32> Argv;
    CombinedArgv.split(Argv, '\0', /*MaxSplit*/ -1, false);

    llvm::opt::ArgStringList CC1Args;
    for (auto &ArgStr : Argv)
      CC1Args.push_back(ArgStr.begin());

    unsigned MissingArgIndex, MissingArgCount;
    Opts = driver::createDriverOptTable();
    llvm::opt::InputArgList ParsedArgs = Opts->ParseArgs(
      CC1Args, MissingArgIndex, MissingArgCount);

    DiagOpts = new DiagnosticOptions();
    ParseDiagnosticArgs(*DiagOpts, ParsedArgs);
    DiagnosticPrinter.reset(new TextDiagnosticPrinter(
      llvm::errs(), &*DiagOpts));
    Diagnostics = new DiagnosticsEngine(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      DiagnosticPrinter.get(), false);

    // Note that LangOpts, TargetOpts can also be read from the AST, but
    // CodeGenOpts need to come from the stored command line.

    Invocation.reset(new CompilerInvocation);
    CompilerInvocation::CreateFromArgs(*Invocation,
                                 const_cast<const char **>(CC1Args.data()),
                                 const_cast<const char **>(CC1Args.data()) +
                                 CC1Args.size(), *Diagnostics);
    Invocation->getFrontendOpts().DisableFree = false;
    Invocation->getCodeGenOpts().DisableFree = false;

    InMemoryFileSystem = new llvm::vfs::InMemoryFileSystem;
    FileMgr = new FileManager(FileSystemOptions(), InMemoryFileSystem);

    const char *Filename = "__clang_jit.pcm";
    StringRef ASTBufferSR((const char *) ASTBuffer, ASTBufferSize);
    InMemoryFileSystem->addFile(Filename, 0,
                                llvm::MemoryBuffer::getMemBufferCopy(ASTBufferSR));

    PCHContainerRdr.reset(new RawPCHContainerReader);
    SourceMgr = new SourceManager(*Diagnostics, *FileMgr,
                                  /*UserFilesAreVolatile*/ false);
    PCMCache = new MemoryBufferCache;
    HSOpts = std::make_shared<HeaderSearchOptions>();
    HSOpts->ModuleFormat = PCHContainerRdr->getFormat();
    HeaderInfo.reset(new HeaderSearch(HSOpts,
                                      *SourceMgr,
                                      *Diagnostics,
                                      *Invocation->getLangOpts(),
                                      /*Target=*/nullptr));
    PPOpts = std::make_shared<PreprocessorOptions>();

    unsigned Counter;

    PP = std::make_shared<Preprocessor>(
        PPOpts, *Diagnostics, *Invocation->getLangOpts(),
        *SourceMgr, *PCMCache, *HeaderInfo, ModuleLoader,
        /*IILookup=*/nullptr,
        /*OwnsHeaderSearch=*/false);

    // For parsing type names in strings later, we'll need to have Preprocessor
    // keep the Lexer around even after it hits the end of the each file (used
    // for each type name).
    PP->enableIncrementalProcessing();

    Ctx = new ASTContext(*Invocation->getLangOpts(), *SourceMgr,
                         PP->getIdentifierTable(), PP->getSelectorTable(),
                         PP->getBuiltinInfo());

    Reader = new ASTReader(*PP, Ctx.get(), *PCHContainerRdr, {},
                           /*isysroot=*/"",
                           /*DisableValidation=*/ false,
                           /*AllowPCHWithCompilerErrors*/ false);

    Reader->setListener(llvm::make_unique<ASTInfoCollector>(
      *PP, Ctx.get(), *HSOpts, *PPOpts, *Invocation->getLangOpts(),
      TargetOpts, Target, Counter));

    Ctx->setExternalSource(Reader);

    switch (Reader->ReadAST(Filename, serialization::MK_MainFile,
                            SourceLocation(), ASTReader::ARR_None)) {
    case ASTReader::Success:
      break;

    case ASTReader::Failure:
    case ASTReader::Missing:
    case ASTReader::OutOfDate:
    case ASTReader::VersionMismatch:
    case ASTReader::ConfigurationMismatch:
    case ASTReader::HadErrors:
      Diagnostics->Report(diag::err_fe_unable_to_load_pch);
      fatal();
      return;
    }

    PP->setCounterValue(Counter);

    // Now that we've read the language options from the AST file, change the JIT mode.
    Invocation->getLangOpts()->setCPlusPlusJIT(LangOptions::JITMode::JM_IsJIT);

    // Keep externally available functions, etc.
    Invocation->getCodeGenOpts().PrepareForLTO = true;

    BackendAction BA = Backend_EmitNothing;
    std::unique_ptr<raw_pwrite_stream> OS(new llvm::raw_null_ostream);
    Consumer.reset(new BackendConsumer(
        BA, *Diagnostics, Invocation->getHeaderSearchOpts(),
        Invocation->getPreprocessorOpts(), Invocation->getCodeGenOpts(),
        Invocation->getTargetOpts(), *Invocation->getLangOpts(), false, Filename,
        std::move(OS), *LCtx));

    // Create a semantic analysis object and tell the AST reader about it.
    S.reset(new Sema(*PP, *Ctx, *Consumer));
    S->Initialize();
    Reader->InitializeSema(*S);

    // Tell the diagnostic client that we have started a source file.
    Diagnostics->getClient()->BeginSourceFile(PP->getLangOpts(), PP.get());

    JFIMapDeclVisitor(FuncMap).TraverseAST(*Ctx);
    JFICSMapDeclVisitor(CSFuncMap).TraverseAST(*Ctx);

    llvm::SMDiagnostic Err;
    StringRef IRBufferSR((const char *) IRBuffer, IRBufferSize);
    RunningMod = parseIR(
      *llvm::MemoryBuffer::getMemBufferCopy(IRBufferSR), Err, *LCtx);

    for (auto &F : RunningMod->functions())
      if (!F.isDeclaration())
        F.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

    for (auto &GV : RunningMod->global_values())
      if (!GV.isDeclaration())
        GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

    Consumer->Initialize(*Ctx);

    for (unsigned Idx = 0; Idx < 2*LocalPtrsCnt; Idx += 2) {
      const char *Name = (const char *) LocalPtrs[Idx];
      const void *Ptr = LocalPtrs[Idx+1];
      LocalSymAddrs[Name] = Ptr;
    }

    for (unsigned Idx = 0; Idx < 2*LocalDbgPtrsCnt; Idx += 2) {
      const char *Name = (const char *) LocalDbgPtrs[Idx];
      const void *Ptr = LocalDbgPtrs[Idx+1];
      LocalSymAddrs[Name] = Ptr;
    }

    CJ = llvm::make_unique<ClangJIT>(LocalSymAddrs);

if (DevCnt)
llvm::errs() << "Devices: " << DevCnt << "\n";
for (unsigned i = 0; i < DevCnt; ++i) {
  llvm::errs() << i << ": " << DeviceData[i].Triple << ": " << DeviceData[i].Arch << "\n";
}
  }

  void restoreFuncDeclContext(FunctionDecl *FunD) {
    // NOTE: This mirrors the corresponding code in
    // Parser::ParseLateTemplatedFuncDef (which is used to late parse a C++
    // function template in Microsoft mode).

    struct ContainingDC {
      ContainingDC(DeclContext *DC, bool ShouldPush) : Pair(DC, ShouldPush) {}
      llvm::PointerIntPair<DeclContext *, 1, bool> Pair;
      DeclContext *getDC() { return Pair.getPointer(); }
      bool shouldPushDC() { return Pair.getInt(); }
    };

    SmallVector<ContainingDC, 4> DeclContextsToReenter;
    DeclContext *DD = FunD;
    DeclContext *NextContaining = S->getContainingDC(DD);
    while (DD && !DD->isTranslationUnit()) {
      bool ShouldPush = DD == NextContaining;
      DeclContextsToReenter.push_back({DD, ShouldPush});
      if (ShouldPush)
        NextContaining = S->getContainingDC(DD);
      DD = DD->getLexicalParent();
    }

    // Reenter template scopes from outermost to innermost.
    for (ContainingDC CDC : reverse(DeclContextsToReenter)) {
      (void) S->ActOnReenterTemplateScope(S->getCurScope(),
                                           cast<Decl>(CDC.getDC()));
      if (CDC.shouldPushDC())
        S->PushDeclContext(S->getCurScope(), CDC.getDC());
    }
  }

  void *resolveFunction(const void *NTTPValues, const char **TypeStrings,
                        unsigned Idx) {
    FunctionDecl *FD = FuncMap[Idx];
    if (!FD)
      fatal();

    RecordDecl *RD =
      Ctx->buildImplicitRecord(llvm::Twine("__clang_jit_args_")
                               .concat(llvm::Twine(Idx))
                               .concat(llvm::Twine("_t"))
                               .str());

    RD->startDefinition();

    enum TASaveKind {
      TASK_None,
      TASK_Type,
      TASK_Value
    };

    SmallVector<TASaveKind, 8> TAIsSaved;

    auto *FTSI = FD->getTemplateSpecializationInfo();
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      auto HandleTA = [&](const TemplateArgument &TA) {
        if (TA.getKind() == TemplateArgument::Type)
          if (TA.getAsType()->isJITFromStringType()) {
            TAIsSaved.push_back(TASK_Type);
            return;
          }

        if (TA.getKind() != TemplateArgument::Expression) {
          TAIsSaved.push_back(TASK_None);
          return;
        }

        SmallVector<PartialDiagnosticAt, 8> Notes;
        Expr::EvalResult Eval;
        Eval.Diag = &Notes;
        if (TA.getAsExpr()->
              EvaluateAsConstantExpr(Eval, Expr::EvaluateForMangling, *Ctx)) {
          TAIsSaved.push_back(TASK_None);
          return;
        }

        QualType FieldTy = TA.getNonTypeTemplateArgumentType();
        auto *Field = FieldDecl::Create(
            *Ctx, RD, SourceLocation(), SourceLocation(), /*Id=*/nullptr,
            FieldTy, Ctx->getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
            /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
        Field->setAccess(AS_public);
        RD->addDecl(Field);

        TAIsSaved.push_back(TASK_Value);
      };

      if (TA.getKind() == TemplateArgument::Pack) {
        for (auto &PTA : TA.getPackAsArray())
          HandleTA(PTA);
        continue;
      }

      HandleTA(TA);
    }

    RD->completeDefinition();
    RD->addAttr(PackedAttr::CreateImplicit(*Ctx));

    const ASTRecordLayout &RLayout = Ctx->getASTRecordLayout(RD);
    assert(Ctx->getCharWidth() == 8 && "char is not 8 bits!");

    QualType RDTy = Ctx->getRecordType(RD);
    auto Fields = cast<RecordDecl>(RDTy->getAsTagDecl())->field_begin();

    SmallVector<TemplateArgument, 8> Builder;

    unsigned TAIdx = 0, TSIdx = 0;
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      auto HandleTA = [&](const TemplateArgument &TA,
                          SmallVector<TemplateArgument, 8> &Builder) {
        if (TAIsSaved[TAIdx] == TASK_Type) {
          PP->ResetForJITTypes();

          PP->setPredefines(TypeStrings[TSIdx]);
          PP->EnterMainSourceFile();

          Parser P(*PP, *S, /*SkipFunctionBodies*/true, /*JITTypes*/true);

	// Reset this to nullptr so that when we call
	// Parser::Initialize it has the clean slate it expects.
          S->CurContext = nullptr;

          P.Initialize();

          Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());

          auto CSFMI = CSFuncMap.find(Idx);
          if (CSFMI != CSFuncMap.end()) {
	  // Note that this restores the context of the function in which the
	  // template was instantiated, but not the state *within* the
	  // function, so local types will remain unavailable.

            auto *FunD = CSFMI->second;
            restoreFuncDeclContext(FunD);
            S->CurContext = S->getContainingDC(FunD);
          }

          TypeResult TSTy = P.ParseTypeName();
          if (TSTy.isInvalid())
            fatal();

          QualType TypeFromString = Sema::GetTypeFromParser(TSTy.get());
          TypeFromString = Ctx->getCanonicalType(TypeFromString);

          Builder.push_back(TemplateArgument(TypeFromString));

          ++TSIdx;
          ++TAIdx;
          return;
        }

        if (TAIsSaved[TAIdx++] != TASK_Value) {
          Builder.push_back(TA);
          return;
        }

        assert(TA.getKind() == TemplateArgument::Expression &&
               "Only expressions template arguments handled here");

        QualType FieldTy = TA.getNonTypeTemplateArgumentType();

        assert(!FieldTy->isMemberPointerType() &&
               "Can't handle member pointers here without ABI knowledge");

        auto *Fld = *Fields++;
        unsigned Offset = RLayout.getFieldOffset(Fld->getFieldIndex()) / 8;
        unsigned Size = Ctx->getTypeSizeInChars(FieldTy).getQuantity();

        unsigned NumIntWords = llvm::alignTo<8>(Size);
        SmallVector<uint64_t, 2> IntWords(NumIntWords, 0);
        std::memcpy((char *) IntWords.data(),
                    ((const char *) NTTPValues) + Offset, Size);
        llvm::APInt IntVal(Size*8, IntWords);

        QualType CanonFieldTy = Ctx->getCanonicalType(FieldTy);

        if (FieldTy->isIntegralOrEnumerationType()) {
          llvm::APSInt SIntVal(IntVal,
                               FieldTy->isUnsignedIntegerOrEnumerationType());
          Builder.push_back(TemplateArgument(*Ctx, SIntVal, CanonFieldTy));
        } else {
          assert(FieldTy->isPointerType() || FieldTy->isReferenceType() ||
                 FieldTy->isNullPtrType());
          if (IntVal.isNullValue()) {
            Builder.push_back(TemplateArgument(CanonFieldTy, /*isNullPtr*/true));
          } else {
	  // Note: We always generate a new global for pointer values here.
	  // This provides a new potential way to introduce an ODR violation:
	  // If you also generate an instantiation using the same pointer value
	  // using some other symbol name, this will generate a different
	  // instantiation.

	  // As we guarantee that the template parameters are not allowed to
	  // point to subobjects, this is useful for optimization because each
	  // of these resolve to distinct underlying objects.

            llvm::SmallString<256> GlobalName("__clang_jit_symbol_");
            IntVal.toString(GlobalName, 16, false);

	  // To this base name we add the mangled type. Stack/heap addresses
	  // can be reused with variables of different type, and these should
	  // have different names even if they share the same address;
            auto &CGM = Consumer->getCodeGenerator()->CGM();
            llvm::raw_svector_ostream MOut(GlobalName);
            CGM.getCXXABI().getMangleContext().mangleTypeName(CanonFieldTy, MOut);

            auto NLDSI = NewLocalSymDecls.find(GlobalName);
            if (NLDSI != NewLocalSymDecls.end()) {
                Builder.push_back(TemplateArgument(NLDSI->second, CanonFieldTy));
            } else {
              Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());
              SourceLocation Loc = FTSI->getPointOfInstantiation();

              QualType STy = CanonFieldTy->getPointeeType();
              auto &II = PP->getIdentifierTable().get(GlobalName);

              if (STy->isFunctionType()) {
                auto *TAFD =
                  FunctionDecl::Create(*Ctx, S->CurContext, Loc, Loc, &II,
                                       STy, /*TInfo=*/nullptr, SC_Extern, false,
                                       STy->isFunctionProtoType());
                TAFD->setImplicit();

                if (const FunctionProtoType *FT = dyn_cast<FunctionProtoType>(STy)) {
                  SmallVector<ParmVarDecl*, 16> Params;
                  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
                    ParmVarDecl *Parm =
                      ParmVarDecl::Create(*Ctx, TAFD, SourceLocation(), SourceLocation(),
                                          nullptr, FT->getParamType(i), /*TInfo=*/nullptr,
                                          SC_None, nullptr);
                    Parm->setScopeInfo(0, i);
                    Params.push_back(Parm);
                  }

                  TAFD->setParams(Params);
                }

                NewLocalSymDecls[II.getName()] = TAFD;
                Builder.push_back(TemplateArgument(TAFD, CanonFieldTy));
              } else {
                auto *TAVD =
                  VarDecl::Create(*Ctx, S->CurContext, Loc, Loc, &II,
                                  STy, Ctx->getTrivialTypeSourceInfo(STy, Loc),
                                  SC_Extern);
                TAVD->setImplicit();

                NewLocalSymDecls[II.getName()] = TAVD;
                Builder.push_back(TemplateArgument(TAVD, CanonFieldTy));
              }

              LocalSymAddrs[II.getName()] = (const void *) IntVal.getZExtValue();
            }
          }
        }
      };

      if (TA.getKind() == TemplateArgument::Pack) {
        SmallVector<TemplateArgument, 8> PBuilder;
        for (auto &PTA : TA.getPackAsArray())
          HandleTA(PTA, PBuilder);
        Builder.push_back(TemplateArgument::CreatePackCopy(*Ctx, PBuilder));
        continue;
      }

      HandleTA(TA, Builder);
    }

    SourceLocation Loc = FTSI->getPointOfInstantiation();
    auto *NewTAL = TemplateArgumentList::CreateCopy(*Ctx, Builder);
    MultiLevelTemplateArgumentList SubstArgs(*NewTAL);

    auto *FunctionTemplate = FTSI->getTemplate();
    DeclContext *Owner = FunctionTemplate->getDeclContext();
    if (FunctionTemplate->getFriendObjectKind())
      Owner = FunctionTemplate->getLexicalDeclContext();

    std::string SMName;
    FunctionTemplateDecl *FTD = FTSI->getTemplate();
    sema::TemplateDeductionInfo Info(Loc);
    {
      Sema::InstantiatingTemplate Inst(
        *S, Loc, FTD, NewTAL->asArray(),
        Sema::CodeSynthesisContext::ExplicitTemplateArgumentSubstitution, Info);
      Sema::ContextRAII TUContext(*S, Ctx->getTranslationUnitDecl());

      auto *Specialization = cast_or_null<FunctionDecl>(
        S->SubstDecl(FunctionTemplate->getTemplatedDecl(), Owner, SubstArgs));
      if (!Specialization || Specialization->isInvalidDecl())
        fatal();

      Specialization->setTemplateSpecializationKind(TSK_ExplicitInstantiationDefinition, Loc);
      S->InstantiateFunctionDefinition(Loc, Specialization, true, true, true);

      SMName = Consumer->getCodeGenerator()->CGM().getMangledName(Specialization);
    }

    if (Diagnostics->hasErrorOccurred())
      fatal();

    // Now we know the name of the symbol, check to see if we already have it.
    if (auto SpecSymbol = CJ->findSymbol(SMName))
      if (SpecSymbol.getAddress())
        return (void *) llvm::cantFail(SpecSymbol.getAddress());

    // There might have been functions/variables with local linkage that were
    // only used by JIT functions. These would not have been used during
    // initial code generation for this translation unit, and so not emitted.
    // We need to make sure that they're emited now (if they're now necessary).

    // Note that we skip having the code generator visiting the decl if it is
    // already defined or already present in our running module. Note that this
    // is not sufficient to prevent all redundant code generation (this might
    // also happen during the instantiation of the top-level function
    // template), and this is why we merge the running module into the new one
    // with the running-module overriding new entities.

    SmallSet<StringRef, 16> LastDeclNames;
    bool Changed;
    do {
      Changed = false;

      Consumer->getCodeGenerator()->CGM().EmitAllDeferred([&](GlobalDecl GD) {
        auto MName = Consumer->getCodeGenerator()->CGM().getMangledName(GD);
        if (!CJ->findSymbol(MName)) {
          Changed = true;
          return false;
        }

        return true;
      });

      SmallSet<StringRef, 16> DeclNames;
      for (auto &F : Consumer->getModule()->functions())
        if (F.isDeclaration() && !F.isIntrinsic())
          if (!LastDeclNames.count(F.getName()))
            DeclNames.insert(F.getName());

      for (auto &GV : Consumer->getModule()->global_values())
        if (GV.isDeclaration())
          if (!LastDeclNames.count(GV.getName()))
            DeclNames.insert(GV.getName());

      for (auto &DeclName : DeclNames) {
        if (CJ->findSymbol(DeclName))
          continue;

        Decl *D = const_cast<Decl *>(Consumer->getCodeGenerator()->
                                       GetDeclForMangledName(DeclName));
        if (!D)
          continue;

        Consumer->HandleInterestingDecl(DeclGroupRef(D));
        LastDeclNames.insert(DeclName);
        Changed = true;
      }
    } while (Changed);

    // Before anything gets optimized, mark the top-level symbol we're
    // generating so that it doesn't get eliminated by the optimizer.

    auto *TopGV =
      cast<GlobalObject>(Consumer->getModule()->getNamedValue(SMName));
    assert(TopGV && "Didn't generate the desired top-level symbol?");

    TopGV->setLinkage(llvm::GlobalValue::ExternalLinkage);
    TopGV->setComdat(nullptr);

    // Finalize the module, generate module-level metadata, etc.

    Consumer->HandleTranslationUnit(*Ctx);

    // First, mark everything we've newly generated with external linkage. When
    // we generate additional modules, we'll mark these functions as available
    // externally, and so we're likely to inline them, but if not, we'll need
    // to link with the ones generated here.

    for (auto &F : Consumer->getModule()->functions()) {
      F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      F.setComdat(nullptr);
    }

    auto IsLocalUnnamedConst = [](llvm::GlobalValue &GV) {
      if (!GV.hasAtLeastLocalUnnamedAddr() || !GV.hasLocalLinkage())
        return false;

      auto *GVar = dyn_cast<llvm::GlobalVariable>(&GV);
      if (!GVar || !GVar->isConstant())
        return false;

      return true;
    };

    for (auto &GV : Consumer->getModule()->global_values()) {
      if (IsLocalUnnamedConst(GV))
        continue;

      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      if (auto *GO = dyn_cast<llvm::GlobalObject>(&GV))
        GO->setComdat(nullptr);
    }

    std::unique_ptr<llvm::Module> ToRunMod =
      llvm::CloneModule(*Consumer->getModule());

    // Here we link our previous cache of definitions, etc. into this module.
    // This includes all of our previously-generated functions (marked as
    // available externally). We prefer our previously-generated versions to
    // our current versions should both modules contain the same entities (as
    // the previously-generated versions have already been optimized).

    // We need to be specifically careful about constants in our module,
    // however. Clang will generate all string literals as .str (plus a
    // number), and these from previously-generated code will conflict with the
    // names chosen for string literals in this module.

    for (auto &GV : ToRunMod->global_values()) {
      if (!IsLocalUnnamedConst(GV))
        continue;

      if (!RunningMod->getNamedValue(GV.getName()))
        continue;

      llvm::SmallString<16> UniqueName(GV.getName());
      unsigned BaseSize = UniqueName.size();
      do {
        // Trim any suffix off and append the next number.
        UniqueName.resize(BaseSize);
        llvm::raw_svector_ostream S(UniqueName);
        S << "." << ++LastUnique;
      } while (RunningMod->getNamedValue(UniqueName));

      GV.setName(UniqueName);
    }

    if (Linker::linkModules(*ToRunMod, llvm::CloneModule(*RunningMod),
                            Linker::Flags::OverrideFromSrc))
      fatal();

    CJ->addModule(std::move(ToRunMod));

    // Now that we've generated code for this module, take them optimized code
    // and mark the definitions as available externally. We'll link them into
    // future modules this way so that they can be inlined.

    for (auto &F : Consumer->getModule()->functions())
      if (!F.isDeclaration())
        F.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

    for (auto &GV : Consumer->getModule()->global_values())
      if (!GV.isDeclaration())
        GV.setLinkage(llvm::GlobalValue::AvailableExternallyLinkage);

    if (Linker::linkModules(*RunningMod, Consumer->takeModule(),
                            Linker::Flags::None))
      fatal();

    Consumer->Initialize(*Ctx);

    auto SpecSymbol = CJ->findSymbol(SMName);
    assert(SpecSymbol && "Can't find the specialization just generated?");

    if (!SpecSymbol.getAddress())
      fatal();

    return (void *) llvm::cantFail(SpecSymbol.getAddress());
  }
};

llvm::sys::SmartMutex<false> Mutex;
bool InitializedTarget = false;
llvm::DenseMap<const void *, std::unique_ptr<CompilerData>> TUCompilerData;

struct InstInfo {
  InstInfo(const char *InstKey, const void *NTTPValues,
           unsigned NTTPValuesSize, const char **TypeStrings,
           unsigned TypeStringsCnt)
    : Key(InstKey),
      NTArgs(StringRef((const char *) NTTPValues, NTTPValuesSize)) {
    for (unsigned i = 0, e = TypeStringsCnt; i != e; ++i)
      TArgs.push_back(StringRef(TypeStrings[i]));
  }

  InstInfo(const StringRef &R) : Key(R) { }

  // The instantiation key (these are always constants, so we don't need to
  // allocate storage for them).
  StringRef Key;

  // The buffer of non-type arguments (this is packed).
  SmallString<16> NTArgs;

  // Vector of string type names.
  SmallVector<SmallString<32>, 1> TArgs;
};

struct ThisInstInfo {
  ThisInstInfo(const char *InstKey, const void *NTTPValues,
               unsigned NTTPValuesSize, const char **TypeStrings,
               unsigned TypeStringsCnt)
    : InstKey(InstKey), NTTPValues(NTTPValues), NTTPValuesSize(NTTPValuesSize),
      TypeStrings(TypeStrings), TypeStringsCnt(TypeStringsCnt) { }

  const char *InstKey;

  const void *NTTPValues;
  unsigned NTTPValuesSize;

  const char **TypeStrings;
  unsigned TypeStringsCnt;
};

struct InstMapInfo {
  static inline InstInfo getEmptyKey() {
    return InstInfo(DenseMapInfo<StringRef>::getEmptyKey());
  }

  static inline InstInfo getTombstoneKey() {
    return InstInfo(DenseMapInfo<StringRef>::getTombstoneKey());
  }

  static unsigned getHashValue(const InstInfo &II) {
    using llvm::hash_code;
    using llvm::hash_combine;
    using llvm::hash_combine_range;

    hash_code h = hash_combine_range(II.Key.begin(), II.Key.end());
    h = hash_combine(h, hash_combine_range(II.NTArgs.begin(),
                                           II.NTArgs.end()));
    for (auto &TA : II.TArgs)
      h = hash_combine(h, hash_combine_range(TA.begin(), TA.end()));

    return (unsigned) h;
  }
  
  static unsigned getHashValue(const ThisInstInfo &TII) {
    using llvm::hash_code;
    using llvm::hash_combine;
    using llvm::hash_combine_range;

    hash_code h =
      hash_combine_range(TII.InstKey, TII.InstKey + std::strlen(TII.InstKey));
    h = hash_combine(h, hash_combine_range((const char *) TII.NTTPValues,
                                           ((const char *) TII.NTTPValues) +
                                             TII.NTTPValuesSize));
    for (unsigned int i = 0, e = TII.TypeStringsCnt; i != e; ++i)
      h = hash_combine(h,
                       hash_combine_range(TII.TypeStrings[i],
                                          TII.TypeStrings[i] +
                                            std::strlen(TII.TypeStrings[i])));

    return (unsigned) h;
  }

  static bool isEqual(const InstInfo &LHS, const InstInfo &RHS) {
    return LHS.Key    == RHS.Key &&
           LHS.NTArgs == RHS.NTArgs &&
           LHS.TArgs  == RHS.TArgs;
  }

  static bool isEqual(const ThisInstInfo &LHS, const InstInfo &RHS) {
    return isEqual(RHS, LHS);
  }

  static bool isEqual(const InstInfo &II, const ThisInstInfo &TII) {
    if (II.Key != StringRef(TII.InstKey))
      return false;
    if (II.NTArgs != StringRef((const char *) TII.NTTPValues,
                               TII.NTTPValuesSize))
      return false;
    if (II.TArgs.size() != TII.TypeStringsCnt)
      return false;
    for (unsigned int i = 0, e = TII.TypeStringsCnt; i != e; ++i)
      if (II.TArgs[i] != StringRef(TII.TypeStrings[i]))
        return false;

    return true; 
  }
};

llvm::sys::SmartMutex<false> IMutex;
llvm::DenseMap<InstInfo, void *, InstMapInfo> Instantiations;

} // anonymous namespace

extern "C"
#ifdef _MSC_VER
__declspec(dllexport)
#endif
void *__clang_jit(const void *CmdArgs, unsigned CmdArgsLen,
                  const void *ASTBuffer, size_t ASTBufferSize,
                  const void *IRBuffer, size_t IRBufferSize,
                  const void **LocalPtrs, unsigned LocalPtrsCnt,
                  const void **LocalDbgPtrs, unsigned LocalDbgPtrsCnt,
                  const DevData *DeviceData, unsigned DevCnt,
                  const void *NTTPValues, unsigned NTTPValuesSize,
                  const char **TypeStrings, unsigned TypeStringsCnt,
                  const char *InstKey, unsigned Idx) {
  {
    llvm::MutexGuard Guard(IMutex);
    auto II =
      Instantiations.find_as(ThisInstInfo(InstKey, NTTPValues, NTTPValuesSize,
                                          TypeStrings, TypeStringsCnt));
    if (II != Instantiations.end())
      return II->second;
  }

  llvm::MutexGuard Guard(Mutex);

  if (!InitializedTarget) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    LCtx.reset(new LLVMContext);

    InitializedTarget = true;
  }

  CompilerData *CD;
  auto TUCDI = TUCompilerData.find(ASTBuffer);
  if (TUCDI == TUCompilerData.end()) {
    CD = new CompilerData(CmdArgs, CmdArgsLen, ASTBuffer, ASTBufferSize,
                          IRBuffer, IRBufferSize, LocalPtrs, LocalPtrsCnt,
                          LocalDbgPtrs, LocalDbgPtrsCnt, DeviceData, DevCnt);
    TUCompilerData[ASTBuffer].reset(CD);
  } else {
    CD = TUCDI->second.get();
  }

  void *FPtr = CD->resolveFunction(NTTPValues, TypeStrings, Idx);

  {
    llvm::MutexGuard Guard(IMutex);
    Instantiations[InstInfo(InstKey, NTTPValues, NTTPValuesSize,
                            TypeStrings, TypeStringsCnt)] = FPtr;
  }

  return FPtr;
}

