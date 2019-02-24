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

class ClangJIT {
public:
  using ObjLayerT = llvm::orc::LegacyRTDyldObjectLinkingLayer;
  using CompileLayerT = llvm::orc::LegacyIRCompileLayer<ObjLayerT, llvm::orc::SimpleCompiler>;

  ClangJIT()
      : Resolver(createLegacyLookupResolver(
            ES,
            [this](const std::string &Name) {
              return ObjectLayer.findSymbol(Name, true);
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

std::unique_ptr<ClangJIT> CJ;
std::unique_ptr<llvm::LLVMContext> LCtx;

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

  DenseMap<unsigned, FunctionDecl *>      FuncMap;

  CompilerData(const void *CmdArgs, unsigned CmdArgsLen,
               const void *ASTBuffer, size_t ASTBufferSize,
               const void *IRBuffer, size_t IRBufferSize) {
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
    Reader->setReadingForJIT();

    Ctx->setExternalSource(Reader);

    switch (Reader->ReadAST(Filename, serialization::MK_MainFile,
                            SourceLocation(), ASTReader::ARR_OutOfDate)) {
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
    Invocation->getLangOpts()->EmitAllDecls = true;
  }

  void *resolveFunction(const void *NTTPValues, unsigned Idx) {
    FunctionDecl *FD = FuncMap[Idx];
    if (!FD)
      fatal();

    // FIXME: Relying on all of this Clang logic to see if we already have
    // generated this has high overhead. We should record enough identifying
    // information in the structure that hashing it will be all that is
    // necessary to lookup a previously-generated pointer!

    RecordDecl *RD =
      Ctx->buildImplicitRecord(llvm::Twine("__clang_jit_args_")
                               .concat(llvm::Twine(Idx))
                               .concat(llvm::Twine("_t"))
                               .str());

    RD->startDefinition();

    SmallVector<bool, 8> TAIsSaved;

    auto *FTSI = FD->getTemplateSpecializationInfo();
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      if (TA.getKind() != TemplateArgument::Expression) {
        TAIsSaved.push_back(false);
        continue;
      }

      SmallVector<PartialDiagnosticAt, 8> Notes;
      Expr::EvalResult Eval;
      Eval.Diag = &Notes;
      if (TA.getAsExpr()->
            EvaluateAsConstantExpr(Eval, Expr::EvaluateForMangling, *Ctx)) {
        TAIsSaved.push_back(false);
        continue;
      }

      QualType FieldTy = TA.getNonTypeTemplateArgumentType();
      auto *Field = FieldDecl::Create(
          *Ctx, RD, SourceLocation(), SourceLocation(), /*Id=*/nullptr,
          FieldTy, Ctx->getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
          /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
      Field->setAccess(AS_public);
      RD->addDecl(Field);

      TAIsSaved.push_back(true);
    }

    RD->completeDefinition();
    RD->addAttr(PackedAttr::CreateImplicit(*Ctx));

    const ASTRecordLayout &RLayout = Ctx->getASTRecordLayout(RD);
    assert(Ctx->getCharWidth() == 8 && "char is not 8 bits!");

    QualType RDTy = Ctx->getRecordType(RD);
    auto Fields = cast<RecordDecl>(RDTy->getAsTagDecl())->field_begin();

    SmallVector<TemplateArgument, 8> Builder;

    unsigned TAIdx = 0;
    for (auto &TA : FTSI->TemplateArguments->asArray()) {
      if (!TAIsSaved[TAIdx++]) {
        Builder.push_back(TA);
        continue;
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
	  // FIXME: If this is a global that already exists, we should use it
	  // instead of making a new global here.
          // We need to lookup function pointers too.
          assert(true && "Not done yet"); 
          fatal();
        }
      }
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

    Consumer->HandleTranslationUnit(*Ctx);

    // llvm::Function *F = Consumer->getModule()->getFunction(SMName);
    for (auto &F : Consumer->getModule()->functions()) {
      F.setLinkage(llvm::GlobalValue::ExternalLinkage);
      F.setComdat(nullptr);
    }

    for (auto &GV : Consumer->getModule()->global_values()) {
      GV.setLinkage(llvm::GlobalValue::ExternalLinkage);
      if (auto *GO = dyn_cast<llvm::GlobalObject>(&GV))
        GO->setComdat(nullptr);
    }

    std::unique_ptr<llvm::Module> ToRunMod =
      llvm::CloneModule(*Consumer->getModule());

    if (Linker::linkModules(*ToRunMod, llvm::CloneModule(*RunningMod),
                            Linker::Flags::OverrideFromSrc))
      fatal();

    CJ->addModule(std::move(ToRunMod));

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

llvm::sys::SmartMutex<false> IMutex;
std::unordered_map<std::string, void *> Instantiations;

} // anonymous namespace

extern "C"
#ifdef _MSC_VER
__declspec(dllexport)
#endif
void *__clang_jit(const void *CmdArgs, unsigned CmdArgsLen,
                  const void *ASTBuffer, size_t ASTBufferSize,
                  const void *IRBuffer, size_t IRBufferSize,
                  const void *NTTPValues, unsigned NTTPValuesSize,
                  unsigned Idx) {
  // FIXME: Use a DenseSet instead of unordered_map, use a SmallVector to hold data.
  const char *KPtr = ((const char *) ASTBuffer) + Idx;
  std::string Key((const char *) &KPtr, ((const char *) &KPtr) + sizeof(KPtr));
  Key +=std::string((const char *) NTTPValues,
                                   ((const char *) NTTPValues) +
                                     NTTPValuesSize);
  {
    llvm::MutexGuard Guard(IMutex);
    auto II = Instantiations.find(Key);
    if (II != Instantiations.end())
      return II->second;
  }

  llvm::MutexGuard Guard(Mutex);

  if (!InitializedTarget) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    LCtx.reset(new LLVMContext);
    CJ = llvm::make_unique<ClangJIT>();

    InitializedTarget = true;
  }

  CompilerData *CD;
  auto TUCDI = TUCompilerData.find(ASTBuffer);
  if (TUCDI == TUCompilerData.end()) {
    CD = new CompilerData(CmdArgs, CmdArgsLen, ASTBuffer, ASTBufferSize,
                          IRBuffer, IRBufferSize);
    TUCompilerData[ASTBuffer].reset(CD);
  } else {
    CD = TUCDI->second.get();
  }

  void *FPtr = CD->resolveFunction(NTTPValues, Idx);

  {
    llvm::MutexGuard Guard(IMutex);
    Instantiations[Key] = FPtr;
  }

  return FPtr;
}

