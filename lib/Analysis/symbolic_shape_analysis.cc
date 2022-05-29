/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mlir-hlo/Analysis/symbolic_shape_analysis.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <numeric>

using namespace llvm;
using namespace mlir;

#define DEBUG_TYPE "symbolic-shape-analysis"

namespace {

constexpr StringRef getSymbolicShapeFuncAttrName() { return "auxiliary_of"; }

Value getArgIfIsAValueAsShapeOp(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (auto valueAsShapeOp = dyn_cast_or_null<shape::ValueAsShapeOp>(defOp)) {
    return valueAsShapeOp.getArg();
  }
  return v;
}

} // namespace

void SymbolicShapeAnalysis::constructSymbolicShapeTable(
    func::FuncOp originalFunc, func::FuncOp symbolicShapeInferFunc) {
  Operation *terminator =
      symbolicShapeInferFunc.getBody().front().getTerminator();
  int cnt = 0;
  for (auto &op : originalFunc.getBody().front().without_terminator()) {
    for (Value v : op.getResults()) {
      symbolisShapeTable_[v] = &terminator->getOpOperand(cnt);
      cnt++;
    }
  }
}

void SymbolicShapeAnalysis::createAuxiliarySymbolicShapeFunc() {
  SmallVector<func::FuncOp> funcOps;
  for (auto funcOp : moduleOp_.getOps<func::FuncOp>()) {
    funcOps.push_back(funcOp);
  }

  SymbolTable symbolTable = SymbolTable(moduleOp_);
  SmallVector<func::FuncOp> shpFuncOps;

  for (auto funcOp : funcOps) {
    StringRef funcSymName = funcOp.getSymName();
    std::string shpFuncSymName = "_shape_infer_" + funcSymName.str();

    OpBuilder builder(funcOp);

    // Create the auxiliary shape infer func signature. The function's return
    // types will be an aggregation of all the body ops's result types and
    // corresponding shape.shape type
    size_t numResults = 0;
    for (auto &op : funcOp.getBody().front().without_terminator()) {
      numResults += op.getNumResults();
    }
    SmallVector<Type> allResultTypes(numResults,
                                     builder.getType<shape::ShapeType>());
    for (auto &op : funcOp.getBody().front().without_terminator()) {
      allResultTypes.insert(allResultTypes.end(), op.getResultTypes().begin(),
                            op.getResultTypes().end());
    }

    auto shpFnType =
        builder.getFunctionType(funcOp.getArgumentTypes(), allResultTypes);
    func::FuncOp shpFuncOp = builder.create<func::FuncOp>(
        funcOp->getLoc(), shpFuncSymName, shpFnType);
    shpFuncOp.setPrivate();
    shpFuncOp->setAttr(getSymbolicShapeFuncAttrName(),
                       builder.getStringAttr(funcSymName));
    symbolTable.insert(shpFuncOp);
    shpFuncOps.push_back(shpFuncOp);

    // add the body of the auxiliary shape infer func
    Block *block = shpFuncOp.addEntryBlock();
    builder.setInsertionPointToStart(block);
    BlockAndValueMapping bvm;
    SmallVector<Value> valResults;
    SmallVector<Value> allResults;
    for (auto it : zip(funcOp.getArguments(), shpFuncOp.getArguments())) {
      bvm.map(std::get<0>(it), std::get<1>(it));
    }

    for (auto &op : funcOp.getBody().front().without_terminator()) {
      Operation *opInShpFn = builder.clone(op, bvm);
      valResults.insert(valResults.end(), opInShpFn->getResults().begin(),
                        opInShpFn->getResults().end());
      for (Value opInShpFnRes : opInShpFn->getResults()) {
        Value shapeOfRes =
            builder.create<shape::ShapeOfOp>(opInShpFn->getLoc(), opInShpFnRes);
        auto shapeTypeRes = builder.create<shape::ValueAsShapeOp>(
            opInShpFn->getLoc(), builder.getType<shape::ShapeType>(),
            shapeOfRes);
        allResults.push_back(shapeTypeRes);
      }
    }

    allResults.append(valResults);
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), allResults);

    originalFuncToAuxiliary_.push_back({funcOp, shpFuncOp});
    constructSymbolicShapeTable(funcOp, shpFuncOp);
  }

  // run shape reification pass on all the auxiliary functions
  PassManager pm(moduleOp_->getContext(), func::FuncOp::getOperationName());
  pm.addPass(mhlo::CreateShapeReificationPass());
  pm.addPass(createCSEPass());
  for (auto funcOp : shpFuncOps) {
    if (mlir::failed(pm.run(funcOp))) {
      llvm::errs() << "Pass pipeline inside symbolic shape analysis failed.";
    }
  }
}

Value SymbolicShapeAnalysis::getSymbolicShape(Value v) {
  auto iter = symbolisShapeTable_.find(v);
  if (iter == symbolisShapeTable_.end()) {
    errs() << "Input is not a valid Value in the original functions, get: " << v
           << "\n";
    return {};
  }
  OpOperand *operand = iter->second;
  return getArgIfIsAValueAsShapeOp(operand->get());
}

llvm::DenseSet<Value> SymbolicShapeAnalysis::findSymbolicExprSourcesRecursively(
    Value symbolicShape,
    llvm::DenseMap<Value, llvm::DenseSet<Value>> &symbolicShapeFnCache,
    const llvm::DenseMap<Value, Value> &auxiValToOrigin) {
  auto resIter = symbolicShapeFnCache.find(symbolicShape);
  if (resIter != symbolicShapeFnCache.end())
    return resIter->second;

  llvm::DenseSet<Value> result;

  Operation *defOp = symbolicShape.getDefiningOp();
  // FIXME: is checking tensor.dim and shape.shape_of here correct and enough?
  if (!defOp) {
    if (auto arg = auxiValToOrigin.lookup(symbolicShape))
      result.insert(arg);
  } else if (auto dimOp = dyn_cast<tensor::DimOp>(defOp)) {
    if (auto originVal = auxiValToOrigin.lookup(dimOp.source()))
      result.insert(originVal);
  } else if (auto shapeOfOp = dyn_cast<shape::ShapeOfOp>(defOp)) {
    if (auto originVal = auxiValToOrigin.lookup(shapeOfOp.getArg()))
      result.insert(originVal);
  } else {
    for (Value inp : defOp->getOperands()) {
      llvm::DenseSet<Value> subSources = findSymbolicExprSourcesRecursively(
          inp, symbolicShapeFnCache, auxiValToOrigin);
      for (Value sV : subSources)
        result.insert(sV);
    }
  }
  symbolicShapeFnCache[symbolicShape] = result;

  // clang-format off
  LLVM_DEBUG(llvm::dbgs() << "intermidiate result in symbolicShapeFnCache: "
                          << symbolicShape << "\n";
             for (Value r : result)
               llvm::dbgs() << r << "\n";);
  // clang-format on

  return result;
}

llvm::DenseMap<Value, llvm::DenseSet<Value>>
SymbolicShapeAnalysis::constructSymbolicExprSourcesTable() {
  llvm::DenseMap<Value, llvm::DenseSet<Value>> result;

  for (auto it : originalFuncToAuxiliary_) {
    func::FuncOp originalFunc = it.first;
    func::FuncOp auxiliaryFunc = it.second;
    llvm::DenseMap<Value, Value> auxiValToOrigin;
    llvm::DenseMap<Value, llvm::DenseSet<Value>> symbolicShapeFnCache;

    unsigned valCnt = 0;
    Operation *auxiTerminator = auxiliaryFunc.getBody().front().getTerminator();
    unsigned numOfAuxiliaryResult = auxiTerminator->getNumOperands();
    assert(numOfAuxiliaryResult % 2 == 0);
    unsigned halfOfNumAuxiliaryResult = numOfAuxiliaryResult / 2;
    for (auto &op : originalFunc.getBody().front().without_terminator()) {
      for (Value v : op.getResults()) {
        Value auxiVal =
            auxiTerminator->getOperand(halfOfNumAuxiliaryResult + valCnt);
        valCnt++;
        auxiValToOrigin[auxiVal] = v;
      }
    }

    for (auto it :
         zip(originalFunc.getArguments(), auxiliaryFunc.getArguments())) {
      auxiValToOrigin[std::get<1>(it)] = std::get<0>(it);
    }

    for (auto &op : originalFunc.getBody().front().without_terminator()) {
      for (Value v : op.getResults()) {
        Value symbolicShape = getSymbolicShape(v);
        llvm::DenseSet<Value> sources = findSymbolicExprSourcesRecursively(
            symbolicShape, symbolicShapeFnCache, auxiValToOrigin);
        result[v] = sources;
      }
    }
  }
  return result;
}

void SymbolicShapeAnalysis::dump(raw_ostream &os) {
  SymbolTable symbolTable = SymbolTable(moduleOp_);
  llvm::DenseMap<Value, llvm::DenseSet<Value>> sourcesTable =
      constructSymbolicExprSourcesTable();
  for (auto it : originalFuncToAuxiliary_) {
    func::FuncOp originalFunc = it.first;
    func::FuncOp auxiliaryFunc = it.second;

    os << "============= auxiliary shape function for @"
       << originalFunc.getSymName() << " =============\n";
    os << auxiliaryFunc << "\n\n";

    os << "============= symbolic shape table for @"
       << originalFunc.getSymName() << " =============\n";
    for (auto &op : originalFunc.getBody().front().without_terminator()) {
      for (Value v : op.getResults()) {
        os << "original value: " << v << "\n";
        os << "symblic shape: " << getSymbolicShape(v) << "\n";
        os << "\n";
      }
    }
    os << "\n";

    os << "============= symbolic expr sources table for @"
       << originalFunc.getSymName() << " =============\n";
    for (auto &op : originalFunc.getBody().front().without_terminator()) {
      for (Value v : op.getResults()) {
        os << "original value: " << v << "\n";
        os << "symblic shape sources: \n";
        for (Value source : sourcesTable[v]) {
          os << source << "\n";
        }
        os << "\n";
      }
    }
    os << "\n";
  }
}
