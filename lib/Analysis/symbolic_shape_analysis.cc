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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <numeric>

using namespace llvm;
using namespace mlir;

namespace {

constexpr StringRef getSymbolicShapeFuncAttrName() { return "auxiliary_of"; }

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
    StringAttr insertedSymbol = symbolTable.insert(shpFuncOp);
    insertedSymbols_.push_back(insertedSymbol);
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

    constructSymbolicShapeTable(funcOp, shpFuncOp);
  }

  // run shape reification pass on all the auxilary functions
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
  return operand->get();
}

void SymbolicShapeAnalysis::dump(raw_ostream &os) {
  SymbolTable symbolTable = SymbolTable(moduleOp_);
  for (StringAttr symbol : insertedSymbols_) {
    Operation *op = symbolTable.lookup(symbol);
    assert(op);
    StringAttr originalSymbol =
        op->getAttrOfType<StringAttr>(getSymbolicShapeFuncAttrName());
    auto originalFunc = cast<func::FuncOp>(symbolTable.lookup(originalSymbol));

    os << "============= auxilary shape function for @"
       << originalSymbol.strref() << " =============\n";
    os << *op << "\n\n";

    os << "============= symbolic shape table =============\n";
    for (auto &op : originalFunc.getBody().front().without_terminator()) {
      for (Value v : op.getResults()) {
        os << "original value: " << v << "\n";
        os << "symblic shape: " << getSymbolicShape(v) << "\n";
      }
    }
    os << "\n";
  }
}
