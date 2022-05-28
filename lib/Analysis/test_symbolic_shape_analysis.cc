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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace {

struct TestSymbolicShapeAnalysisPass
    : public TestSymbolicShapeAnalysisBase<TestSymbolicShapeAnalysisPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    ModuleOp op = getOperation();
    SymbolicShapeAnalysis(op).dump(llvm::outs());
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createTestSymbolicShapeAnalysisPass() {
  return std::make_unique<TestSymbolicShapeAnalysisPass>();
}

} // namespace mlir
