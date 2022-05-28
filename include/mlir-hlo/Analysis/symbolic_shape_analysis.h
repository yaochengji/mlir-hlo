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

#ifndef MLIR_HLO_ANALYSIS_SYMBOLIC_SHAPE_ANALYSIS_H
#define MLIR_HLO_ANALYSIS_SYMBOLIC_SHAPE_ANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

// A auxiliary symbolic shape inference function will be created for each
// original function, and a ShapeReification pass will be run on the newly
// created function. A table mapping the value in the original function to the
// corresponding symbolic shape will also be created for later query and
// analysis.
// Ex. Let's say the original function is as below:
// clang-format off
// func @original_func(%arg0: tensor<?x2xf32>, %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
//   %0 = mhlo.add %arg0, %arg1 : tensor<?x2xf32>
//   return %0 : tensor<?x12xf32>
// }
// Then the created auxilary shape infer will be
// func.func private @_shape_infer_simple(%arg0: tensor<?x4xf32>, %arg1: tensor<?x4xf32>) -> (!shape.shape, tensor<?x4xf32>)  {
//   %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
//   %1 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>  %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
//   return %2, %0 : !shape.shape, tensor<?x4xf32>
// }
// clang-format on
class SymbolicShapeAnalysis {
public:
  SymbolicShapeAnalysis(ModuleOp moduleOp) : moduleOp_(moduleOp) {
    createAuxiliarySymbolicShapeFunc();
  }

  /// Delete all auxiliary function in destructor
  virtual ~SymbolicShapeAnalysis() {
    SymbolTable symbolTable = SymbolTable(moduleOp_);
    for (StringAttr symbol : insertedSymbols_) {
      Operation *op = symbolTable.lookup(symbol);
      assert(op);
      op->erase();
    }
  }

  Value getSymbolicShape(Value v);

  /// Dumps the symbolic shape information to the given stream.
  void dump(raw_ostream &os);

private:
  void createAuxiliarySymbolicShapeFunc();
  void constructSymbolicShapeTable(func::FuncOp originalFunc,
                                   func::FuncOp symbolicShapeInferFunc);

  ModuleOp moduleOp_;
  SmallVector<StringAttr> insertedSymbols_;

  // A table mapping value in original function to symbolic shape in
  // corresponding auxilary shape infer function. The symbolic shape is stored
  // as an pointer to an OpOperand of the terminator in case other intermediate
  // ops be modified after some passes.
  DenseMap<Value, OpOperand *> symbolisShapeTable_;
};

} // namespace mlir

#endif // MLIR_HLO_ANALYSIS_SYMBOLIC_SHAPE_ANALYSIS_H