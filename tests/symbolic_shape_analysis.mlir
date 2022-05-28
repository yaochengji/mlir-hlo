// RUN: mlir-hlo-opt --test-print-symbolic-shape --split-input-file %s | FileCheck %s

func.func @simple(%arg0 : tensor<?x4xf32>, %arg1 : tensor<?x4xf32>) -> tensor<?x4xf32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}
// CHECK-LABEL: ============= auxilary shape function for @simple =============
// CHECK-NEXT: func private @_shape_infer_simple
// CHECK-NEXT:   %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
// CHECK-NEXT:   %1 = shape.shape_of %arg0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-NEXT:   %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape
// CHECK-NEXT:   return %2, %0 : !shape.shape, tensor<?x4xf32>

// CHECK-LABEL: ============= symbolic shape table =============
// CHECK-NEXT: original value: %0 = mhlo.add %arg0, %arg1 : tensor<?x4xf32>
// CHECK-NEXT: symblic shape: %2 = shape.value_as_shape %1 : tensor<2xindex> -> !shape.shape

func.func @several_ops(%arg0: tensor<?x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >,
   precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
  %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
  %3 = mhlo.add %0, %2 : tensor<?x4xf32>
  return %3 : tensor<?x4xf32>
}
// CHECK-LABEL: ============= auxilary shape function for @several_ops =============
// CHECK-NEXT: func private @_shape_infer_several_ops
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[V0:.+]] = shape.const_shape [2] : tensor<1xindex>
// CHECK-DAG:   %[[V2:.+]] = tensor.dim %arg0, %[[C0]] : tensor<?x4xf32>
// CHECK-DAG:   %[[V3:.+]] = tensor.from_elements %[[V2]], %[[C4]] : tensor<2xindex>  
// CHECK-DAG:   %[[V4:.+]] = shape.value_as_shape %[[V3]] : tensor<2xindex> -> !shape.shape
// CHECK-DAG:   %[[V5:.+]] = shape.value_as_shape %[[V0]] : tensor<1xindex> -> !shape.shape
// CHECK-DAG:   return %[[V4]], %[[V5]], %[[V4]], %[[V4]],

// CHECK-LABEL: ============= symbolic shape table =============
// CHECK-NEXT: original value: %0 = "mhlo.dot_general"(%arg0, %arg1)
// CHECK-NEXT: symblic shape: %4 = shape.value_as_shape %3 : tensor<2xindex> -> !shape.shape
// CHECK-NEXT: original value: %1 = shape.shape_of %0 : tensor<?x4xf32> -> tensor<2xindex>
// CHECK-NEXT: symblic shape: %5 = shape.value_as_shape %0 : tensor<1xindex> -> !shape.shape
// CHECK-NEXT: original value: %2 = "mhlo.dynamic_broadcast_in_dim"(%arg2, %1) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<4xf32>, tensor<2xindex>) -> tensor<?x4xf32>
// CHECK-NEXT: symblic shape: %4 = shape.value_as_shape %3 : tensor<2xindex> -> !shape.shape
// CHECK-NEXT: original value: %3 = mhlo.add %0, %2 : tensor<?x4xf32>
// CHECK-NEXT: symblic shape: %4 = shape.value_as_shape %3 : tensor<2xindex> -> !shape.shape
