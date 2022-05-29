add_mlir_library(MLIRHLOAnalysis
  shape_component_analysis.cc
  userange_analysis.cc
  symbolic_shape_analysis.cc

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRIR
)

add_mlir_library(MLIRHLOTestAnalysis
  test_shape_component_analysis.cc
  test_symbolic_shape_analysis.cc
  test_userange_analysis.cc

  DEPENDS
  LMHLOTransformsPassIncGen
  MhloPasses

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  LmhloDialect
  LmhloGPUDialect
  MhloPasses
  MLIRHLOAnalysis
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  )
