// RUN: dataflow-scheduler-opt --ktir-bufferize %s | FileCheck %s

// The indirect tile keeps its result type: only its base memory view is cast
// to the mapped space and reinterpreted over its full shape, with the view's
// strides. The tiles feeding the IAB fill go through ktdp.load / ktdp.store,
// so they are left alone.

// CHECK-LABEL: func.func @gather
// CHECK:         %[[IAB:.*]] = ktdp_lowering.construct_memory_view {{.*}} : memref<32xindex, "IBR">
// CHECK:         scf.if
// CHECK:           ktdp.construct_access_tile
// CHECK:           ktdp.load
// CHECK:           ktdp.construct_access_tile %[[IAB]]
// CHECK:           ktdp.store
// CHECK:         %[[VIEW:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x2x64xf16>
// CHECK:         %[[MSC:.*]] = memref.memory_space_cast %[[VIEW]] : memref<64x2x64xf16> to memref<64x2x64xf16, "DDR">
// CHECK:         %[[RC:.*]] = memref.reinterpret_cast %[[MSC]] to offset: [%{{.*}}], sizes: [64, 2, 64], strides: [64, 4096, 1] : memref<64x2x64xf16, "DDR"> to memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">
// CHECK:         %[[TILE:.*]] = ktdp_lowering.construct_indirect_access_tile intermediate_variables(%{{.*}}) base_ptr = %[[IAB]][%{{.*}}] %[[RC]][0, %{{.*}} + %{{.*}}, %{{.*}}] {{.*}} : memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, memref<32xindex, "IBR"> -> !ktdp.access_tile<64xindex>
// CHECK:         ktdp_lowering.load %[[TILE]]
// CHECK-NOT:     memref.reinterpret_cast

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @gather() -> tensor<64xf16> attributes {grid = [1]} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1] {coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>, memory_space = "IBR"} : memref<32xindex, "IBR">
    %1 = ktdp.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<2x32xi32, #ktdp.memory_space<global>>
    %r = scf.for %arg1 = %c0 to %c32 step %c1 iter_args(%acc = %c0) -> (index) {
      %2 = arith.cmpi eq, %arg1, %c0 : index
      scf.if %2 {
        %8 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 31 >= 0)>} : memref<2x32xi32, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %9 = ktdp.load %8 : <1x32xindex> -> tensor<1x32xindex>
        %collapsed = tensor.collapse_shape %9 [[0, 1]] : tensor<1x32xindex> into tensor<32xindex>
        %10 = ktdp.construct_access_tile %0[%c0] {access_tile_order = affine_map<(d0) -> (d0)>, access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>} : memref<32xindex, "IBR"> -> !ktdp.access_tile<32xindex>
        ktdp.store %collapsed, %10 : tensor<32xindex>, <32xindex>
      }
      scf.yield %acc : index
    }
    %3 = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1] {coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<64x2x64xf16>
    %4 = ktdp_lowering.construct_indirect_access_tile intermediate_variables(%arg3) base_ptr = %0[%r] %3[0, %c0 + %c1, %arg3] {variables_space_order = affine_map<(d0) -> (d0)>, variables_space_set = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>} : memref<64x2x64xf16>, memref<32xindex, "IBR"> -> !ktdp.access_tile<64xindex>
    %5 = ktdp_lowering.load %4[%c0] [64] [1] : !ktdp.access_tile<64xindex> -> tensor<64xf16>
    return %5 : tensor<64xf16>
  }
}
