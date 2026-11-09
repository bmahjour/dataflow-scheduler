// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-access-loop-materialization --convert-elementwise-to-linalg --linalg-morph-ops="category-to-generic=true named-to-generic=true" --fuse-linalg --ktir-map-and-tile --ktir-bufferize %s | FileCheck %s

// Verify that an indirect gather with a secondary direct source input
// successfully lowers through loop materialization and KTIRBufferize without
// rank / offset mismatches on ktdp_lowering.load.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:       ktdp_lowering.ind_load
// CHECK:       ktdp_lowering.load
// CHECK-SAME:    memref<2x32x2x64xf16,
// CHECK:       linalg.generic
// CHECK:       ktdp_lowering.store

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_ab_pin   = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab      = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_vars     = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set4_pin = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %c2000 = arith.constant 2000 : index
    %c3000 = arith.constant 3000 : index

    %addr_buf = ktdp.construct_memory_view %c2000,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"}
        : memref<32xindex, "IAB">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        %eq0 = arith.cmpi eq, %i2, %c0 : index
        scf.if %eq0 {
          %addr_at = ktdp.construct_access_tile %addr_buf[%i1, %c0]
              {access_tile_order = #map1, access_tile_set = #set_ab_pin}
              : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
          %addr_stick_2d = ktdp.load %addr_at : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
          %addr_stick = tensor.collapse_shape %addr_stick_2d [[0, 1]]
              : tensor<1x32xindex> into tensor<32xindex>
          %iab_at = ktdp.construct_access_tile %iab_mv[%c0]
              {access_tile_order = affine_map<(d0) -> (d0)>, access_tile_set = #set_iab}
              : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
          ktdp.store %addr_stick, %iab_at : tensor<32xindex>, !ktdp.access_tile<32xindex>
        }

        %desc_1 = ktdp.construct_memory_view %c0,
            sizes: [64, 2, 64], strides: [64, 4096, 1]
            {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
            : memref<64x2x64xf16>

        %tmp1 = ktdp_lowering.construct_indirect_access_tile
            intermediate_variables(%arg2, %arg3)
            base_ptr = %iab_mv[%i2]
            %desc_1[%c0, (%c0 + %arg2), %arg3]
            {variables_space_order = #map2, variables_space_set = #set_vars}
            : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<2x64xindex>
        %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

        %desc_direct = ktdp.construct_memory_view %c3000,
            sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
            {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
            : memref<2x32x2x64xf16>
        %direct_tile = ktdp.construct_access_tile %desc_direct[%i1, %i2, %c0, %c0]
            {access_tile_order = #map, access_tile_set = #set4_pin}
            : memref<2x32x2x64xf16> -> !ktdp.access_tile<1x1x2x64xindex>
        %direct_load = ktdp.load %direct_tile : !ktdp.access_tile<1x1x2x64xindex> -> tensor<1x1x2x64xf16>
        %direct_val = tensor.collapse_shape %direct_load [[0, 1, 2], [3]]
            : tensor<1x1x2x64xf16> into tensor<2x64xf16>

        %empty = tensor.empty() : tensor<2x64xf16>
        %computed = linalg.generic {
            indexing_maps = [#map2, #map2, #map2],
            iterator_types = ["parallel", "parallel"]}
            ins(%tmp1_0, %direct_val : tensor<2x64xf16>, tensor<2x64xf16>)
            outs(%empty : tensor<2x64xf16>) {
        ^bb0(%in_a: f16, %in_b: f16, %out: f16):
          %sum = arith.addf %in_a, %in_b : f16
          linalg.yield %sum : f16
        } -> tensor<2x64xf16>

        %desc_2 = ktdp.construct_memory_view %c1000,
            sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
            {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
            : memref<2x32x2x64xf16>
        %expanded = tensor.expand_shape %computed [[0, 1, 2], [3]] output_shape [1, 1, 2, 64]
            : tensor<2x64xf16> into tensor<1x1x2x64xf16>
        %tmp2 = ktdp.construct_access_tile %desc_2[%i1, %i2, %c0, %c0]
            {access_tile_order = #map, access_tile_set = #set4_pin}
            : memref<2x32x2x64xf16> -> !ktdp.access_tile<1x1x2x64xindex>
        ktdp.store %expanded, %tmp2 : tensor<1x1x2x64xf16>, !ktdp.access_tile<1x1x2x64xindex>
      }
    }
    return
  }
}
