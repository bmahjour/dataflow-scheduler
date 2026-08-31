//===-- KTDPLoweringOps.cpp -------------------------------------*- C++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ktdp_lowering dialect operations.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
// clang-format on

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

using namespace mlir;
using namespace mlir::ktdp_lowering;

//===----------------------------------------------------------------------===//
// KTDPLoweringDialect — op registration
//===----------------------------------------------------------------------===//

void KTDPLoweringDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstructIndirectAccessTileOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructIndirectAccessTileOp::verify() {
  // ind_addr_buf_subscripts count must equal the rank of ind_addr_buf_memref.
  auto iab_type = mlir::cast<mlir::MemRefType>(getIndAddrBufMemref().getType());
  auto num_subscripts = static_cast<int64_t>(getIndAddrBufSubscripts().size());
  if (num_subscripts != iab_type.getRank())
    return emitOpError() << "ind_addr_buf_subscripts has " << num_subscripts
                         << " operand(s) but ind_addr_buf_memref has rank "
                         << iab_type.getRank() << "; they must be equal";

  return success();
}

//===----------------------------------------------------------------------===//
// ConstructMemoryViewOp — verifier
//===----------------------------------------------------------------------===//

LogicalResult ConstructMemoryViewOp::verify() {
  unsigned nDims = getStaticSizes().size();
  if (getStaticStrides().size() != nDims)
    return emitOpError(
        "static_sizes and static_strides must have equal length");
  unsigned dynSizes = llvm::count(getStaticSizes(), mlir::ShapedType::kDynamic);
  unsigned dynStrides =
      llvm::count(getStaticStrides(), mlir::ShapedType::kDynamic);
  if (getSizes().size() != dynSizes)
    return emitOpError(
        "number of dynamic size operands does not match "
        "kDynamic entries in static_sizes");
  if (getStrides().size() != dynStrides)
    return emitOpError(
        "number of dynamic stride operands does not match "
        "kDynamic entries in static_strides");
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(getResult().getType());
  if (!memrefType) return emitOpError("result must be a memref type");
  if (memrefType.getRank() != static_cast<int64_t>(nDims))
    return emitOpError(
        "result memref rank does not match sizes/strides length");
  return success();
}
