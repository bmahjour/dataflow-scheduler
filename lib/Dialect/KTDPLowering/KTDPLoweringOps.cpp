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
// ConstructIndirectAccessTileOp — custom assembly format
//
// Syntax:
//   ktdp_lowering.construct_indirect_access_tile
//       intermediate_variables(%v0, %v1, ...)
//       base_ptr = %iab_mv[%s0, %s1, ...]
//       %base[(%d0), (%d1), ...]
//       {variables_space_set = #set, variables_space_order = #map}
//       : <base-type>, <iab-type> -> <result-type>
//
// Notes:
//   - %s0, %s1, ... may be values drawn from intermediate_variables.
//   - The trailing type list includes both the base memref type and the
//     iab memref type so that the iab type round-trips faithfully.
//===----------------------------------------------------------------------===//

ParseResult ConstructIndirectAccessTileOp::parse(OpAsmParser& parser,
                                                 OperationState& result) {
  auto idx_type = parser.getBuilder().getIndexType();

  // --- intermediate_variables(%v0, ...) ---
  SmallVector<OpAsmParser::UnresolvedOperand> int_var_operands;
  if (parser.parseKeyword("intermediate_variables") ||
      parser.parseOperandList(int_var_operands, OpAsmParser::Delimiter::Paren))
    return failure();

  // --- base_ptr = %iab_mv[%s0, ...] ---
  if (parser.parseKeyword("base_ptr") || parser.parseEqual()) return failure();
  OpAsmParser::UnresolvedOperand iab_operand;
  if (parser.parseOperand(iab_operand)) return failure();
  SmallVector<OpAsmParser::UnresolvedOperand> iab_subscript_operands;
  if (parser.parseOperandList(iab_subscript_operands,
                              OpAsmParser::Delimiter::Square))
    return failure();

  // --- %base[(%d0), (%d1), ...] ---
  OpAsmParser::UnresolvedOperand base_operand;
  if (parser.parseOperand(base_operand)) return failure();
  SmallVector<OpAsmParser::UnresolvedOperand> direct_subscript_operands;
  if (parser.parseLSquare()) return failure();
  if (parser.parseOptionalRSquare()) {
    do {
      OpAsmParser::UnresolvedOperand sub;
      if (parser.parseLParen() || parser.parseOperand(sub) ||
          parser.parseRParen())
        return failure();
      direct_subscript_operands.push_back(sub);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) return failure();
  }

  // --- {attr-dict} ---
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  // --- : <base-type>, <iab-type> -> <result-type> ---
  Type base_type, iab_memref_type, result_type;
  if (parser.parseColon() || parser.parseType(base_type) ||
      parser.parseComma() || parser.parseType(iab_memref_type) ||
      parser.parseArrow() || parser.parseType(result_type))
    return failure();
  result.addTypes(result_type);

  // Resolve operands now that we have all types.
  if (parser.resolveOperand(base_operand, base_type, result.operands) ||
      parser.resolveOperand(iab_operand, iab_memref_type, result.operands) ||
      parser.resolveOperands(iab_subscript_operands, idx_type,
                             result.operands) ||
      parser.resolveOperands(direct_subscript_operands, idx_type,
                             result.operands) ||
      parser.resolveOperands(int_var_operands, idx_type, result.operands))
    return failure();

  // AttrSizedOperandSegments counts, in .td declaration order:
  //   base(1), ind_addr_buf_memref(1), ind_addr_buf_subscripts(N),
  //   direct_subscripts(M), intermediate_variables(K)
  result.addAttribute(
      ConstructIndirectAccessTileOp::getOperandSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, 1, static_cast<int32_t>(iab_subscript_operands.size()),
           static_cast<int32_t>(direct_subscript_operands.size()),
           static_cast<int32_t>(int_var_operands.size())}));

  return success();
}

void ConstructIndirectAccessTileOp::print(OpAsmPrinter& p) {
  // intermediate_variables(...)
  p << " intermediate_variables(";
  llvm::interleaveComma(getIntermediateVariables(), p,
                        [&](Value v) { p.printOperand(v); });
  p << ")";

  // base_ptr = %iab_mv[%s0, ...]
  p << " base_ptr = ";
  p.printOperand(getIndAddrBufMemref());
  p << "[";
  llvm::interleaveComma(getIndAddrBufSubscripts(), p,
                        [&](Value v) { p.printOperand(v); });
  p << "]";

  // %base[(%d0), (%d1), ...]
  p << " ";
  p.printOperand(getBase());
  p << "[";
  llvm::interleaveComma(getDirectSubscripts(), p, [&](Value v) {
    p << "(";
    p.printOperand(v);
    p << ")";
  });
  p << "]";

  // attr-dict (elide the auto-generated segment-size attr)
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{
          ConstructIndirectAccessTileOp::getOperandSegmentSizeAttr()});

  // : <base-type>, <iab-type> -> <result-type>
  p << " : ";
  p.printType(getBase().getType());
  p << ", ";
  p.printType(getIndAddrBufMemref().getType());
  p << " -> ";
  p.printType(getResult().getType());
}

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
