// [Role] Isolate a tiled output-store path for primitive calibration.
// [Invariants] No gating arithmetic or model geometry; twice a 64x8 seed is stored.
//   Runtime rows/columns and 64x8 tiles retain the compiler's strided-copy path.
// [Timeouts & Locks] Finite caller-supplied dimensions; no block locks.
// [Side-Effect Policy] Writes only the caller's output allocation.
module attributes {hacc.target = #hacc.target<"Ascend950DT_9582">} {
  func.func @strided_store(
      %lock: memref<?xi8>, %workspace: memref<?xi8>,
      %output: memref<?xbf16> {tt.tensor_kind = 1 : i32},
      %input: memref<?xbf16> {tt.tensor_kind = 0 : i32},
      %heads: i32, %tokens: i32, %iterations: i32,
      %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32)
      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64,
                  global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %rows = arith.constant 64 : index
    %cols = arith.constant 8 : index
    %seven = arith.constant 7 : index
    %seed = memref.reinterpret_cast %input to offset: [0],
        sizes: [64, 8], strides: [8, 1] : memref<?xbf16> to memref<64x8xbf16>
    %buffer = memref.alloc() : memref<64x8xbf16>
    memref.copy %seed, %buffer : memref<64x8xbf16> to memref<64x8xbf16>
    %loaded = bufferization.to_tensor %buffer restrict writable : memref<64x8xbf16>
    %wide = arith.extf %loaded : tensor<64x8xbf16> to tensor<64x8xf32>
    %twice = arith.addf %wide, %wide : tensor<64x8xf32>
    %tile = arith.truncf %twice : tensor<64x8xf32> to tensor<64x8xbf16>
    %h = arith.index_cast %heads : i32 to index
    %t = arith.index_cast %tokens : i32 to index
    %it = arith.index_cast %iterations : i32 to index
    %pid = arith.index_cast %px : i32 to index
    %rounded = arith.addi %h, %seven : index
    %col_iterations = arith.divsi %rounded, %cols : index
    %program_base = arith.muli %pid, %it : index
    scf.for %i = %zero to %it step %one {
      %row_index = arith.addi %program_base, %i : index
      %row = arith.muli %row_index, %rows : index
      %remaining_rows = arith.subi %t, %row : index
      %row_limit = arith.minsi %remaining_rows, %rows : index
      %valid_rows = arith.maxsi %row_limit, %zero : index
      %base = arith.muli %row, %h : index
      scf.for %j = %zero to %col_iterations step %one {
        %col = arith.muli %j, %cols : index
        %remaining_cols = arith.subi %h, %col : index
        %valid_cols = arith.minsi %remaining_cols, %cols : index
        %offset = arith.addi %base, %col : index
        %target = memref.reinterpret_cast %output to offset: [%offset],
            sizes: [64, 8], strides: [%h, 1]
            : memref<?xbf16> to memref<64x8xbf16, strided<[?, 1], offset: ?>>
        %selected = tensor.extract_slice %tile[0, 0] [%valid_rows, %valid_cols] [1, 1]
            : tensor<64x8xbf16> to tensor<?x?xbf16>
        %destination = memref.subview %target[0, 0] [%valid_rows, %valid_cols] [1, 1]
            : memref<64x8xbf16, strided<[?, 1], offset: ?>>
            to memref<?x?xbf16, strided<[?, 1], offset: ?>>
        bufferization.materialize_in_destination %selected in writable %destination
            : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, 1], offset: ?>>) -> ()
      }
    }
    return
  }
}
