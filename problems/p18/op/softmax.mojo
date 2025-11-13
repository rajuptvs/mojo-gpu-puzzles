from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


alias SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
alias layout = Layout.row_major(SIZE)
alias GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
alias BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: UInt,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    # FILL IN (roughly 31 lines)
    shared_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    shared_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    global_i = thread_idx.x

    # Initialize out-of-bounds (shared_max[local_i], global_i >= input_size) shared memory addresses to the minimum
    # finite value for dtype, ensuring that if these elements are accessed in the parallel max reduction below they
    # do not influence the result (max(min_finite, x) == x for any x).
    var value: Scalar[dtype] = min_finite[dtype]()
    if global_i < input_size:
        value = rebind[Scalar[dtype]](input[global_i])
    shared_max[global_i] = value
    barrier()

    stride = BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_max[global_i] = max(
                shared_max[global_i], shared_max[global_i + stride]
            )

        barrier()
        stride //= 2

    block_max = shared_max[0]  # we do a parallel reduction to get the max

    var exp_value: Scalar[dtype] = 0.0
    if global_i < input_size:
        exp_value = rebind[Scalar[dtype]](exp(value - block_max))

    shared_sum[global_i] = exp_value
    barrier()
    stride = BLOCK_DIM_X // 2
    while stride > 0:
        if global_i < stride:
            shared_sum[global_i] += shared_sum[global_i + stride]

        barrier()
        stride //= 2

    block_sum = shared_sum[0]  # we do a parallel reduction to get the sum
    # so we have exp_value/block_sum which is softmax
    if global_i < input_size:
        output[global_i] = exp_value / block_sum

    # if global_i == 0:
    #     print(dtype)
    #     print(value)


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: UInt,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    var max_val: Scalar[dtype] = min_finite[dtype]()
    for i in range(input_size):
        max_val = max(max_val, rebind[Scalar[dtype]](input[i]))

    var sum_exp: Scalar[dtype] = 0.0
    for i in range(input_size):
        var exp_val = rebind[Scalar[dtype]](exp(input[i] - max_val))
        output[i] = exp_val
        sum_exp += exp_val

    for i in range(input_size):
        output[i] = output[i] / sum_exp


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: UInt,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
