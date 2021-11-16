# Generate the tf_op for GPUs

import tvm
#from tvm.contrib import tf_op
import tvm.topi as topi
import os

lib_path = "tvm_add_op.so"

def export_addfunc_lib():
    n = tvm.te.var("n")
    ph_a = tvm.te.placeholder((n,), name='ph_a')
    ph_b = tvm.te.placeholder((n,), name='ph_b')
    ph_c = tvm.te.compute(ph_a.shape, lambda i: ph_a[i] + ph_b[i], name='ph_c')
#    sched = tvm.te.create_schedule(ph_c.op)

    with tvm.target.cuda() as cuda_tgt:
        sched = topi.cuda.schedule_injective([ph_c])
        # m = tvm.lower(s, [A, B, C], name="test_add")
        m = tvm.lower(sched, [ph_a, ph_b, ph_c], name="vector_add")
        fadd_dylib = tvm.build({"cuda": m}, target_host="llvm")

    fadd_dylib.export_library(lib_path)


import tensorflow as tf
from tvm.contrib import tf_op

# Test for use the tf_op to verifing
def test_tvm_cpu_add_so():
    abs_lib_path = os.path.abspath(lib_path)
    #"/bert/search-team/tvm-tfop/tvm_cpu_add.so"
    module = tf_op.OpModule(abs_lib_path)
    #print("module loaded")
    tvm_add = module.func("vector_add", output_shape=[4], output_dtype="float")

    #print("func loaded")
    #print(tvm_add)
    x = tf.constant([1.0, 2.0, 3.0, 4.0])
    y = tf.constant([1.0, 3.0, 5.0, 7.0])

    #print("before compute")
    r = tvm_add(x, y)
    #print("after compute")
    print(r.numpy())


if __name__ == "__main__":
    export_addfunc_lib()

    #with tf.device('/CPU:0'):
    with tf.device('/GPU:0'):
        test_tvm_cpu_add_so()

