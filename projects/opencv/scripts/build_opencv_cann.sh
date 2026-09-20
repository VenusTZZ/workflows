#!/usr/bin/env bash
# Build opencv + opencv_contrib with WITH_CANN=ON for the runner.
# Idempotent — re-runs are no-ops if the build is already installed
# at /usr/local/opencv-cann.
#
# This is the build half of the setup. The 5 source patches are the
# same ones Quick-start-Ascend.md §"打 5 个源码补丁" applies; see
# that section for the per-patch rationale. We replicate them here so
# the example-guard setup is self-contained.
#
# $1 is the upstream ref (tag/branch/SHA). Defaults to 5.0.0.
#
# Environment: expects CANN env to be sourced (the caller
# setup_example.sh does this) so cmake's WITH_CANN can find the
# toolkit.

set -euo pipefail

UPSTREAM_REF="${1:-5.0.0}"
WORK="${WORK:-/home/coder/work}"
INSTALL_PREFIX=/usr/local/opencv-cann

mkdir -p "$WORK"
cd "$WORK"

# 1) Clone (idempotent) with retry. CI runners hit transient GitHub
#    HTTP 500s on the opencv_contrib clone (observed 2026-09-18 on
#    linux-aarch64-a2-1: "RPC failed; HTTP 500 ... error reading
#    section header 'shallow-info'" left opencv_contrib/ with a broken
#    .git and no working tree, which propagated to the patch script
#    as "FileNotFoundError: opencv_contrib/modules/cannops/src/
#    cann_call.cpp" and then to cmake configure as "Configuring
#    incomplete, errors occurred!" — surfaced too far downstream to
#    diagnose). After each clone, verify a sentinel file exists; if
#    not, nuke .git and retry up to 3 times before giving up.
clone_repo() {
    local url="$1" dir="$2"
    local tries=0 max=3
    while (( tries < max )); do
        tries=$((tries + 1))
        if [[ ! -d "$dir/.git" ]]; then
            echo "build: cloning $url (try $tries/$max)"
            if git clone --depth 1 --branch "$UPSTREAM_REF" "$url" "$dir"; then
                :
            else
                rm -rf "$dir"
            fi
        fi
        # Sentinel files prove the working tree actually arrived; a
        # half-finished --depth 1 clone passes `[[ -d $dir/.git ]]`
        # but is missing everything else.
        local sentinel="$dir/.git/HEAD"
        if [[ -f "$sentinel" ]] && [[ -d "$dir/modules" ]]; then
            echo "build: $dir clone OK (sentinel: $sentinel)"
            return 0
        fi
        echo "build: $dir clone incomplete (sentinel: $sentinel), retrying" >&2
        rm -rf "$dir"
    done
    echo "build: $dir clone FAILED after $max tries" >&2
    return 1
}

clone_repo https://github.com/opencv/opencv.git          opencv
clone_repo https://github.com/opencv/opencv_contrib.git  opencv_contrib

# 2) Apply 5 source patches (each idempotent — checks before patching)
python3 - <<PY
import pathlib
def patch(rel, old, new, *, expect_present=True, count=1):
    p = pathlib.Path(rel)
    if not p.exists():
        if expect_present:
            print(f'(skip) {rel}: file not present')
        return
    s = p.read_text()
    if new in s:
        print(f'(ok)   {rel}: already patched')
        return
    if old not in s:
        if not expect_present:
            return
        raise SystemExit(f'{rel}: pattern not found')
    p.write_text(s.replace(old, new, count))
    print(f'(patch) {rel}: applied')

# (a) CannConstOp ctor type: std::vector<int> -> cv::MatShape
for rel in ['opencv/modules/dnn/src/op_cann.hpp',
            'opencv/modules/dnn/src/op_cann.cpp']:
    patch(rel,
          'const std::vector<int>& shape, const std::string& name',
          'const cv::MatShape& shape, const std::string& name')

# (b) hpp delegating overload declaration
hp = pathlib.Path('opencv/modules/dnn/src/op_cann.hpp')
hs = hp.read_text()
decl_old = 'CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name);'
decl_new = (decl_old
            + '\n        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);')
if decl_new.splitlines()[1] not in hs and decl_old in hs:
    hp.write_text(hs.replace(decl_old, decl_new, 1))
    print('(patch) op_cann.hpp: +1 std::vector<int> overload decl')
elif decl_new.splitlines()[1] in hs:
    print('(ok)   op_cann.hpp: decl already patched')

# (c) cpp delegating ctor
cp = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
cs = cp.read_text()
anchor = 'op_ = std::make_shared<ge::op::Const>(name);\n    op_->set_attr_value(*ge_tensor);\n}\n'
deleg = ('\nCannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)\n'
         '    : CannConstOp(data, dtype, cv::MatShape(shape), name) {}\n')
if deleg.strip() in cs:
    print('(ok)   op_cann.cpp: delegating ctor already patched')
elif anchor in cs:
    cp.write_text(cs.replace(anchor, anchor + deleg, 1))
    print('(patch) op_cann.cpp: +1 delegating ctor')
else:
    raise SystemExit('op_cann.cpp: anchor not found')

# (d) gemm + matmul MatShape assignments
patch('opencv/modules/dnn/src/layers/gemm_layer.cpp',
      '            shape_C = std::vector<int>{dim};',
      '            shape_C = cv::MatShape(1, &dim);')
patch('opencv/modules/dnn/src/layers/matmul_layer.cpp',
      '                if (real_ndims_C == 1 && bias_shape.front() != 1) {',
      '                if (real_ndims_C == 1 && bias_shape[0] != 1) {')
patch('opencv/modules/dnn/src/layers/matmul_layer.cpp',
      '                    bias_shape = std::vector<int>{bias_shape.front()};',
      '                    int _bias_val = bias_shape[0]; bias_shape = cv::MatShape(1, &_bias_val);')

# (e) all_ops.h -> array_ops.h (1500-op header blows aarch64 link)
old_block = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/all_ops.h" // ge::Conv2D, ...
#else
    #include "built-in/op_proto/inc/all_ops.h" // ge::Conv2D, ...
#endif'''
new_block = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#else
    #include "built-in/op_proto/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#endif'''
patch('opencv/modules/dnn/src/op_cann.hpp', old_block, new_block)

# (f) per-layer narrow headers
INSERTS = {
    'layers/batch_norm_layer.cpp': ['nn_batch_norm_ops'],
    'layers/concat_layer.cpp': ['split_combination_ops'],
    'layers/convolution_layer.cpp': ['nn_calculation_ops'],
    'layers/deconvolution_layer.cpp': ['nn_calculation_ops'],
    'layers/depth_space_ops_layer.cpp': ['transformation_ops'],
    'layers/elementwise_layers.cpp': ['nonlinear_fuc_ops', 'elewise_calculation_ops'],
    'layers/eltwise_layer.cpp': ['elewise_calculation_ops'],
    'layers/flatten_layer.cpp': ['transformation_ops'],
    'layers/fully_connected_layer.cpp': ['matrix_calculation_ops'],
    'layers/gemm_layer.cpp': ['matrix_calculation_ops'],
    'layers/instance_norm_layer.cpp': ['nn_norm_ops'],
    'layers/layer_norm.cpp': ['nn_norm_ops'],
    'layers/lrn_layer.cpp': ['nn_norm_ops'],
    'layers/matmul_layer.cpp': ['matrix_calculation_ops'],
    'layers/nary_eltwise_layers.cpp': ['elewise_calculation_ops'],
    'layers/padding_layer.cpp': ['pad_ops'],
    'layers/permute_layer.cpp': ['transformation_ops'],
    'layers/pooling_layer.cpp': ['nn_pooling_ops'],
    'layers/reduce_layer.cpp': ['reduce_ops'],
    'layers/resize2_layer.cpp': ['image_ops'],
    'layers/resize_layer.cpp': ['image_ops'],
    'layers/slice_layer.cpp': ['split_combination_ops', 'selection_ops'],
    'layers/softmax_layer.cpp': ['nn_norm_ops'],
}
ANCHOR = '#include "../op_cann.hpp"'
for rel, headers in INSERTS.items():
    p = pathlib.Path('opencv/modules/dnn/src') / rel
    if not p.exists():
        print(f'(skip) {rel}: not present')
        continue
    s = p.read_text()
    if 'built-in/op_proto/inc/' in s:
        continue
    if ANCHOR not in s:
        print(f'(skip) {rel}: anchor missing (file may not need patch in this tag)')
        continue
    block = ANCHOR + '\n' + '\n'.join(
        f'#include "built-in/op_proto/inc/{h}.h"' for h in headers)
    p.write_text(s.replace(ANCHOR, block, 1))
    print(f'(patch) {rel}: +{len(headers)} narrow header(s)')

# (g) OperatorRunner NULL stream fallback
p = pathlib.Path('opencv_contrib/modules/cannops/src/cann_call.cpp')
s = p.read_text()
old_g = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
new_g = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, execStream));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));
    if (rawStream != nullptr)
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
patch('opencv_contrib/modules/cannops/src/cann_call.cpp', old_g, new_g)

# (h) kernel_launch NULL stream + acl_rt.h include
p = pathlib.Path('opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp')
s = p.read_text()
old_h = '''    std::shared_ptr<uchar> tilingDevice =
        mallocAndUpload(&tiling, sizeof(TILING_TYPE), stream, AscendMat::defaultAllocator());
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(kernel(1, rawStream, tilingDevice.get(), args...));
    if (rawStream == nullptr)
    {
        stream.waitForCompletion();
    }'''
new_h = '''    std::shared_ptr<uchar> tilingDevice =
        mallocAndUpload(&tiling, sizeof(TILING_TYPE), stream, AscendMat::defaultAllocator());
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(kernel(1, execStream, tilingDevice.get(), args...));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));'''
inc_old = '#include <acl/acl_base.h>'
inc_new = '#include <acl/acl_base.h>\n#include <acl/acl_rt.h>'
if 'aclrtCtxGetCurrentDefaultStream' in s:
    print('(ok)   cann_call.hpp: kernel_launch body already patched')
elif old_h in s:
    s = s.replace(old_h, new_h, 1)
    if inc_old in s and inc_new not in s:
        s = s.replace(inc_old, inc_new, 1)
    p.write_text(s)
    print('(patch) cann_call.hpp: NULL-stream + acl_rt.h include')
else:
    print('(skip) cann_call.hpp: kernel_launch body not found (may not need patch)')

# (i) + (j) H2D/D2H host-buffer clamp
p = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
s = p.read_text()
if 'host_size = (size_t)input_wrappers[i]->host->total()' in s:
    print('(ok)   op_cann.cpp: H2D/D2H clamp already patched')
else:
    old_in = '''        auto db = aclmdlGetDatasetBuffer(inputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);

        ACL_CHECK_RET(aclrtMemcpy(p_device, db_size, p_host, db_size, ACL_MEMCPY_HOST_TO_DEVICE));'''
    new_in = '''        auto db = aclmdlGetDatasetBuffer(inputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);
        size_t host_size = (size_t)input_wrappers[i]->host->total() * input_wrappers[i]->host->elemSize();
        size_t copy_size = db_size < host_size ? db_size : host_size;
        ACL_CHECK_RET(aclrtMemcpy(p_device, db_size, p_host, copy_size, ACL_MEMCPY_HOST_TO_DEVICE));'''
    if old_in in s:
        s = s.replace(old_in, new_in, 1)
    old_out = '''        auto db = aclmdlGetDatasetBuffer(outputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);

        ACL_CHECK_RET(aclrtMemcpy(p_host, db_size, p_device, db_size, ACL_MEMCPY_DEVICE_TO_HOST));'''
    new_out = '''        auto db = aclmdlGetDatasetBuffer(outputs, i);
        auto p_device = aclGetDataBufferAddr(db);
        auto db_size = aclGetDataBufferSizeV2(db);
        size_t host_size = (size_t)output_wrappers[i]->host->total() * output_wrappers[i]->host->elemSize();
        size_t copy_size = db_size < host_size ? db_size : host_size;
        CV_LOG_INFO(NULL, "DNN/CANN: output[" << i << "] acl_size=" << db_size << " host_size=" << host_size);
        ACL_CHECK_RET(aclrtMemcpy(p_host, host_size, p_device, copy_size, ACL_MEMCPY_DEVICE_TO_HOST));'''
    if old_out in s:
        s = s.replace(old_out, new_out, 1)
    p.write_text(s)
    print('(patch) op_cann.cpp: H2D/D2H host-buffer clamp applied')
PY

# 3) Symlink CANN libs into the locations OpenCVFindCANN.cmake probes
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux   /usr/local/Ascend/cann-9.1.0/acllib
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/lib64
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/compiler/lib64

# 4) cmake configure + build + install
cd "$WORK/opencv"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
      -DWITH_CANN=ON \
      -DBUILD_opencv_world=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=ON \
      -DOPENCV_BUILD_TEST_MODULES_LIST=cannops \
      -DINSTALL_TESTS=ON \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3,cannops,ts \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR=/usr/local/python3.12.13/include/python3.12 \
      -DPYTHON_LIBRARY=/usr/local/python3.12.13/lib/libpython3.12.so \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      .. > cmake.log 2>&1
CMAKE_RC=$?
if [[ $CMAKE_RC -ne 0 ]]; then
    echo "cmake configure FAILED rc=$CMAKE_RC (tail cmake.log)" >&2
    tail -50 cmake.log >&2
    exit $CMAKE_RC
fi

cmake --build . --target install --parallel 2 > build.log 2>&1
BUILD_RC=$?
if [[ $BUILD_RC -ne 0 ]]; then
    echo "build FAILED rc=$BUILD_RC (tail build.log)" >&2
    tail -100 build.log >&2
    exit $BUILD_RC
fi

echo "build DONE: $($INSTALL_PREFIX/bin/opencv_version)"
