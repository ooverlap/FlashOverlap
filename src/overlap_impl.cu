#include "nccl_utils.h"
#include "wait.cuh"

#include "tiling/gemm_tiling.cuh"
#include "tiling/signal_tiling.cuh"
#include "tiling/scatter_tiling.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include "overlap_impl.h"

#include "ooverlap/comm.h"

#include <vector>
#include <stdexcept>
#include <cstdint>

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MAX_GROUP_SIZE 64

namespace {

const char* oo_status_to_string(oo_status_t status) {
    switch (status) {
        case OO_SUCCESS:
            return "OO_SUCCESS";
        case OO_ERROR_INVALID_ARGUMENT:
            return "OO_ERROR_INVALID_ARGUMENT";
        case OO_ERROR_INVALID_DEVICE:
            return "OO_ERROR_INVALID_DEVICE";
        case OO_ERROR_UNSUPPORTED:
            return "OO_ERROR_UNSUPPORTED";
        case OO_ERROR_CUDA:
            return "OO_ERROR_CUDA";
        case OO_ERROR_INTERNAL:
            return "OO_ERROR_INTERNAL";
        default:
            return "OO_ERROR_UNKNOWN";
    }
}

void OO_CHECK_OR_THROW(oo_status_t status, const char* what) {
    TORCH_CHECK(
        status == OO_SUCCESS,
        what,
        " failed with ",
        oo_status_to_string(status));
}

void CUDA_CHECK_OR_THROW(cudaError_t err, const char* what) {
    TORCH_CHECK(
        err == cudaSuccess,
        what,
        " failed with ",
        cudaGetErrorString(err));
}

} // namespace

/// NIL Implementation: Overlap CUTLASS GEMM and NCCL AllReduce
OverlapImpl::OverlapImpl() {
    this->gemm_stream = nullptr;
    this->comm_stream = nullptr;
    this->gemm_finished = nullptr;

    this->comm = nullptr;
    this->my_rank = -1;
    this->my_size = 0;
}

OverlapImpl::~OverlapImpl() {
    this->OoverlapRelease();
}

/*
 * OOverlap stuff
 */
void OverlapImpl::OoverlapIpcInit(
        const int64_t tp_rank,
        const int64_t tp_size,
        const std::vector<int64_t> devices,
        const std::string broker_key) {

    TORCH_CHECK(tp_size == 2, "ooverlap IPC allreduce currently supports exactly 2 ranks");
    TORCH_CHECK(devices.size() == 2, "devices must contain exactly two CUDA device ids");
    TORCH_CHECK(tp_rank == 0 || tp_rank == 1, "tp_rank must be 0 or 1 for ooverlap IPC");
    TORCH_CHECK(!broker_key.empty(), "broker_key must be non-empty");

    this->OoverlapRelease();

    this->oo_rank = tp_rank;
    this->oo_size = tp_size;
    this->oo_devices[0] = static_cast<int>(devices[0]);
    this->oo_devices[1] = static_cast<int>(devices[1]);

    const int local_device = this->oo_devices[this->oo_rank];
    CUDA_CHECK_OR_THROW(cudaSetDevice(local_device), "cudaSetDevice(ooverlap local device)");

    int devs[2] = {this->oo_devices[0], this->oo_devices[1]};

    oo_group_t* new_group = nullptr;
    oo_node_t* new_node = nullptr;

    oo_status_t st = oo_group_create_ipc(
        devs,
        2,
        static_cast<int>(this->oo_rank),
        broker_key.c_str(),
        &new_group);

    OO_CHECK_OR_THROW(st, "oo_group_create_ipc");

    st = oo_node_create(
        new_group,
        static_cast<int>(this->oo_rank),
        &new_node);

    if (st != OO_SUCCESS) {
        oo_group_destroy(new_group);
        OO_CHECK_OR_THROW(st, "oo_node_create");
    }

    this->oo_group = new_group;
    this->oo_node = new_node;
    this->oo_initialized = true;
}

void OverlapImpl::OoverlapUnregisterBuffer() {
    if (this->oo_group != nullptr) {
        try {
            oo_group_sync(this->oo_group);
        } catch (...) {
        }
    }

    if (this->oo_peer_buf != nullptr) {
        oo_buffer_destroy(this->oo_peer_buf);
        this->oo_peer_buf = nullptr;
    }

    if (this->oo_group != nullptr) {
        try {
            oo_group_sync(this->oo_group);
        } catch (...) {
        }
    }

    if (this->oo_local_buf != nullptr) {
        oo_buffer_destroy(this->oo_local_buf);
        this->oo_local_buf = nullptr;
    }

    if (this->oo_group != nullptr) {
        try {
            oo_group_sync(this->oo_group);
        } catch (...) {
        }
    }

    this->oo_registered_ptr = nullptr;
    this->oo_registered_bytes = 0;
}

void OverlapImpl::OoverlapRelease() {
    this->OoverlapUnregisterBuffer();

    if (this->oo_node != nullptr) {
        oo_node_destroy(this->oo_node);
        this->oo_node = nullptr;
    }

    if (this->oo_group != nullptr) {
        oo_group_destroy(this->oo_group);
        this->oo_group = nullptr;
    }

    this->oo_rank = -1;
    this->oo_size = 0;
    this->oo_devices[0] = -1;
    this->oo_devices[1] = -1;
    this->oo_initialized = false;
}

void OverlapImpl::OoverlapEnsureBuffer(at::Tensor C) {
    TORCH_CHECK(this->oo_initialized, "Call ooverlap_ipc_init before gemm_allreduce_overlap");
    TORCH_CHECK(this->oo_group != nullptr, "ooverlap group is null");
    TORCH_CHECK(this->oo_node != nullptr, "ooverlap node is null");

    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous for ooverlap IPC wrapping");
    TORCH_CHECK(C.scalar_type() == at::kHalf, "Only torch.float16 C is supported for now");

    const int expected_device = this->oo_devices[this->oo_rank];
    TORCH_CHECK(
        C.get_device() == expected_device,
        "C is on CUDA device ",
        C.get_device(),
        " but ooverlap local rank expects device ",
        expected_device);

    void* c_ptr = C.data_ptr<at::Half>();
    const size_t bytes = static_cast<size_t>(C.numel()) * sizeof(half);

    TORCH_CHECK(bytes > 0, "C must be non-empty");

    if (this->oo_registered_ptr == c_ptr &&
        this->oo_registered_bytes == bytes &&
        this->oo_local_buf != nullptr &&
        this->oo_peer_buf != nullptr) {
        return;
    }

    this->OoverlapUnregisterBuffer();

    CUDA_CHECK_OR_THROW(cudaSetDevice(expected_device), "cudaSetDevice(before oo_buffer_wrap)");

    OO_CHECK_OR_THROW(
        oo_buffer_wrap(
            this->oo_node,
            c_ptr,
            bytes,
            &this->oo_local_buf),
        "oo_buffer_wrap(C)");

    OO_CHECK_OR_THROW(
        oo_buffer_exchange_ipc_peer(
            this->oo_node,
            this->oo_local_buf,
            &this->oo_peer_buf),
        "oo_buffer_exchange_ipc_peer(C)"); 

    this->oo_registered_ptr = c_ptr;
    this->oo_registered_bytes = bytes;
}

void OverlapImpl::OoverlapAllReduceSlice(
        size_t element_offset,
        size_t count,
        cudaStream_t stream) {

    if (count == 0) {
        return;
    }

    TORCH_CHECK(this->oo_local_buf != nullptr, "ooverlap local buffer is not registered");
    TORCH_CHECK(this->oo_peer_buf != nullptr, "ooverlap peer buffer is not registered");

    const size_t byte_offset = element_offset * sizeof(half);
    const size_t bytes = count * sizeof(half);

    TORCH_CHECK(
        byte_offset + bytes <= this->oo_registered_bytes,
        "ooverlap allreduce slice is out of registered C buffer bounds");

    OO_CHECK_OR_THROW(
        oo_allreduce_offset(
            this->oo_node,
            this->oo_local_buf,
            this->oo_peer_buf,
            element_offset,
            count,
            OO_DTYPE_FLOAT16,
            OO_REDUCE_SUM,
            stream),
        "oo_allreduce_offset(slice)");
}



void OverlapImpl::CutlassInit(){
    this->gemm_stream = at::cuda::getCurrentCUDAStream().stream();
}

void OverlapImpl::Gemm(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );
}

void OverlapImpl::NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id){

    this->my_rank = tp_rank;
    this->my_size = tp_size;

    ncclUniqueId tp_uid;
    memcpy(tp_uid.internal, &tp_id[0], NCCL_UNIQUE_ID_BYTES);

    if (this->my_size == 1) {
        this->comm = nullptr;
        return;
    }
    NCCL_CHECK(ncclCommInitRank(&this->comm, this->my_size, tp_uid, this->my_rank));
}

void OverlapImpl::GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    NCCL_CHECK(ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::GemmReduceScatter(
        at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)d_ptr, recvcount, 
        ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, 
    int64_t Algo, at::Tensor mLen_CPU){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());
    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();

    gemm_func_table[Algo](
        M, N, K, a_ptr, b_ptr, c_ptr, this->gemm_stream
    );

    // Launch All2All after GEMM
    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->gemm_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->gemm_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}

void OverlapImpl::OverlapInit(){
    cudaStreamCreateWithPriority(&this->comm_stream, cudaStreamNonBlocking, -5);
}

void OverlapImpl::NcclAllReduce(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    ncclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), ncclFloat16, ncclSum, this->comm, this->gemm_stream);
}

void OverlapImpl::SegAllReduce(at::Tensor C, at::Tensor cSEG_CPU, int64_t SegNum){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();

    int acc_addr = 0;
    for (int s = 0; s < SegNum; s++){
        int commSize = M * N / SegNum * cseg_cpu_ptr[s];
        NCCL_CHECK(ncclAllReduce((void *)(c_ptr + acc_addr), (void *)(c_ptr + acc_addr), commSize, ncclFloat16, ncclSum, this->comm, this->gemm_stream));
        acc_addr += commSize;
    }
}

void OverlapImpl::NcclReduceScatter(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    size_t recvcount = (M * N) / this->my_size;
    NCCL_CHECK(ncclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        ncclFloat16, ncclSum, this->comm, this->gemm_stream));
}

void OverlapImpl::NcclAll2All(at::Tensor C, 
    at::Tensor D, // [world_size - 1, M, N]
    at::Tensor mLen_CPU // [world_size, world_size]
    ){
    
    int M = C.size(0);
    int N = C.size(1);

    assert(mLen_CPU.size(0) == this->my_size);
    assert(mLen_CPU.size(1) == this->my_size);

    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        NCCL_CHECK(ncclSend((void *)(c_ptr + src_acc_addr), sendcount, ncclFloat16, i, this->comm, this->gemm_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        NCCL_CHECK(ncclRecv((void *)(d_ptr + dst_acc_addr), recvcount, ncclFloat16, i, this->comm, this->gemm_stream));
        dst_acc_addr += recvcount;
    }
    NCCL_CHECK(ncclGroupEnd());
}

void OverlapImpl::GemmAllReduceOverlap(
        at::Tensor A,  // M, K
        at::Tensor B,  // N, K
        at::Tensor C,  // M, N
        at::Tensor MM, // TM + 1, TN
        at::Tensor RA, // TM, TN
        int64_t rLDN, 
        at::Tensor cSEG_CPU, // SegSize, how many communication segments
        at::Tensor cSEG_GPU, // SegSize, how many communication segments, on GPU
        int64_t Algo,
        bool if_monitor
        ){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    int TM = RA.size(0);
    int TN = RA.size(1);
    int TileNum = TM * TN;

    int SegSize = cSEG_GPU.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    int* mm_ptr = MM.data_ptr<int>();
    int* ra_ptr = RA.data_ptr<int>();

    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    int* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    const bool use_ooverlap = this->oo_initialized;
    if (use_ooverlap) {
        this->OoverlapEnsureBuffer(C);
    }

    int acc_addr = 0;
    signal_func_table[Algo](
        M, N, K, rLDN, cseg_gpu_ptr, a_ptr, b_ptr, c_ptr, mm_ptr, ra_ptr, if_monitor, this->gemm_stream
    );
    for (int iter = 0; iter < SegSize; iter++){
        int this_seg = cseg_cpu_ptr[iter];
        int commSize = M * N / TileNum * this_seg;
        // The signal is reset by the wait kernel
        kernel_wait_flag<<<1, 1, 0, this->comm_stream>>> (this_seg, (mm_ptr + iter));
        // Communicate the data
        if (use_ooverlap) {
            this->OoverlapAllReduceSlice(
                static_cast<size_t>(acc_addr),
                static_cast<size_t>(commSize),
                this->comm_stream);
        } else {
            NCCL_CHECK(ncclAllReduce(
                (void *)(c_ptr + acc_addr),
                (void *)(c_ptr + acc_addr),
                commSize,
                ncclFloat16,
                ncclSum,
                this->comm,
                this->comm_stream));
        }
        acc_addr += commSize;
    }

    cudaEventCreateWithFlags(&this->gemm_finished, cudaEventDisableTiming);
    cudaEventRecord(this->gemm_finished, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);
    cudaEventDestroy(this->gemm_finished);
}

void OverlapImpl::GemmReduceScatterOverlap(
        at::Tensor A,  // M, K
        at::Tensor B,  // N, K
        at::Tensor C,  // M, N
        at::Tensor D,  // M / world_size, N
        at::Tensor MM, // TM + 1, TN
        at::Tensor RA, // TM, TN
        at::Tensor RE, // M
        int64_t rLDN, 
        at::Tensor cSEG_CPU, // SegSize, how many communication segments
        at::Tensor cSEG_GPU, // SegSize, how many communication segments, on GPU
        int64_t Algo, 
        bool if_monitor
        ){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    int TM = RA.size(0);
    int TN = RA.size(1);
    int TileNum = TM * TN;

    int SegSize = cSEG_GPU.size(0);

    half* a_ptr = reinterpret_cast<half *>(A.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half *>(B.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());
    int* mm_ptr = MM.data_ptr<int>();
    int* ra_ptr = RA.data_ptr<int>();
    int* re_ptr = RE.data_ptr<int>();

    int* cseg_cpu_ptr = cSEG_CPU.data_ptr<int>();
    int* cseg_gpu_ptr = cSEG_GPU.data_ptr<int>();

    int acc_addr = 0;
    scatter_func_table[Algo](
        M, N, K, rLDN, cseg_gpu_ptr, a_ptr, b_ptr, c_ptr, mm_ptr, ra_ptr, re_ptr, if_monitor, this->gemm_stream
    );
    for (int iter = 0; iter < SegSize; iter++){
        int this_seg = cseg_cpu_ptr[iter];
        int commSize = M * N / TileNum * this_seg;
        // The signal is reset by the wait kernel
        kernel_wait_flag<<<1, 1, 0, this->comm_stream>>> (this_seg, (mm_ptr + iter));
        // Communicate the data
        NCCL_CHECK(ncclReduceScatter((void *)(c_ptr + acc_addr), (void *)(d_ptr + acc_addr / this->my_size), 
            (commSize / this->my_size), ncclFloat16, ncclSum, this->comm, this->comm_stream));
        acc_addr += commSize;
    }

    cudaEventCreateWithFlags(&this->gemm_finished, cudaEventDisableTiming);
    cudaEventRecord(this->gemm_finished, this->comm_stream);
    cudaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);
    cudaEventDestroy(this->gemm_finished);
}
