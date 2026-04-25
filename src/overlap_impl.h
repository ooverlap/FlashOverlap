#pragma once

#include <nccl.h>
#include <vector>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/util/device_memory.h"
#include "ooverlap/comm.h"

class OverlapImpl : public torch::CustomClassHolder {
    public:
        OverlapImpl();
        ~OverlapImpl();

        void CutlassInit();
        void NcclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void OoverlapIpcInit(
            const int64_t tp_rank,
            const int64_t tp_size,
            const std::vector<int64_t> devices,
            const std::string broker_key);
        void OverlapInit();

        void Gemm(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);

        void GemmAllReduceOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);
        void GemmReduceScatterOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo, at::Tensor mLen_CPU);

        void SegAllReduce(at::Tensor C, at::Tensor cSEG_CPU, int64_t SegNum);
        void NcclAllReduce(at::Tensor C);
        void NcclReduceScatter(at::Tensor C);
        void NcclAll2All(at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);

    private:
        void OoverlapUnregisterBuffer();
        void OoverlapEnsureBuffer(at::Tensor C);
        void OoverlapAllReduceSlice(size_t element_offset, size_t count, cudaStream_t stream);
    
        oo_group_t* oo_group = nullptr;
        oo_node_t* oo_node = nullptr;
        oo_buffer_t* oo_local_buf = nullptr;
        oo_buffer_t* oo_peer_buf = nullptr;
    
        void* oo_registered_ptr = nullptr;
        size_t oo_registered_bytes = 0;
    
        int64_t oo_rank = -1;
        int64_t oo_size = 0;
        int oo_devices[2] = {-1, -1};
        bool oo_initialized = false;

        cudaStream_t gemm_stream;
        cudaStream_t comm_stream;
        cudaEvent_t gemm_finished;

        ncclComm_t comm;
        int64_t my_rank;
        int64_t my_size;
};
