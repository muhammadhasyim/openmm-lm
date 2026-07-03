#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kBlockSize = 256;

inline void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

__global__ void extract_posq_f4_kernel(
        const float4* posq,
        const float4* correction,
        double* dst,
        int num_atoms) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        const float4 p = posq[atom];
        double x = static_cast<double>(p.x);
        double y = static_cast<double>(p.y);
        double z = static_cast<double>(p.z);
        if (correction != nullptr) {
            const float4 c = correction[atom];
            x += static_cast<double>(c.x);
            y += static_cast<double>(c.y);
            z += static_cast<double>(c.z);
        }
        dst[3 * atom] = x * 10.0;
        dst[3 * atom + 1] = y * 10.0;
        dst[3 * atom + 2] = z * 10.0;
    }
}

__global__ void extract_posq_d4_kernel(const double4* posq, double* dst, int num_atoms) {
    for (int atom = blockIdx.x * blockDim.x + threadIdx.x; atom < num_atoms;
         atom += blockDim.x * gridDim.x) {
        const double4 p = posq[atom];
        dst[3 * atom] = p.x * 10.0;
        dst[3 * atom + 1] = p.y * 10.0;
        dst[3 * atom + 2] = p.z * 10.0;
    }
}

__global__ void gather_by_atom_index_kernel(
        const double* slot_xyz,
        const int* atom_index,
        double* particle_xyz,
        int num_atoms) {
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < num_atoms;
         slot += blockDim.x * gridDim.x) {
        const int particle = atom_index[slot];
        particle_xyz[3 * particle] = slot_xyz[3 * slot];
        particle_xyz[3 * particle + 1] = slot_xyz[3 * slot + 1];
        particle_xyz[3 * particle + 2] = slot_xyz[3 * slot + 2];
    }
}

__global__ void write_positions_f4_kernel(
        const float* pos_tensor,
        float4* posq,
        const int* atom_index,
        int num_atoms) {
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < num_atoms;
         slot += blockDim.x * gridDim.x) {
        const int particle = atom_index[slot];
        float4 p = posq[slot];
        p.x = pos_tensor[3 * particle] * 0.1f;
        p.y = pos_tensor[3 * particle + 1] * 0.1f;
        p.z = pos_tensor[3 * particle + 2] * 0.1f;
        posq[slot] = p;
    }
}

__global__ void write_positions_d4_kernel(
        const double* pos_tensor,
        double4* posq,
        const int* atom_index,
        int num_atoms) {
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < num_atoms;
         slot += blockDim.x * gridDim.x) {
        const int particle = atom_index[slot];
        double4 p = posq[slot];
        p.x = pos_tensor[3 * particle] * 0.1;
        p.y = pos_tensor[3 * particle + 1] * 0.1;
        p.z = pos_tensor[3 * particle + 2] * 0.1;
        posq[slot] = p;
    }
}

__global__ void read_forces_kernel(
        const long long* force_buffers,
        double* force_tensor,
        const int* atom_index,
        int num_atoms,
        int padded_num_atoms) {
    const double scale = 1.0 / (4294967296.0 * 96.4853321233100184 * 10.0);
    for (int slot = blockIdx.x * blockDim.x + threadIdx.x; slot < num_atoms;
         slot += blockDim.x * gridDim.x) {
        const int particle = atom_index[slot];
        force_tensor[3 * particle] =
                static_cast<double>(force_buffers[slot]) * scale;
        force_tensor[3 * particle + 1] =
                static_cast<double>(force_buffers[slot + padded_num_atoms]) * scale;
        force_tensor[3 * particle + 2] =
                static_cast<double>(force_buffers[slot + 2 * padded_num_atoms]) * scale;
    }
}

} // namespace

extern "C" void openmm_cuda_bridge_write_positions(
        std::uintptr_t pos_tensor_ptr,
        std::uintptr_t posq_ptr,
        std::uintptr_t correction_ptr,
        std::uintptr_t atom_index_ptr,
        int posq_element_size,
        int num_atoms,
        cudaStream_t stream) {
    const int blocks = (num_atoms + kBlockSize - 1) / kBlockSize;
    if (posq_element_size == static_cast<int>(sizeof(float4))) {
        write_positions_f4_kernel<<<blocks, kBlockSize, 0, stream>>>(
                reinterpret_cast<const float*>(pos_tensor_ptr),
                reinterpret_cast<float4*>(posq_ptr),
                reinterpret_cast<const int*>(atom_index_ptr),
                num_atoms);
    } else if (posq_element_size == static_cast<int>(sizeof(double4))) {
        write_positions_d4_kernel<<<blocks, kBlockSize, 0, stream>>>(
                reinterpret_cast<const double*>(pos_tensor_ptr),
                reinterpret_cast<double4*>(posq_ptr),
                reinterpret_cast<const int*>(atom_index_ptr),
                num_atoms);
    } else {
        throw std::runtime_error("unsupported posq element size for write_positions");
    }
    check_cuda(cudaGetLastError(), "openmm_cuda_bridge_write_positions");
}

extern "C" void openmm_cuda_bridge_read_positions(
        std::uintptr_t posq_ptr,
        std::uintptr_t correction_ptr,
        int posq_element_size,
        std::uintptr_t atom_index_ptr,
        std::uintptr_t particle_xyz_ptr,
        std::uintptr_t slot_xyz_ptr,
        int num_atoms,
        cudaStream_t stream) {
    const int blocks = (num_atoms + kBlockSize - 1) / kBlockSize;
    if (posq_element_size == static_cast<int>(sizeof(float4))) {
        extract_posq_f4_kernel<<<blocks, kBlockSize, 0, stream>>>(
                reinterpret_cast<const float4*>(posq_ptr),
                correction_ptr == 0 ? nullptr : reinterpret_cast<const float4*>(correction_ptr),
                reinterpret_cast<double*>(slot_xyz_ptr),
                num_atoms);
    } else if (posq_element_size == static_cast<int>(sizeof(double4))) {
        extract_posq_d4_kernel<<<blocks, kBlockSize, 0, stream>>>(
                reinterpret_cast<const double4*>(posq_ptr),
                reinterpret_cast<double*>(slot_xyz_ptr),
                num_atoms);
    } else {
        throw std::runtime_error("unsupported posq element size for read_positions");
    }
    check_cuda(cudaGetLastError(), "openmm_cuda_bridge_extract_posq");
    gather_by_atom_index_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const double*>(slot_xyz_ptr),
            reinterpret_cast<const int*>(atom_index_ptr),
            reinterpret_cast<double*>(particle_xyz_ptr),
            num_atoms);
    check_cuda(cudaGetLastError(), "openmm_cuda_bridge_gather_positions");
}

extern "C" void openmm_cuda_bridge_read_forces(
        std::uintptr_t force_buffers_ptr,
        std::uintptr_t force_tensor_ptr,
        std::uintptr_t atom_index_ptr,
        int num_atoms,
        int padded_num_atoms,
        cudaStream_t stream) {
    const int blocks = (num_atoms + kBlockSize - 1) / kBlockSize;
    read_forces_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const long long*>(force_buffers_ptr),
            reinterpret_cast<double*>(force_tensor_ptr),
            reinterpret_cast<const int*>(atom_index_ptr),
            num_atoms,
            padded_num_atoms);
    check_cuda(cudaGetLastError(), "openmm_cuda_bridge_read_forces");
}
