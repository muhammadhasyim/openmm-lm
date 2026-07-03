#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaPlatform.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace OpenMM;

namespace py = pybind11;

#define CHECK_CUDA_DRIVER(result, prefix)                                        \
    if ((result) != CUDA_SUCCESS) {                                             \
        const char* err = nullptr;                                              \
        cuGetErrorString(static_cast<CUresult>(result), &err);                  \
        std::stringstream msg;                                                  \
        msg << (prefix) << ": " << (err != nullptr ? err : "unknown CUDA error") \
            << " (" << (result) << ") at " << __FILE__ << ":" << __LINE__;    \
        throw OpenMMException(msg.str());                                       \
    }

namespace {

std::uintptr_t parse_hex_pointer_from_repr(const std::string& repr) {
    const std::string marker = "0x";
    const std::size_t start = repr.find(marker);
    if (start == std::string::npos) {
        throw std::runtime_error("could not find a SWIG pointer address in object repr: " + repr);
    }
    std::size_t end = start + marker.size();
    while (end < repr.size() && std::isxdigit(static_cast<unsigned char>(repr[end]))) {
        ++end;
    }
    std::uintptr_t value = 0;
    std::stringstream stream;
    stream << std::hex << repr.substr(start + marker.size(), end - start - marker.size());
    stream >> value;
    if (value == 0) {
        throw std::runtime_error("parsed a null SWIG pointer from object repr: " + repr);
    }
    return value;
}

std::uintptr_t pointer_from_swig_object(const py::object& object) {
    py::object raw = py::hasattr(object, "this") ? object.attr("this") : object;
    try {
        py::object as_int = py::module_::import("builtins").attr("int")(raw);
        return static_cast<std::uintptr_t>(as_int.cast<unsigned long long>());
    } catch (const py::error_already_set&) {
        PyErr_Clear();
    }
    return parse_hex_pointer_from_repr(py::repr(raw).cast<std::string>());
}

Context& unwrap_context(const py::object& context_object) {
    auto* context = reinterpret_cast<Context*>(pointer_from_swig_object(context_object));
    if (context == nullptr) {
        throw std::runtime_error("received a null OpenMM Context pointer");
    }
    return *context;
}

} // namespace

extern "C" void openmm_cuda_bridge_write_positions(
        std::uintptr_t pos_tensor_ptr,
        std::uintptr_t posq_ptr,
        std::uintptr_t correction_ptr,
        std::uintptr_t atom_index_ptr,
        int posq_element_size,
        int num_atoms,
        cudaStream_t stream);

extern "C" void openmm_cuda_bridge_read_positions(
        std::uintptr_t posq_ptr,
        std::uintptr_t correction_ptr,
        int posq_element_size,
        std::uintptr_t atom_index_ptr,
        std::uintptr_t particle_xyz_ptr,
        std::uintptr_t slot_xyz_ptr,
        int num_atoms,
        cudaStream_t stream);

extern "C" void openmm_cuda_bridge_read_forces(
        std::uintptr_t force_buffers_ptr,
        std::uintptr_t force_tensor_ptr,
        std::uintptr_t atom_index_ptr,
        int num_atoms,
        int padded_num_atoms,
        cudaStream_t stream);

static CudaContext& get_cuda_context(ContextImpl& impl) {
    auto* data = static_cast<CudaPlatform::PlatformData*>(impl.getPlatformData());
    if (data == nullptr || data->contexts.empty()) {
        throw OpenMMException("Context is not backed by an initialized CUDA PlatformData");
    }
    return *data->contexts[0];
}

static void* tensor_data_pointer(CudaContext& cu, torch::Tensor& tensor) {
    if (cu.getUseDoublePrecision()) {
        return tensor.data_ptr<double>();
    }
    return tensor.data_ptr<float>();
}

static cudaStream_t openmm_stream(CudaContext& cu) {
    return reinterpret_cast<cudaStream_t>(cu.getCurrentStream());
}

static std::uintptr_t device_pointer(CudaArray& array) {
    return static_cast<std::uintptr_t>(array.getDevicePointer());
}

static std::uintptr_t correction_pointer(CudaContext& cu, CudaArray& posq) {
    CudaArray& correction = cu.getPosqCorrection();
    if (!correction.isInitialized() || correction.getSize() == 0) {
        return 0;
    }
    if (correction.getElementSize() != posq.getElementSize()) {
        return 0;
    }
    return device_pointer(correction);
}


class CudaBridge {
public:
    explicit CudaBridge(py::object context_object, int groups = 0xFFFFFFFF)
        : context_object_(std::move(context_object)),
          context_(&unwrap_context(context_object_)),
          impl_(&context_->getImplementation()),
          cu_(get_cuda_context(*impl_)),
          groups_(groups) {}

    ~CudaBridge() = default;

    int num_atoms() const {
        return cu_.getNumAtoms();
    }

    int padded_num_atoms() const {
        return cu_.getPaddedNumAtoms();
    }

    int device_index() const {
        return cu_.getDeviceIndex();
    }

    void set_positions(torch::Tensor positions_angstrom) {
        validate_positions(positions_angstrom);
        torch::Tensor pos = canonical_tensor(positions_angstrom);
        void* pos_data = tensor_data_pointer(cu_, pos);
        const int num_atoms = cu_.getNumAtoms();
        {
            ContextSelector selector(cu_);
            CudaArray& posq = cu_.getPosq();
            openmm_cuda_bridge_write_positions(
                    reinterpret_cast<std::uintptr_t>(pos_data),
                    device_pointer(posq),
                    correction_pointer(cu_, posq),
                    device_pointer(cu_.getAtomIndexArray()),
                    posq.getElementSize(),
                    num_atoms,
                    openmm_stream(cu_));
            cudaStreamSynchronize(openmm_stream(cu_));
        }
    }

    torch::Tensor get_positions_angstrom() {
        const int num_atoms = cu_.getNumAtoms();
        torch::Tensor positions = torch::empty(
            {num_atoms, 3},
            torch::TensorOptions().device(torch::kCUDA, cu_.getDeviceIndex()).dtype(torch::kFloat64)
        );
        torch::Tensor slot_xyz = torch::empty(
            {num_atoms, 3},
            torch::TensorOptions().device(torch::kCUDA, cu_.getDeviceIndex()).dtype(torch::kFloat64)
        );

        {
            ContextSelector selector(cu_);
            CudaArray& posq = cu_.getPosq();
            openmm_cuda_bridge_read_positions(
                    device_pointer(posq),
                    correction_pointer(cu_, posq),
                    posq.getElementSize(),
                    device_pointer(cu_.getAtomIndexArray()),
                    reinterpret_cast<std::uintptr_t>(positions.data_ptr<double>()),
                    reinterpret_cast<std::uintptr_t>(slot_xyz.data_ptr<double>()),
                    num_atoms,
                    openmm_stream(cu_));
            cudaStreamSynchronize(openmm_stream(cu_));
        }
        if (cu_.getUseDoublePrecision()) {
            return positions;
        }
        return positions.to(torch::kFloat32);
    }

    torch::Tensor compute_forces() {
        const int num_atoms = cu_.getNumAtoms();
        const int padded_num_atoms = cu_.getPaddedNumAtoms();
        torch::Tensor forces_fp64 = torch::empty(
            {num_atoms, 3},
            torch::TensorOptions().device(torch::kCUDA, cu_.getDeviceIndex()).dtype(torch::kFloat64)
        );

        {
            ContextSelector selector(cu_);
            impl_->computeVirtualSites();
            impl_->calcForcesAndEnergy(true, false, groups_);
            openmm_cuda_bridge_read_forces(
                    device_pointer(cu_.getForce()),
                    reinterpret_cast<std::uintptr_t>(forces_fp64.data_ptr<double>()),
                    device_pointer(cu_.getAtomIndexArray()),
                    num_atoms,
                    padded_num_atoms,
                    openmm_stream(cu_));
            cudaStreamSynchronize(openmm_stream(cu_));
        }
        if (cu_.getUseDoublePrecision()) {
            return forces_fp64;
        }
        return forces_fp64.to(torch::kFloat32);
    }

    torch::Tensor evaluate(torch::Tensor positions_angstrom) {
        set_positions(positions_angstrom);
        return compute_forces();
    }

private:
    void validate_positions(const torch::Tensor& positions_angstrom) const {
        if (!positions_angstrom.is_cuda()) {
            throw OpenMMException("positions_angstrom must be a CUDA tensor");
        }
        if (positions_angstrom.dim() != 2 || positions_angstrom.size(1) != 3) {
            throw OpenMMException("positions_angstrom must have shape (N, 3)");
        }
        if (positions_angstrom.size(0) != cu_.getNumAtoms()) {
            std::stringstream msg;
            msg << "positions_angstrom has " << positions_angstrom.size(0)
                << " atoms, but OpenMM context has " << cu_.getNumAtoms();
            throw OpenMMException(msg.str());
        }
        if (positions_angstrom.get_device() != cu_.getDeviceIndex()) {
            std::stringstream msg;
            msg << "positions tensor is on CUDA device " << positions_angstrom.get_device()
                << ", but OpenMM context is on CUDA device " << cu_.getDeviceIndex();
            throw OpenMMException(msg.str());
        }
    }

    torch::Tensor canonical_tensor(torch::Tensor tensor) const {
        const auto dtype = cu_.getUseDoublePrecision() ? torch::kFloat64 : torch::kFloat32;
        return tensor.to(torch::TensorOptions().device(torch::kCUDA, cu_.getDeviceIndex()).dtype(dtype)).contiguous();
    }

    py::object context_object_;
    Context* context_;
    ContextImpl* impl_;
    CudaContext& cu_;
    int groups_;
};

PYBIND11_MODULE(_openmm_cuda_bridge, m) {
    py::class_<CudaBridge>(m, "CudaBridge")
        .def(
            py::init([](py::object ctx, int groups_in) {
                return new CudaBridge(std::move(ctx), groups_in);
            }),
            py::arg("context"),
            py::arg("groups") = static_cast<int>(0xFFFFFFFF))
        .def("num_atoms", &CudaBridge::num_atoms)
        .def("padded_num_atoms", &CudaBridge::padded_num_atoms)
        .def("device_index", &CudaBridge::device_index)
        .def("set_positions", &CudaBridge::set_positions)
        .def("get_positions_angstrom", &CudaBridge::get_positions_angstrom)
        .def("compute_forces", &CudaBridge::compute_forces)
        .def("evaluate", &CudaBridge::evaluate);
}
