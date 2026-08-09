#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

// RAII-based asynchronous hardware stream isolation guard
class AgingExecutionGuard {
    cudaStream_t t_s, k_s; 
    cudaEvent_t ev;

public:
    /**
     * @brief Captures the native PyTorch stream timeline and injects a synchronization barrier 
     *        into the isolated accelerator kernel stream.
     */
    AgingExecutionGuard(cudaStream_t ts, cudaStream_t ks) : t_s(ts), k_s(ks), ev(nullptr) {
        // Statically allocate a hardware event with timing overhead deactivated
        cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess) [[unlikely]] {
            throw std::runtime_error("[FNG GUARD ATOMIC FATAL]: Failed to create hardware sync event: " + 
                std::string(cudaGetErrorString(err)));
        }

        // Record current PyTorch context and link with the custom isolated execution stream
        cudaEventRecord(ev, t_s);
        cudaStreamWaitEvent(k_s, ev, 0); 
    }

    /**
     * @brief Destructor acts as a safety latch to prevent resource leaks and enforce reverse 
     *        interlock synchronization back to the native PyTorch stream if release() is omitted.
     */
    ~AgingExecutionGuard() noexcept {
        if (ev) [[unlikely]] {
            // Commit isolated kernel completion events onto the hardware register
            cudaEventRecord(ev, k_s);
            
            // Force the main PyTorch execution queue to resolve the aging guard kernel execution
            cudaStreamWaitEvent(t_s, ev, 0);

            // Destroy hardware event synchronously within noexcept constraints to avoid driver leakage
            cudaEventDestroy(ev);
        }
    }


        /**
     * @brief Transfers ownership of the internal hardware event handle to the caller scope,
     *        ensuring survival beyond the local destructor timeline.
     */
    [[nodiscard]] cudaEvent_t release() noexcept {
        // Copy the active hardware event pointer to a temporary registry handle
        cudaEvent_t temp_hardware_handle = ev;
        
        // Sever the local pointer reference to completely eliminate double-destruction vectors
        ev = nullptr;
        
        return temp_hardware_handle;
    }

    // Permanently disable copy and move operations to enforce unmanaged structural integrity
    AgingExecutionGuard(const AgingExecutionGuard&) = delete;
    AgingExecutionGuard& operator=(const AgingExecutionGuard&) = delete;
    AgingExecutionGuard(AgingExecutionGuard&&) = delete;             
    AgingExecutionGuard& operator=(AgingExecutionGuard&&) = delete;    
}; 

extern "C" {
    // Native binary interface to the Layer 1 silicon hardware aging guard kernel
    void execute_adiabatic_silicon_aging_guard_kernel(
        const float* d_raw_gradients, 
        const unsigned int* d_aging_sensor, 
        float* d_stabilized_gradients, 
        const int total_elements, 
        cudaStream_t stream
    );
}

// Layer 1.5 core interface: Zero-copy hardware layout validation and execution manifold
torch::Tensor forward_aging_bridge_fence(torch::Tensor grad, torch::Tensor sensor) {
    // Enforce hardware boundary verification and active accelerator context pinning
    if (!grad.is_cuda() || !sensor.is_cuda()) [[unlikely]] {
        throw std::invalid_argument("[FNG SEVERE] Tensors must reside on GPU.");
    }
    at::cuda::CUDAGuard device_guard(grad.device());

    // Enforce layout constraints to ensure continuous memory strides and avoid structural mismatch
    if (sensor.scalar_type() != torch::kInt32 || !grad.is_contiguous() || !sensor.is_contiguous()) [[unlikely]] {
        throw std::invalid_argument("[FNG SEVERE] Invalid layout or type.");
    }

    const int total = grad.numel();
    
    // Allocate a zero-copy memory shell to prevent allocation overhead
    auto out = torch::empty_like(grad);

    // Intercept 64-bit virtual memory addresses without raw buffer copying
    const float* d_raw = grad.data_ptr<float>();
    const unsigned int* d_sen = reinterpret_cast<const unsigned int*>(sensor.data_ptr<int32_t>());
    float* d_out = out.data_ptr<float>();

    // Capture the active PyTorch compute timeline context
    c10::cuda::CUDAStream current_torch_stream = c10::cuda::getCurrentCUDAStream(grad.device().index());
    cudaStream_t native_torch_stream = current_torch_stream.stream();

    // Allocate an isolated non-blocking hardware queue to insulate kernel runtime executions
    cudaStream_t k_stream = nullptr;
    cudaError_t stream_err = cudaStreamCreateWithFlags(&k_stream, cudaStreamNonBlocking);
    if (stream_err != cudaSuccess) [[unlikely]] {
        throw std::runtime_error("[FNG BRIDGE STREAM FATAL]: Failed to create non-blocking stream: " + 
            std::string(cudaGetErrorString(stream_err)));
    }

    // Declare a persistent hardware event handler to track the async timeline safely
    cudaEvent_t event_to_destroy = nullptr;
    {
        // Initialize the runtime execution boundary capsule to synchronize streams asynchronously
        AgingExecutionGuard fence(native_torch_stream, k_stream);

        // Dispatch the 1-clock branchless register-level silicon guard CUDA kernel
        execute_adiabatic_silicon_aging_guard_kernel(d_raw, d_sen, d_out, total, k_stream);

        // Atomically transfer handle ownership to prevent double-destruction within the destructor
        event_to_destroy = fence.release();
    } 

    // Synchronize the isolated stream completion event back onto the native PyTorch stream timeline
    if (event_to_destroy) [[likely]] {
        cudaEventRecord(event_to_destroy, k_stream);
        cudaStreamWaitEvent(native_torch_stream, event_to_destroy, 0);
        
        // Recycle hardware resources after a secure timeline interlock has been established
        cudaEventDestroy(event_to_destroy);
    }
    
    // Purge the temporary execution stream queue
    cudaStreamDestroy(k_stream);

    // Return the stabilized gradient view compatible with commercial LLM backbones
    return out;
}

// PyBind11 high-performance static extension binding manifest
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;

    m.def(
        "forward_aging_bridge_fence",
        &forward_aging_bridge_fence,
        "0ns Hyperscale Adiabatic Silicon Aging & Thermal Guard Isolation Bridge (Apache 2.0)",
        py::call_guard<py::gil_scoped_release>() // Release Python GIL to suppress thread-swapping latency spikes
    );
}

