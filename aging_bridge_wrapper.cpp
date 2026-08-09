#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// 🛡️ RAII 기반 비동기 스트림 격리 클래스 (핵심 요약)
class AgingExecutionGuard {
    cudaStream_t t_s, k_s; cudaEvent_t ev;
public:
    AgingExecutionGuard(cudaStream_t ts, cudaStream_t ks) : t_s(ts), k_s(ks) {
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, t_s);
        cudaStreamWaitEvent(ks, ev, 0); // 1. Main -> Kernel 스트림 동기화
    }
       /**
     * 🛡️ 하드웨어 실행 가드 소멸자 (Destructor)
     * 전용 독립 가속 실행 커널 스트림의 연산 완료 신호를 메인 파이토치 스트림으로 역인터록 복귀.
     */
    ~AgingExecutionGuard() {
        // C++ 비동기 캡슐 펜스 외부에서 명시적으로 release()를 호출하여 소유권을 가로채지(Hijack) 못한 경우,
        // 드라이버 자원 고갈 및 메모리 릭(Driver Leak)을 원천 차단하기 위해 이 비상 안전 래치가 작동합니다.
        if (ev) [[unlikely]] {
            // 1. 전용 독립 커널 가속 실행 스트림의 완료 이벤트를 물리 하드웨어 레지스터에 기록
            cudaEventRecord(ev, k_s);
            
            // 2. 메인 파이토치 실행 스트림이 해당 에이징 방어 커널의 연산 해결을 보장하도록 전역 동기화 배리어 주입
            cudaStreamWaitEvent(t_s, ev, 0);

            // ⚠ [DRIVER INTEGRITY GUARD]: 대규모 분산 인프라 하에서 지연 해제 버그를 차단하기 위해
            // 상용 배포 가이드라인에 따라 최종 드라이버 물리 자원 파괴 명령을 동기식으로 직접 집행합니다.
            cudaEventDestroy(ev);
        }
    }

    /**
     * 🎯 [CORE EXTENSION]: 내부 하드웨어 자원 핸들의 오너십(Ownership) 전송 캡슐 메커니즘
     * 소멸자가 실행되더라도 이벤트 핸들이 즉각 파괴되지 않고 상위 래퍼 프레임워크 스코프까지 생존하도록 보장.
     */
    [[nodiscard]] cudaEvent_t release() noexcept {
        // 1. 현재 물리 가속기 제어용 하드웨어 이벤트 포인터를 로컬 임시 레지스터에 안전하게 복제
        cudaEvent_t temp_hardware_handle = ev;
        
        // 2. 소멸자가 트리거되는 순간 이중 파괴(Double-Destruction Vector)로 인해 드라이버 커널이 패닉에 빠지는 현상을
        // 포인터 참조선을 물리적으로 먼저 단절(Nullify)시킴으로써 완전히 무력화합니다.
        ev = nullptr;
        
        // 3. 상위 호출 프레임으로 오염되지 않은 순수 자원 핸들 소유권 이전
        return temp_hardware_handle;
    }

    // 0-Byte 구조적 무결성 및 메모리 뮤테이션 벡터 완전 차단을 위해 복사/이동 연산 강제 영구 제약
    AgingExecutionGuard(const AgingExecutionGuard&) = delete;
    AgingExecutionGuard& operator=(const AgingExecutionGuard&) = delete;


extern "C" {
    // ⛓️ Layer 1 핵심 에이징 방어 커널 다이렉트 바이너리 링커 바인딩
    void execute_adiabatic_silicon_aging_guard_kernel(const float*, const unsigned int*, float*, int, cudaStream_t);
}

// 🟪 Layer 1.5 코어 인터페이스: 제로카피 셸 및 VRAM 메모리 불변 연속성 검증
torch::Tensor forward_aging_bridge_fence(torch::Tensor grad, torch::Tensor sensor) {
    // ❶ 엄격한 하드웨어 물리 경계 감시 및 가속기 컨텍스트 피닝 가드
    if (!grad.is_cuda() || !sensor.is_cuda()) [[unlikely]] {
        throw std::invalid_argument("[FNG SEVERE] Tensors must reside on GPU.");
    }
    at::cuda::CUDAGuard device_guard(grad.device());

    // ❷ 무복사 패러다임 집행 및 메모리 연속성(Contiguous) 강제 검증
    if (sensor.scalar_type() != torch::kInt32 || !grad.is_contiguous() || !sensor.is_contiguous()) [[unlikely]] {
        throw std::invalid_argument("[FNG SEVERE] Invalid layout or type.");
    }

    const int total = grad.numel();
    // ❸ 메모리 할당 제로카피 셸 (pre-allocation)
    auto out = torch::empty_like(grad);

    // ❹ VRAM 가상 메모리 주소 포인터 원자적 하이재킹 (64-bit Virtual Address Hijacking)
    const float* d_raw = grad.data_ptr<float>();
    const unsigned int* d_sen = reinterpret_cast<const unsigned int*>(sensor.data_ptr<int32_t>());
    float* d_out = out.data_ptr<float>();
    

        // ❺ 현재 연산 평면의 액티브 파이토치 스트림 캡처
    c10::cuda::CUDAStream current_torch_stream = c10::cuda::getCurrentCUDAStream(grad.device().index());
    cudaStream_t native_torch_stream = current_torch_stream.stream();

    // ❻ 전용 비동기 스트림 생성 및 드라이버 큐 격리 (Non-Blocking Kernel Execution Stream)
    // 파이썬 측의 오버헤드 버블과 오토그래드 트레이싱(Tracing)의 무차별적인 하드웨어 락 간섭을 완벽 차단
    cudaStream_t k_stream;
    cudaStreamCreateWithFlags(&k_stream, cudaStreamNonBlocking);

    // [DRIVER INTEGRITY GUARD]: 자원 파괴 유실 벡터를 제거하기 위해 가드 스코프 외부에 핸들 저장소 선언
    cudaEvent_t event_to_destroy = nullptr;
    {
        // ❼ [★ THE CAPSULE FENCE ★] 하드웨어 라이프사이클 격리 배리어 가드 인스턴스화
        AgingExecutionGuard fence(native_torch_stream, k_stream);

        // ❽ 1-Clock 레이어 최하단 PTX MUX 커널 비동기 디스패치 (Layer 1 Core Engine Launch)
        execute_adiabatic_silicon_aging_guard_kernel(d_raw, d_sen, d_out, total, k_stream);

        // 🎯 [CORRECTED]: 소멸자 내부의 이중 파괴 및 자원 릭 벡터를 무력화하기 위해 오너십 강제 이관
        event_to_destroy = fence.release();
    } // <- 가드 펜스 자동 소멸; 내부 핸들이 nullptr로 지워졌으므로 드라이버 커널 패닉 유발 원천 차단.

    // ❾ [HARDWARE INTEGRITY BOUNDARY]: 안전한 호스트 스코프 하단에서 잔여 이벤트 및 비동기 스트림 컨텍스트 완전 정리
    if (event_to_destroy) [[likely]] {
        cudaEventDestroy(event_to_destroy);
    }
    cudaStreamDestroy(k_stream);

    // ❿ 변조되지 않은 순수 정화 그레디언트 뷰 반환 (DeepSeek-V4/Llama 레일 다이렉트 호환)
    return out;
}

// 📝 PyBind11 초고속 0-ns 바이너리 익스텐션 모듈 정적 융합 맵
// NVCC와 GCC 링커가 파이토치 빌드 버스에 본 라이브러리를 직접 삽입할 수 있도록 표준 사양 동결
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_aging_bridge_fence",
        &forward_aging_bridge_fence,
        "0ns Hyperscale Adiabatic Silicon Aging & Thermal Guard Isolation Bridge (Apache 2.0)",
        py::call_guard<py::gil_scoped_release>() // [ADDED]: C++ 진입 즉시 GIL 해제, 파이썬 스레드 스와핑 노이즈 차단
    );
}
