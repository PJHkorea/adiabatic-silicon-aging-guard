#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

// ⚡ L1/L2 캐시 라인 미스 및 일관성(Coherency) 보장을 위한 32바이트 물리 정렬 구조체
struct alignas(32) AgingTelemetryCell {
    float gradient_footprint[8]; // 4바이트 * 8 = 32바이트 정렬 고정
};

/**
 * @brief 인라인 PTX를 활용하여 Warp Divergence를 원천 차단하는 브랜치리스(Branchless) 선택 함수
 * @note NVCC 컴파일러 최적화 파이프라인에서 분기 예측 실패 오버헤드를 1클럭 수준으로 격리합니다.
 */
__device__ __forceinline__ float pinn_branchless_select_f32(bool condition, float true_val, float false_val) {
    float output_reg;
    
    // PTX 내 컴파일러 로컬 프레디케이션 레지스터 정의 최적화 및 탭 정렬
    asm volatile (
        "{\n\t"
        "  .reg .pred p_state;\n\t"
        "  setp.ne.u32 p_state, %3, 0;\n\t"
        "  selp.f32    %0, %1, %2, p_state;\n\t"
        "}"
        : "=f"(output_reg)
        : "f"(true_val), "f"(false_val), "r"((unsigned int)condition)
    );
    
    return output_reg;
}

extern "C" {
__global__ void adiabatic_silicon_aging_guard_kernel(
    const float* __restrict__ d_raw_gradients,
    const unsigned int* __restrict__ d_aging_sensor,
    float* __restrict__ d_stabilized_gradients,
    const int total_elements
) {
    // ❶ 하드웨어 그리드 매핑 및 유효성 래치 격리
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_valid_thread = (thread_idx < total_elements);

    // ❷ Tiled Warp Partitioning (32개 스레드 단일 물리 실행 유닛 고정)
    cooperative_groups::thread_block_tile<32> warp_tile = 
        cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    // ❸ 1-Clock Crossbar Shuffling (레지스터 기반 초저지연 상호 교환)
    // 메모리 가드: 유효하지 않은 스레드는 안전하게 0.0f 패딩 값을 레지스터에 로드
    float raw_grad_register = is_valid_thread ? d_raw_gradients[thread_idx] : 0.0f;
    float right_neighbor_grad = warp_tile.shfl_down(raw_grad_register, 1);
    float left_neighbor_grad  = warp_tile.shfl_up(raw_grad_register, 1);

    // ❹ 테일 워프(Tail-Warp) 경계면 클램핑 및 유효 플래그 추적
    int lane_id = warp_tile.thread_rank();
    unsigned int has_right_node = warp_tile.shfl_down((unsigned int)is_valid_thread, 1);
    unsigned int has_left_node  = warp_tile.shfl_up((unsigned int)is_valid_thread, 1);

    bool is_right_edge = (lane_id == 31) || (has_right_node == 0);
    bool is_left_edge  = (lane_id == 0)  || (has_left_node == 0);

    // 경계면에 도달하면 인접 값이 아닌 자신의 레지스터 값을 복제하여 매니폴드 파괴 방지
    right_neighbor_grad = pinn_branchless_select_f32(is_right_edge, raw_grad_register, right_neighbor_grad);
    left_neighbor_grad  = pinn_branchless_select_f32(is_left_edge, raw_grad_register, left_neighbor_grad);

    // ❺ Burgers' Formulation 기반의 2차 공간 라플라시안 점성 제어 수식
    // 노화/열화 노드 누락 시 그레디언트 폭발(NaN)을 막기 위한 0.012f(Beta) 비열적 평탄화 감쇠
    float laplacian_gradient = right_neighbor_grad + left_neighbor_grad - (2.0f * raw_grad_register);
    const float beta_viscosity_alpha = 0.012f;
    float adiabatic_damped_gradient = raw_grad_register + (beta_viscosity_alpha * laplacian_gradient);

    // ❻ 전역 쇼크웨이브 열화/고장 텔레메트리 비트 집합 (32비트 하드웨어 Ballot Aggregation)
    // 0xFFFFFFFF 액티브 마스크 유입 및 패딩 스레드의 불법 VRAM 메모리 주소선 접근 차단
    unsigned int raw_aging_sensor_bit = is_valid_thread ? d_aging_sensor[thread_idx] : 0;
    unsigned int global_aging_telemetry_mask = warp_tile.ballot(is_valid_thread && (raw_aging_sensor_bit > 0));

    // ❼ 하드웨어 레지스터 1비트 마스킹 및 0.0f Vacuum Erasure 회로
    // 로컬 레인 ID에 대응하는 특정 비트를 마이크로 단위로 정밀 추출
    bool local_silicon_fault = (global_aging_telemetry_mask & (1u << lane_id)) != 0;

    // 에이징 임계치를 돌파한 불량 노드는 0.0f로 강제 소멸, 정상 노드는 단열 평탄화 수식 결과물 출력
    float stabilized_output = pinn_branchless_select_f32(
        local_silicon_fault,
        0.0f,                       // 오염 확산을 완벽히 격리하는 진공 소멸 상태
        adiabatic_damped_gradient  // Burgers' Formulation으로 보정된 그레디언트 매니폴드
    );

    // ❽ 글로벌 VRAM 메모리 주소선 보호 가드 및 주입 커밋
    // 워프 동기화 정렬을 위해 활성화해 둔 찌꺼기 패딩 쓰레드의 잘못된 쓰기(Segmentation Fault) 원천 차단
    if (is_valid_thread) {
        d_stabilized_gradients[thread_idx] = stabilized_output;
    }
}
} // extern "C"


/**
 * 🚀 호스트 사이드 C++ 트램펄린 링커 (Called by Layer 1.5 aging_bridge_wrapper.cpp)
 * 가속기 내부 하드웨어 제약을 역산하여 최적의 블록/그리드 점유율을 0ns 호스트 오버헤드로 도출
 */
void execute_adiabatic_silicon_aging_guard_kernel(
    const float* d_raw_gradients,
    const unsigned int* d_aging_sensor,
    float* d_stabilized_gradients,
    const int total_elements,
    cudaStream_t stream
) {
    // ❶ 빈 텐서 유입 시의 즉각적인 하드웨어 조기 이탈 보호 가드
    if (total_elements <= 0) [[unlikely]] return;

    int block_size = 0;
    int min_grid_size = 0;

    // ❷ 하드웨어 아키텍처 점유율 역산 자동 최적화 회로 (Occupancy Max Potential Calculator)
    // NVCC 컴파일러의 템플릿 타입 추론을 살려 컴파일 타임 형 안전성을 보장하고 void* 캐스팅 오류 제거
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        adiabatic_silicon_aging_guard_kernel,
        0, // 동적 공유 메모리(Shared Memory) 프로파일링 오프셋 제거
        0  // 정적 블록 크기 상한선 제약 바이패스
    );

    // ❸ total_elements 수량에 완벽하게 정렬된 정적 가속 실행 그리드 규격 도출
    int grid_size = (total_elements + block_size - 1) / block_size;

    // ❹ 호스트 비블로킹(Non-blocking) 방식의 액티브 XLA / PyTorch 스트림 직접 디스패치
    adiabatic_silicon_aging_guard_kernel<<<grid_size, block_size, 0, stream>>>(
        d_raw_gradients,
        d_aging_sensor,
        d_stabilized_gradients,
        total_elements
    );

    // ❺ C++20 [[unlikely]] 어트리뷰트 기반 분기 최적화 및 에러 트래핑 방화벽
    // 디스패치 시점의 큐 주입 무결성을 우선 검증 (비동기 런타임 에러는 스트림 동기화 단계에서 캐치)
    cudaError_t kernel_launch_err = cudaGetLastError();
    if (kernel_launch_err != cudaSuccess) [[unlikely]] {
        // 프로덕션 스트림을 마비시키는 호스트 print 계열 문구를 배제하고 런타임 표준 익셉션으로 상위 래치에 토스
        throw std::runtime_error("[FNG CRYOGENIC AGING FATAL]: PTX MUX Kernel launch failed: " + 
            std::string(cudaGetErrorString(kernel_launch_err)));
    }
}

