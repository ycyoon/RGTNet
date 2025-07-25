#!/usr/bin/env python3
"""
CPU 병목 분석 스크립트
"""

import os
import sys
import time
import psutil
import threading
from collections import defaultdict
import multiprocessing as mp

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except:
    NVIDIA_AVAILABLE = False

class PerformanceMonitor:
    def __init__(self):
        self.monitoring = False
        self.data = defaultdict(list)
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started...")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("⏹️  Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Monitoring loop"""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            avg_cpu = sum(cpu_percent) / len(cpu_percent)
            self.data['cpu_avg'].append((timestamp, avg_cpu))
            self.data['cpu_cores'].append((timestamp, cpu_percent))
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            self.data['memory_percent'].append((timestamp, memory.percent))
            self.data['memory_available_gb'].append((timestamp, memory.available / 1024**3))
            
            # 프로세스별 CPU 사용률
            try:
                current_process = psutil.Process()
                self.data['process_cpu'].append((timestamp, current_process.cpu_percent()))
                self.data['process_memory_mb'].append((timestamp, current_process.memory_info().rss / 1024**2))
                self.data['process_threads'].append((timestamp, current_process.num_threads()))
            except:
                pass
            
            # GPU 사용률 (가능한 경우)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_reserved = torch.cuda.memory_reserved() / 1024**3
                    self.data['gpu_memory_allocated_gb'].append((timestamp, gpu_memory))
                    self.data['gpu_memory_reserved_gb'].append((timestamp, gpu_reserved))
                except:
                    pass
            
            if NVIDIA_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.data['gpu_utilization'].append((timestamp, gpu_util.gpu))
                    self.data['gpu_memory_utilization'].append((timestamp, gpu_util.memory))
                except:
                    pass
            
            time.sleep(0.5)  # Monitor every 0.5 seconds
    
    def print_summary(self):
        """Print performance summary"""
        if not self.data:
            print("❌ No monitoring data available")
            return
            
        print("\n" + "="*60)
        print("🔍 CPU 병목 분석 결과")
        print("="*60)
        
        # CPU 분석
        if 'cpu_avg' in self.data:
            cpu_values = [v for t, v in self.data['cpu_avg']]
            avg_cpu = sum(cpu_values) / len(cpu_values)
            max_cpu = max(cpu_values)
            print(f"💻 CPU 사용률:")
            print(f"   평균: {avg_cpu:.1f}%")
            print(f"   최대: {max_cpu:.1f}%")
            
            if avg_cpu > 80:
                print("   🚨 CPU 병목 감지! (평균 80% 이상)")
            elif avg_cpu > 60:
                print("   ⚠️  CPU 사용률 높음 (평균 60% 이상)")
            else:
                print("   ✅ CPU 사용률 정상")
        
        # 메모리 분석
        if 'memory_percent' in self.data:
            memory_values = [v for t, v in self.data['memory_percent']]
            avg_memory = sum(memory_values) / len(memory_values)
            max_memory = max(memory_values)
            print(f"\n🧠 메모리 사용률:")
            print(f"   평균: {avg_memory:.1f}%")
            print(f"   최대: {max_memory:.1f}%")
            
            if avg_memory > 90:
                print("   🚨 메모리 부족! (평균 90% 이상)")
            elif avg_memory > 75:
                print("   ⚠️  메모리 사용률 높음 (평균 75% 이상)")
            else:
                print("   ✅ 메모리 사용률 정상")
        
        # 프로세스 분석
        if 'process_cpu' in self.data:
            process_cpu_values = [v for t, v in self.data['process_cpu']]
            avg_process_cpu = sum(process_cpu_values) / len(process_cpu_values)
            max_process_cpu = max(process_cpu_values)
            print(f"\n🏃 프로세스 성능:")
            print(f"   평균 CPU: {avg_process_cpu:.1f}%")
            print(f"   최대 CPU: {max_process_cpu:.1f}%")
            
            if 'process_threads' in self.data:
                thread_values = [v for t, v in self.data['process_threads']]
                avg_threads = sum(thread_values) / len(thread_values)
                max_threads = max(thread_values)
                print(f"   평균 스레드: {avg_threads:.0f}")
                print(f"   최대 스레드: {max_threads:.0f}")
        
        # GPU 분석
        gpu_available = False
        if 'gpu_utilization' in self.data:
            gpu_util_values = [v for t, v in self.data['gpu_utilization']]
            if gpu_util_values:
                avg_gpu = sum(gpu_util_values) / len(gpu_util_values)
                max_gpu = max(gpu_util_values)
                print(f"\n🎮 GPU 사용률:")
                print(f"   평균: {avg_gpu:.1f}%")
                print(f"   최대: {max_gpu:.1f}%")
                gpu_available = True
                
                if avg_gpu < 20:
                    print("   🚨 GPU 저사용! (평균 20% 미만) - CPU 병목 가능성 높음")
                elif avg_gpu < 50:
                    print("   ⚠️  GPU 사용률 낮음 (평균 50% 미만)")
                else:
                    print("   ✅ GPU 잘 활용됨")
        
        if 'gpu_memory_allocated_gb' in self.data:
            gpu_mem_values = [v for t, v in self.data['gpu_memory_allocated_gb']]
            if gpu_mem_values:
                avg_gpu_mem = sum(gpu_mem_values) / len(gpu_mem_values)
                max_gpu_mem = max(gpu_mem_values)
                print(f"\n🎮 GPU 메모리:")
                print(f"   평균: {avg_gpu_mem:.1f} GB")
                print(f"   최대: {max_gpu_mem:.1f} GB")
        
        # 병목 분석 및 권장사항
        print(f"\n💡 병목 분석 및 권장사항:")
        
        cpu_high = 'cpu_avg' in self.data and sum([v for t, v in self.data['cpu_avg']]) / len(self.data['cpu_avg']) > 70
        gpu_low = 'gpu_utilization' in self.data and sum([v for t, v in self.data['gpu_utilization']]) / len(self.data['gpu_utilization']) < 30
        
        if cpu_high and gpu_low:
            print("   🎯 CPU 병목 감지!")
            print("   📌 해결 방안:")
            print("      - 배치 크기 증가")
            print("      - 병렬 처리 스레드 수 증가")
            print("      - 데이터 로딩 최적화")
            print("      - 토크나이저 병렬화")
        elif cpu_high:
            print("   🎯 CPU 사용률 높음")
            print("   📌 CPU 최적화 필요")
        elif gpu_low and gpu_available:
            print("   🎯 GPU 저사용 감지")
            print("   📌 GPU 활용도 개선 필요")
        else:
            print("   ✅ 전반적인 성능 양호")
        
        print("="*60)

def simulate_cpu_intensive_task():
    """CPU 집약적 작업 시뮬레이션"""
    print("🔄 CPU 집약적 작업 시뮬레이션 중...")
    
    # 토크나이저 병목 시뮬레이션
    import time
    import threading
    
    def cpu_task():
        # 문자열 처리 시뮬레이션 (토크나이저와 유사)
        for i in range(100000):
            text = f"This is a sample text for processing {i}" * 10
            words = text.split()
            processed = " ".join(words[:50])
    
    # 여러 스레드로 병렬 처리
    threads = []
    num_threads = os.cpu_count() or 4
    
    start_time = time.time()
    for i in range(num_threads):
        t = threading.Thread(target=cpu_task)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    print(f"✅ CPU 작업 완료 (소요시간: {end_time - start_time:.2f}초)")

def simulate_gpu_task():
    """GPU 작업 시뮬레이션"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("⚠️  GPU 사용 불가능")
        return
    
    print("🔄 GPU 작업 시뮬레이션 중...")
    
    # 간단한 행렬 연산
    device = torch.device('cuda')
    
    start_time = time.time()
    for i in range(10):
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    end_time = time.time()
    print(f"✅ GPU 작업 완료 (소요시간: {end_time - start_time:.2f}초)")

def main():
    print("🚀 CPU 병목 분석 테스트")
    print("="*50)
    
    print(f"💻 시스템 정보:")
    print(f"   CPU 코어: {os.cpu_count()}")
    print(f"   메모리: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    if TORCH_AVAILABLE:
        print(f"   PyTorch: 사용 가능")
        if torch.cuda.is_available():
            print(f"   CUDA: 사용 가능 (GPU: {torch.cuda.get_device_name()})")
        else:
            print(f"   CUDA: 사용 불가능")
    else:
        print(f"   PyTorch: 사용 불가능")
    
    # 성능 모니터링 시작
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # CPU 집약적 작업 실행
        simulate_cpu_intensive_task()
        
        # 잠시 대기
        time.sleep(2)
        
        # GPU 작업 실행
        simulate_gpu_task()
        
        # 잠시 더 대기
        time.sleep(3)
        
    finally:
        # 모니터링 중지 및 결과 출력
        monitor.stop_monitoring()
        monitor.print_summary()

if __name__ == "__main__":
    main()
