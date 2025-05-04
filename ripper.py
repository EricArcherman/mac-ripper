import torch
import torchvision.models as models
import time
import coremltools as ct
import numpy as np
from PIL import Image
import multiprocessing as mp
import psutil
import os
import platform
import sys
from typing import List, Optional
import subprocess

def get_temperature() -> Optional[float]:
    """Get CPU temperature if available on macOS."""
    try:
        result = subprocess.run(['osx-cpu-temp'], capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip().replace('°C', ''))
    except:
        pass
    return None

def cpu_task():
    try:
        x = 0
        while True:
            x = x ** 2 + 1
    except KeyboardInterrupt:
        pass

def benchmark_cpu_multithreaded(num_workers: int = None) -> List[mp.Process]:
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    print(f"🔧 Multithreaded CPU Stress Test ({num_workers} workers)...")
    processes = []
    try:
        for _ in range(num_workers):
            p = mp.Process(target=cpu_task)
            p.start()
            processes.append(p)
    except Exception as e:
        print(f"⚠️  Error starting CPU benchmark: {e}")
        for p in processes:
            p.terminate()
        raise
    return processes

def benchmark_gpu():
    if not torch.backends.mps.is_available():
        print("⚠️  MPS (GPU) not available. Skipping GPU benchmark.")
        return
    
    print("🖥️  GPU Stress Test (PyTorch MPS)...")
    try:
        # Test different model sizes
        models_to_test = {
            'ResNet18': models.resnet18(),
            'ResNet50': models.resnet50(),
            'EfficientNet-B0': models.efficientnet_b0(),
            'MobileNetV3': models.mobilenet_v3_small()
        }
        
        results = {}
        for name, model in models_to_test.items():
            print(f"\nTesting {name}...")
            model = model.to('mps').eval()
            
            # Test different batch sizes
            batch_sizes = [1, 4, 8, 16]
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, 3, 224, 224).to('mps')
                
                # Warmup
                print(f"  Warming up with batch size {batch_size}...")
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(dummy_input)
                
                # Benchmark
                print(f"  Benchmarking batch size {batch_size}...")
                with torch.no_grad():
                    t0 = time.perf_counter()
                    for _ in range(20):  # Reduced iterations but more batch sizes
                        _ = model(dummy_input)
                    t1 = time.perf_counter()
                
                avg_time = (t1 - t0) / 20
                results[f"{name} (batch={batch_size})"] = avg_time
                print(f"  ✅ {name} (batch={batch_size}): {avg_time:.3f}s per inference")
        
        # Print summary
        print("\n📊 GPU Benchmark Summary:")
        print("-" * 50)
        for name, avg_time in results.items():
            print(f"{name:<30}: {avg_time:.3f}s per inference")
        print("-" * 50)
        
    except Exception as e:
        print(f"⚠️  Error during GPU benchmark: {e}")

def benchmark_neural_engine():
    print("🧠 ANE Stress Test (Core ML)...")
    try:
        # Test different models
        models_to_test = {
            'MobileNetV2': models.mobilenet_v2(weights='DEFAULT'),
            'MobileNetV3': models.mobilenet_v3_small(weights='DEFAULT'),
            'EfficientNet-B0': models.efficientnet_b0(weights='DEFAULT')
        }
        
        results = {}
        for name, torch_model in models_to_test.items():
            print(f"\nTesting {name}...")
            torch_model = torch_model.eval()
            
            # Convert to Core ML
            print(f"  Converting {name} to Core ML...")
            example_input = torch.rand(1, 3, 224, 224)
            traced_model = torch.jit.trace(torch_model, example_input)
            
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.ImageType(name="input", shape=example_input.shape)],
                compute_units=ct.ComputeUnit.ALL
            )
            
            # Test with different batch counts (since we can't change input size)
            batch_counts = [1, 5, 10, 20]
            for batch_count in batch_counts:
                print(f"  Testing with {batch_count} sequential inferences...")
                img = np.random.rand(224, 224, 3).astype(np.float32)
                img = Image.fromarray((img * 255).astype(np.uint8))
                input_dict = {"input": img}
                
                # Warmup
                print(f"    Warming up...")
                for _ in range(5):
                    _ = mlmodel.predict(input_dict)
                
                # Benchmark
                print(f"    Benchmarking...")
                t0 = time.perf_counter()
                for _ in range(batch_count):
                    _ = mlmodel.predict(input_dict)
                t1 = time.perf_counter()
                
                avg_time = (t1 - t0) / batch_count
                results[f"{name} ({batch_count} inferences)"] = avg_time
                print(f"    ✅ {name} ({batch_count} inferences): {avg_time:.3f}s per inference")
        
        # Print summary
        print("\n📊 Neural Engine Benchmark Summary:")
        print("-" * 50)
        for name, avg_time in results.items():
            print(f"{name:<35}: {avg_time:.3f}s per inference")
        print("-" * 50)
        
    except Exception as e:
        print(f"⚠️  Error during Neural Engine benchmark: {e}")

def monitor_system(duration=30):
    print("📊 Monitoring system for 30 seconds...")
    print("Press Ctrl+C to stop monitoring")
    try:
        for _ in range(duration):
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            temp = get_temperature()
            
            print(f"🧠 CPU: {cpu:>5.1f}%  💾 RAM: {ram.percent:>5.1f}% ({ram.used/1024/1024/1024:.1f}GB/{ram.total/1024/1024/1024:.1f}GB)", end='')
            if temp is not None:
                print(f"  🌡️  Temp: {temp:.1f}°C")
            else:
                print()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def cleanup(processes: List[mp.Process]):
    print("\n🛑 Stopping CPU threads...")
    for p in processes:
        try:
            p.terminate()
            p.join(timeout=2)
        except:
            pass
    print("✅ Cleanup complete.")

if __name__ == "__main__":
    print(f"🧪 Running on: {platform.processor()} | {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")
    print("=" * 50)
    
    processes = []
    try:
        processes = benchmark_cpu_multithreaded()
        time.sleep(2)  # Give CPU threads a head start
        benchmark_gpu()
        benchmark_neural_engine()
        monitor_system()
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n⚠️  Error during benchmark: {e}")
    finally:
        cleanup(processes)
        print("✅ Done.")
