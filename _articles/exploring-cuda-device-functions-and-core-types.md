---
title: "Exploring CUDA `__device__` Functions and Core Types on NVIDIA GPUs"
categories: ["CUDA & Parallel Computing"]
layout: default
---

# Exploring CUDA `__device__` Functions and Core Types on NVIDIA GPUs

When programming with CUDA, understanding where and how code executes is critical to writing efficient and performant GPU applications. One of the most frequently used CUDA keywords is `__device__`, which indicates that a function runs on the GPU and is callable only from other GPU functions. But what exactly happens when such code runs? What types of GPU cores handle it? Let’s break it down.

---

## 🧠 What Is a `__device__` Function?

In CUDA, the `__device__` keyword specifies that a function:

- **Executes on the GPU**
- **Is called only from GPU code** (like other `__device__` or `__global__` functions)

These functions are not directly callable from CPU-side host code.

Example:
```cpp
__device__ int square(int x) {
    return x * x;
}
````

You can call `square()` from another GPU function like:

```cpp
__global__ void computeSquare(int *d_out, int *d_in) {
    int idx = threadIdx.x;
    d_out[idx] = square(d_in[idx]);
}
```

---

## 🔄 How `__device__` Code Is Executed

When a kernel launches (i.e., a `__global__` function), each GPU thread executes a copy of that function. If it calls a `__device__` function, that function executes **within the same thread context**. It’s like an inline subroutine.

These GPU threads are grouped into:

* **Warps**: 32 threads that execute instructions in lockstep.
* **Thread blocks**: Groups of warps (usually up to 1024 threads).
* **Grids**: Collections of thread blocks.

The execution happens on the GPU's **Streaming Multiprocessors (SMs)** using its compute units.

---

## 🔧 CUDA Core Types and Where `__device__` Code Runs

NVIDIA GPUs contain various types of specialized cores:

### 1. **FP32 CUDA Cores**

* Handle 32-bit floating-point operations.
* Most typical `__device__` arithmetic like `+`, `-`, `*`, `/` run here.

### 2. **INT32 Cores**

* Deal with 32-bit integer operations.
* Execute functions that rely on integer arithmetic (`mod`, bitwise ops, counters).

### 3. **Tensor Cores**

* Used for matrix operations and deep learning.
* Activated only with specific libraries (like cuBLAS, cuDNN, or `wmma` intrinsics).
* Not typically used for custom `__device__` code unless optimized.

### 4. **Special Function Units (SFUs)**

* Handle `sin`, `cos`, `exp`, `log`, etc.
* When your `__device__` code includes these, it offloads to SFUs.

### 5. **Load/Store Units**

* Move data between registers and shared/global memory.
* `__device__` code that reads/writes memory uses these units.

---

## 📍 Where Is My Code Running?

When you write `__device__` code, it's compiled to run on the **GPU's streaming multiprocessors (SMs)**. The SM schedules and dispatches your thread instructions to available CUDA cores based on:

* Instruction type (float/int/SFU)
* Resource availability
* Warp scheduling policies

---

## 🧩 Optimization Tips

* Avoid unnecessary `__device__` function calls in critical paths.
* Use `__forceinline__` to force the compiler to inline small `__device__` functions.
* Keep memory access patterns coalesced inside `__device__` functions for performance.
* Profile your kernels with Nsight to see which core types are bottlenecks.

---

## ✅ Summary

| CUDA Keyword | Runs On | Callable From | Core Types Involved        |
| ------------ | ------- | ------------- | -------------------------- |
| `__device__` | GPU     | GPU only      | FP32, INT32, SFU, LS Units |
| `__global__` | GPU     | Host (CPU)    | Same as above              |
| `__host__`   | CPU     | CPU only      | CPU cores only             |

Understanding the internals of CUDA execution and the roles of different core types helps you write faster and more reliable GPU code.


