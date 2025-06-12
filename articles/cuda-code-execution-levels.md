### 📘 Article: *Understanding Execution Levels in CUDA: Host, Device, and Global Explained*

CUDA (Compute Unified Device Architecture) enables general-purpose computing on GPUs using C/C++ extensions. One of the core concepts in CUDA programming is understanding **where your code is executed** and **from where it is invoked**. CUDA provides different levels of execution that help optimize performance by distributing work efficiently between the CPU (host) and the GPU (device).

Let’s explore the different execution levels in CUDA: `__host__`, `__device__`, and `__global__`.

---

## 🚀 Execution Qualifiers in CUDA

CUDA introduces specific function qualifiers to indicate the execution level:

### 1. `__host__`

* **Purpose**: Specifies that the function runs on the **host (CPU)**.
* **Default Behavior**: If no qualifier is given, the function is implicitly a host function.
* **Usage**: Typically used for high-level orchestration logic and CPU-based tasks.
* **Example**:

  ```cpp
  __host__ void printHello() {
      printf("Hello from the host!\n");
  }
  ```

### 2. `__device__`

* **Purpose**: Marks a function that runs on the **device (GPU)** and can **only be called from other device or global functions**.
* **Cannot be called from the CPU**.
* **Usage**: Helper functions or utility routines within a kernel.
* **Example**:

  ```cpp
  __device__ int square(int x) {
      return x * x;
  }
  ```

### 3. `__global__`

* **Purpose**: Specifies a kernel function that runs on the **GPU** and is **called from the host (CPU)**.
* **Returns void only.**
* **Invoked using CUDA kernel launch syntax: `<<<blocks, threads>>>`.**
* **Usage**: Main GPU kernels launched by the host code.
* **Example**:

  ```cpp
  __global__ void addVectors(int *a, int *b, int *c, int n) {
      int i = threadIdx.x;
      if (i < n)
          c[i] = a[i] + b[i];
  }
  ```

---

## 🧠 Combining Qualifiers

You can combine `__host__` and `__device__` to make a function usable in both environments:

```cpp
__host__ __device__ int multiply(int a, int b) {
    return a * b;
}
```

This is useful when you want the same logic to work both on the CPU and GPU without rewriting code.

---

## 🔄 Summary Table

| Qualifier    | Runs On | Called From | Return Type |
| ------------ | ------- | ----------- | ----------- |
| `__host__`   | CPU     | CPU         | Any         |
| `__device__` | GPU     | GPU         | Any         |
| `__global__` | GPU     | CPU         | void        |

---

## 🎯 Final Thoughts

Understanding where your functions execute and how they interact across the host-device boundary is key to writing efficient CUDA programs. Use these qualifiers wisely to structure your code in a way that balances the CPU’s orchestration capabilities with the GPU’s massive parallelism.

