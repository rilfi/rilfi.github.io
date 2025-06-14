---
title: "Essential CUDA Built-in Functions Every Developer Should Know"
categories: ["CUDA & Parallel Computing"]
---

# Essential CUDA Built-in Functions Every Developer Should Know

When programming with CUDA, you interact directly with the GPU's memory and resources using a set of powerful built-in functions. These functions help you allocate memory on the device, transfer data, manage device state, and more. In this article, we’ll explore the most commonly used CUDA Runtime API functions that every CUDA developer should know.

---

## 1. `cudaMalloc(void** devPtr, size_t size)`

### ➤ Purpose:
Allocates memory on the GPU device.

### ✅ Example:
```cpp
int *d_array;
cudaMalloc((void**)&d_array, 100 * sizeof(int));
````

### 🧠 Notes:

* Always check return values for debugging.
* Allocated memory must be manually freed using `cudaFree`.

---

## 2. `cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)`

### ➤ Purpose:

Copies memory between host and device.

### ✅ Example:

```cpp
int h_array[100], *d_array;
cudaMalloc(&d_array, 100 * sizeof(int));
cudaMemcpy(d_array, h_array, 100 * sizeof(int), cudaMemcpyHostToDevice);
```

### 🎯 Types of `cudaMemcpyKind`:

* `cudaMemcpyHostToDevice`
* `cudaMemcpyDeviceToHost`
* `cudaMemcpyDeviceToDevice`
* `cudaMemcpyHostToHost`

---

## 3. `cudaFree(void* devPtr)`

### ➤ Purpose:

Frees memory allocated on the GPU device.

### ✅ Example:

```cpp
cudaFree(d_array);
```

### ⚠️ Tip:

Always free device memory to avoid memory leaks.

---

## 4. `cudaMemset(void* devPtr, int value, size_t count)`

### ➤ Purpose:

Initializes/fills device memory with a constant byte value.

### ✅ Example:

```cpp
cudaMemset(d_array, 0, 100 * sizeof(int));
```

---

## 5. `cudaDeviceSynchronize()`

### ➤ Purpose:

Blocks the CPU until the GPU has completed all preceding tasks.

### ✅ Example:

```cpp
kernel<<<blocks, threads>>>();
cudaDeviceSynchronize();
```

### 🧠 Why Use It?

Ensures deterministic timing and debugging consistency.

---

## 6. `cudaDeviceReset()`

### ➤ Purpose:

Resets the GPU, releasing all allocated memory and state.

### ✅ Example:

```cpp
cudaDeviceReset();
```

### 📌 Note:

This is useful during cleanup or if you want to reinitialize the GPU state in long-running programs.

---

## 7. `cudaGetDeviceCount(int* count)`

### ➤ Purpose:

Returns the number of available CUDA-capable GPUs.

### ✅ Example:

```cpp
int count;
cudaGetDeviceCount(&count);
```

---

## 8. `cudaGetDeviceProperties(cudaDeviceProp* prop, int device)`

### ➤ Purpose:

Retrieves properties of a specific GPU.

### ✅ Example:

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Device Name: %s\n", prop.name);
```

---

## 9. `cudaSetDevice(int device)`

### ➤ Purpose:

Sets the active GPU device.

### ✅ Example:

```cpp
cudaSetDevice(0);
```

### 🧠 Useful For:

Multi-GPU systems where explicit control is needed.

---

## 10. `free(void* ptr)` (Standard C Function)

### ➤ Purpose:

Frees memory allocated on the CPU with `malloc()`.

### ⚠️ Important:

Do **not** use `free()` to release memory allocated with `cudaMalloc()`. Use `cudaFree()` instead.

---

## 🧩 Summary Table

| Function                  | Purpose                           | Applicable On |
| ------------------------- | --------------------------------- | ------------- |
| `cudaMalloc`              | Allocates GPU memory              | Device        |
| `cudaMemcpy`              | Copies memory between host/device | Both          |
| `cudaFree`                | Frees GPU memory                  | Device        |
| `cudaMemset`              | Sets GPU memory values            | Device        |
| `cudaDeviceSynchronize`   | Waits for GPU tasks to finish     | Host          |
| `cudaDeviceReset`         | Resets GPU device                 | Host          |
| `cudaGetDeviceCount`      | Gets number of CUDA devices       | Host          |
| `cudaGetDeviceProperties` | Retrieves GPU info                | Host          |
| `cudaSetDevice`           | Sets the active GPU device        | Host          |
| `free`                    | Frees host-allocated memory       | Host          |

---

## 🚀 Final Thoughts

Mastering these CUDA built-in functions forms the foundation of effective GPU programming. As you grow more advanced, you'll begin combining them with CUDA streams, asynchronous operations, unified memory, and more—but it all starts here.
