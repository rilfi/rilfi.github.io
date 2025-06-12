# **100 Short Questions & Answers on GPU Architecture (Threads, Warps, Blocks, Grids, SMs, Memory)**  

## **1. Basics of GPU Execution Model**  
**Q1. What is a thread in GPU programming?**  
A: The smallest execution unit that runs a kernel function.  

**Q2. How many threads make a warp in NVIDIA GPUs?**  
A: 32 threads.  

**Q3. What is a thread block?**  
A: A group of threads (e.g., 256–1024) that execute together on an SM.  

**Q4. What is a grid in CUDA?**  
A: A collection of thread blocks that execute a kernel.  

**Q5. What is an SM in a GPU?**  
A: Streaming Multiprocessor—the core processing unit that executes warps.  

**Q6. What is a kernel in CUDA?**  
A: A GPU function launched from the CPU.  

**Q7. How do threads in a warp execute?**  
A: In lockstep (SIMD—Single Instruction, Multiple Data).  

**Q8. What happens if threads in a warp diverge (e.g., `if-else`)?**  
A: Performance drops due to **warp divergence**.  

**Q9. What is the maximum number of threads per block in modern GPUs?**  
A: 1024 (NVIDIA A100/H100).  

**Q10. What is the smallest possible block size?**  
A: 1 thread (but highly inefficient).  

---

## **2. Warps & Thread Scheduling**  
**Q11. Why are warps important in GPU execution?**  
A: They allow efficient SIMD execution.  

**Q12. Can a warp execute different instructions?**  
A: No, all threads in a warp execute the same instruction.  

**Q13. What is warp shuffling?**  
A: A technique where threads in a warp exchange data directly.  

**Q14. How many warps can an SM run concurrently?**  
A: 64 warps (2048 threads) on A100/H100.  

**Q15. What is occupancy in GPU programming?**  
A: The ratio of active warps to maximum possible warps per SM.  

**Q16. How can you check occupancy in CUDA?**  
A: Using `cudaOccupancyMaxPotentialBlockSize`.  

**Q17. What limits warp occupancy?**  
A: Register usage, shared memory, and block size.  

**Q18. What is a "zero-overhead" warp scheduler?**  
A: A GPU feature that switches warps without CPU overhead.  

**Q19. What is the ideal block size for maximizing occupancy?**  
A: 256–1024 threads (multiples of 32).  

**Q20. Can warps from different blocks communicate?**  
A: No, warps only sync within a block.  

---

## **3. Thread Blocks & Grids**  
**Q21. What is the maximum number of blocks per SM?**  
A: 32 (A100/H100).  

**Q22. How is a block assigned to an SM?**  
A: The GPU scheduler dynamically assigns blocks to free SMs.  

**Q23. Can blocks be rescheduled during execution?**  
A: No, once assigned, they run to completion.  

**Q24. What happens if there are more blocks than SMs?**  
A: Blocks wait in a queue until an SM is free.  

**Q25. What is the maximum number of blocks per grid?**  
A: ~2 billion (2³¹ - 1).  

**Q26. How many grids can run simultaneously on a GPU?**  
A: One kernel (grid) at a time, but multiple can be queued.  

**Q27. How do you calculate grid dimensions in CUDA?**  
A: `dim3 grid((N + block_size - 1) / block_size, 1);`  

**Q28. What is a grid-stride loop?**  
A: A loop where threads process multiple elements to handle large datasets.  

**Q29. Can grids synchronize with each other?**  
A: No, grids are independent.  

**Q30. What is the maximum grid size in CUDA?**  
A: 2³² - 1 threads (~4 billion).  

---

## **4. Memory Hierarchy**  
**Q31. What is the fastest memory in a GPU?**  
A: Registers (per-thread).  

**Q32. What is shared memory?**  
A: A fast, programmable cache shared by threads in a block.  

**Q33. How much shared memory does A100 have per SM?**  
A: 164 KB (configurable with L1).  

**Q34. What is global memory?**  
A: The GPU’s main memory (high-latency, high-bandwidth).  

**Q35. What is constant memory?**  
A: A read-only cache optimized for broadcast reads.  

**Q36. What is L1 cache used for?**  
A: Caching local and global memory accesses.  

**Q37. What is L2 cache used for?**  
A: Reducing global memory latency (shared across all SMs).  

**Q38. What is the L2 cache size in A100?**  
A: 40 MB.  

**Q39. What is register spilling?**  
A: When a thread runs out of registers and uses slower local memory.  

**Q40. How do you optimize memory access in CUDA?**  
A: Use **coalesced memory access** and shared memory.  

---

## **5. NVIDIA A100 vs. H100**  
**Q41. How many SMs does A100 have?**  
A: 108.  

**Q42. How many SMs does H100 have?**  
A: 144.  

**Q43. What is the max threads per SM in A100?**  
A: 2048.  

**Q44. What is the max threads per SM in H100?**  
A: 2048.  

**Q45. What is the total thread capacity of A100?**  
A: 108 × 2048 = 221,184 threads.  

**Q46. What is the total thread capacity of H100?**  
A: 144 × 2048 = 294,912 threads.  

**Q47. Does H100 have more shared memory than A100?**  
A: Yes (228 KB vs. 164 KB).  

**Q48. What is the L2 cache size in H100?**  
A: 50 MB (vs. 40 MB in A100).  

**Q49. Does H100 support more blocks per SM than A100?**  
A: No, both support 32 blocks/SM.  

**Q50. Which GPU has higher memory bandwidth?**  
A: H100 (~3 TB/s vs. A100’s ~2 TB/s).  

---

## **6. Advanced Concepts & Optimization**  
**Q51. What is CUDA’s execution model called?**  
A: SIMT (Single Instruction, Multiple Threads).  

**Q52. What is a CUDA core?**  
A: A single floating-point unit inside an SM.  

**Q53. How many CUDA cores does A100 have?**  
A: 6,912 (108 SMs × 64 FP32 cores/SM).  

**Q54. How many CUDA cores does H100 have?**  
A: 18,432 (144 SMs × 128 FP32 cores/SM).  

**Q55. What is dynamic parallelism?**  
A: A feature where a kernel launches another kernel.  

**Q56. What is warp voting?**  
A: A mechanism where threads in a warp perform collective operations.  

**Q57. What is `__syncthreads()` used for?**  
A: Synchronizing all threads in a block.  

**Q58. What is `__shfl_sync` used for?**  
A: Exchanging data between threads in a warp.  

**Q59. What is Tensor Core?**  
A: Specialized hardware for matrix operations (AI/ML).  

**Q60. Does H100 have more Tensor Cores than A100?**  
A: Yes, 4th-gen vs. 3rd-gen.  

---

## **7. Performance & Debugging**  
**Q61. What is the biggest bottleneck in GPU programming?**  
A: Memory bandwidth.  

**Q62. How do you measure GPU performance?**  
A: Using **FLOPs (Floating-Point Operations per Second)**.  

**Q63. What is the peak FP32 performance of A100?**  
A: 19.5 TFLOPS.  

**Q64. What is the peak FP32 performance of H100?**  
A: 60 TFLOPS (with Hopper).  

**Q65. What is `nvprof`?**  
A: NVIDIA’s profiler for GPU applications.  

**Q66. What is `nsight-compute`?**  
A: A tool for analyzing kernel performance.  

**Q67. How do you avoid bank conflicts in shared memory?**  
A: By ensuring threads access different memory banks.  

**Q68. What is a memory coalesced access pattern?**  
A: When threads read contiguous memory locations.  

**Q69. What is the penalty for uncoalesced memory access?**  
A: Reduced bandwidth utilization.  

**Q70. How do you debug GPU kernels?**  
A: Using `printf` in CUDA or `cuda-gdb`.  

---

## **8. Programming Best Practices**  
**Q71. Should block size be a multiple of 32?**  
A: Yes (to avoid underutilized warps).  

**Q72. What is the best block size for matrix multiplication?**  
A: 16×16 (256 threads).  

**Q73. How do you handle large datasets in CUDA?**  
A: Using grid-stride loops.  

**Q74. Should you use shared memory for global data reuse?**  
A: Yes, to reduce global memory traffic.  

**Q75. What is the best way to reduce warp divergence?**  
A: Restructuring branches to minimize divergence.  

**Q76. Should you use `__restrict__` in CUDA?**  
A: Yes, to help the compiler optimize memory access.  

**Q77. What is `__ldg` used for?**  
A: Reading data through the read-only cache.  

**Q78. What is `__launch_bounds__`?**  
A: A hint to the compiler about kernel occupancy.  

**Q79. Should you use `atomicAdd` in CUDA?**  
A: Only when necessary (slow due to serialization).  

**Q80. What is the best way to initialize GPU memory?**  
A: Using `cudaMemset` or `cudaMemcpy`.  

---

## **9. Real-World GPU Comparisons**  
**Q81. How does A100 compare to V100?**  
A: A100 has more SMs (108 vs. 80) and Tensor Cores.  

**Q82. How does H100 improve over A100?**  
A: More SMs (144), faster memory, and 4th-gen Tensor Cores.  

**Q83. What is the memory bandwidth of A100?**  
A: ~2 TB/s (with HBM2e).  

**Q84. What is the memory bandwidth of H100?**  
A: ~3 TB/s (with HBM3).  

**Q85. Can A100 and H100 run the same CUDA code?**  
A: Yes, but H100 can optimize further with new features.  

**Q86. Does H100 support PCIe 5.0?**  
A: Yes (vs. PCIe 4.0 in A100).  

**Q87. What is NVLink, and is it faster in H100?**  
A: A high-speed GPU interconnect (900 GB/s in H100 vs. 600 GB/s in A100).  

**Q88. Does H100 support FP8 precision?**  
A: Yes (useful for AI workloads).  

**Q89. What is the TDP (power consumption) of A100?**  
A: 400W (PCIe) / 700W (SXM).  

**Q90. What is the TDP of H100?**  
A: 700W (SXM5).  

---

## **10. Future of GPU Computing**  
**Q91. What is CUDA’s main competitor?**  
A: AMD ROCm / HIP.  

**Q92. Will future GPUs have more SMs?**  
A: Yes (e.g., NVIDIA’s next-gen Blackwell).  

**Q93. Will AI replace CUDA programming?**  
A: No, but AI-assisted optimization tools will help.  

**Q94. What is the future of GPU memory?**  
A: HBM4 (faster, higher capacity).  

**Q95. Will quantum computing replace GPUs?**  
A: Not soon—GPUs will remain dominant for parallel workloads.  

**Q96. What is the biggest challenge in GPU programming?**  
A: Efficient memory management.  

**Q97. Will CUDA support more languages?**  
A: Likely (Python/C++ integration is growing).  

**Q98. What is the next big GPU architecture after Hopper?**  
A: Blackwell (expected 2024).  

**Q99. Will GPUs replace CPUs?**  
A: No, they complement each other (CPU for serial, GPU for parallel).  

**Q100. What’s the best way to learn GPU programming?**  
A: Start with CUDA C/C++ and experiment with real-world projects!  

---

### **Final Thoughts**  
This Q&A covers **threads, warps, blocks, grids, SMs, memory, and optimization** in modern GPUs. Key takeaways:  
- **A100**: 108 SMs, 221K threads, 40MB L2 cache.  
- **H100**: 144 SMs, 295K threads, 50MB L2 cache.  
- **Optimize** with shared memory, coalesced access, and proper block sizing.  

🚀 **Now go write efficient CUDA code!**
