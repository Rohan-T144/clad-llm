# Enhancing LLM Training Efficiency with Clad for Automatic Differentiation in C++

**GSoC 2024 Project under CERN-HSF**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Project Overview

Modern LLM training often relies on Python-based deep learning frameworks. While flexible, these frameworks can introduce performance bottlenecks due to interpreted execution and dynamic computation graphs.  This project explores a C++-centric approach, using Clad, a Clang-based Automatic Differentiation (AD) plugin, to perform automatic differentiation at the compiler level. This has the potential to reduce computational overhead, improve memory efficiency, and lead to more performant and resource-conscious LLM training workflows.

At the heart of this project is **`cladtorch`**, a custom C++ tensor library being developed specifically for this purpose. Inspired by minimalist projects like `llm.c` and `tinytorch`, `cladtorch` is designed to be Clad-friendly, enabling seamless integration of compiler-based AD into LLM training.

## Methodology (High-Level)

1.  **Baseline Implementation:** First, a complete GPT-2 training setup will be implemented in C++ using `cladtorch` but without Clad for automatic differentiation. This baseline will use manual AD gradient computation.
2.  **Clad Integration Strategy Exploration:** Different strategies for integrating Clad will be investigated, focusing on how to bridge the gap between Clad's static analysis and the dynamic nature of neural networks.
3.  **Incremental Clad Integration:** Based on the chosen strategy, Clad will be integrated incrementally, starting with core operations and gradually extending to more complex layers of the GPT-2 model.
4.  **Benchmarking and Optimization:** Performance will be rigorously benchmarked at each stage to address bottlenecks and assess the changes.

