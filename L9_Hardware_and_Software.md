Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 9: Hardware and Software
    
**DL Hardware**
- Inside a computer
    - CPU: “Central Processing Unit”
    - GPU: “Graphics Processing Unit”
        - NVIDIA vs. AMD
            - DL clear winner: NVIDIA
        - Slide 9-11: GigaFLOPs per Dollar
        <img src='static/9-11.png' width='600'> 
    - CPU vs GPU
        - CPU: Fewer cores, but each core is much faster and much more capable; great at sequential tasks
        - GPU: More cores, but each core is much slower and “dumber”; great for parallel tasks
- Inside a GPU
    - RTX Titan
        - 12 * 2GB memory modules
        - Processor
            - 72 Streaming multiprocessors (SMs)
                - 64 FP32 cores per SM
                - 8 Tensore Core per SM
- **Tensor Core**: Special hardware!
    - Slide 9-21, 9-22
    <img src='static/9-21.png' width='600'> 
    <img src='static/9-22.png' width='600'> 
    
    - Slide 9-24: GigaFLOPs per Dollar
    <img src='static/9-24.png' width='600'> 
- Example: Matrix Multiplication
    - Slide 9-25
        - Perfect for GPUs! All output elements are independent, can be trivially parallelized
    <img src='static/9-25.png' width='600'> 
- Programming GPUs
    - CUDA (NVIDIA only)
        - Write C-like code that runs directly on the GPU
        - NVIDIA provides optimized APIs: cuBLAS, cuFFT, cu DNN, etc
    - OpenCL
        - Similar to CUDA, but runs on anything
        - Usually slower on NVIDIA hardware
    - EECS 598.009: Applied GPU Programming
- Scaling up: Typically 8 GPUs per server
- Google Tensor Processing Units (TPU)
    - Special hardware for matrix multiplication, similar to NVIDIA Tensor Cores; also runs in mixed precision (bfloat16)
    - Cloud TPU v2: $4.50/hour
    - Cloud TPU v2 Pod: $384/hour
    - Cloud TPU v3: $8/hour
    - Cloud TPU v3 Pod: Talk to a sales rep
    - In order to use TPUs, you have to use TensorFlow (for now)

**DL Software**
- A zoo of frameworks!
    - Caffe(UC Berkley) —> Caffe2(Facebook)
    - Torch(NYU/Facebook) —> PyTorch(Facebook)
    - Theno(U Montreal) —> TensorFlow(Google)
    - JAX(Google)
- Mainstream frameworks
    - PyTorch
    - TensorFlow
- The point of DL frameworks
    1. Allow rapid prototyping of new ideas
    2. Automatically compute gradients for you
    3. Run it all efficiently on GPU (or TPU)
- PyTorch
    - Versions
        - For this class we are using **PyTorch version 1.2**
        - Be careful if you are looking at older PyTorch code
    - Fundamental Concepts
        - **Tensor**: Like a numpy array, but can run on GPU
        - **Autograd**: Package for building computational graphs out of Tensors, and automatically computing gradients
        - **Module**: A neural network layer; may store state or learnable weights
    - Tensors
        - Running example: Train a two-layer ReLU network on random data with L2 loss
        - Slide 9-44: Create random tensors for data and weights 
        <img src='static/9-44.png' width='600'> 
        
        - Slide 9-45: Foward pass: compute predictions and loss
        <img src='static/9-45.png' width='600'> 
        
        - Slide 9-46: Backward pass: manually compute gradients
        <img src='static/9-46.png' width='600'> 
        
        - Slide 9-47: Gradient descent step on weights
        <img src='static/9-47.png' width='600'> 
        
        - Slide 9-48: To run on GPU, just use a different device!
        <img src='static/9-48.png' width='600'> 
    - Autograd
        - Slide 9-49: Creating Tensors with requires_grad=True enables autograd
        <img src='static/9-49.png' width='600'> 
        
        - Slide 9-50
        <img src='static/9-50.png' width='600'> 
        
        - Slide 9-51: Forward pass
        <img src='static/9-51.png' width='600'>
        
        - Slide 9-52: Backward pass
        <img src='static/9-52.png' width='600'>
        
        - Slide 9-53: mm
        <img src='static/9-53.png' width='600'>
        
        - Slide 9-54: clamp
        <img src='static/9-54.png' width='600'>
        
        - Slide 9-55: mm
        <img src='static/9-55.png' width='600'>
        
        - Slide 9-56: pred
        <img src='static/9-56.png' width='600'>
        
        - Slide 9-57: pow
        <img src='static/9-57.png' width='600'>
        
        - Slide 9-58: sum
        <img src='static/9-58.png' width='600'>
        
        - Slide 9-59: Backprop to all inputs that require grad
        <img src='static/9-59.png' width='600'>
        
        - Slide 9-60 
            - After backward finishes, gradients are **accumulated** into w1.grad and w2.grad and the graphs is destroyed 
        <img src='static/9-60.png' width='600'>
        
        - Slide 9-61: Make gradient step on weights
        <img src='static/9-61.png' width='600'>
        
        - Slide 9-62: Set gradients to zero - forgetting this is a common bug!
        <img src='static/9-62.png' width='600'>
        
        - Slide 9-63: Tell PyTorch not to build a graph for these operations
        <img src='static/9-63.png' width='600'>
    - Autograd: Sigmoid
        - Slide 9-65
            - Can define new  operations using Python functions
                - JJ: Can be numerically unstable
        <img src='static/9-65.png' width='600'>
        
        - Slide 9-67
            - Use the PyTorch version
        <img src='static/9-67.png' width='600'>
        
        - In practice this is pretty rare - in most cases Python functions are good enough
    - nn Module
        - Slide 9-69
            - Higher-level wrapper for working with neural nets
            - Use this! It will make your life easier
        <img src='static/9-69.png' width='600'>
        
        - Slide 9-70: Object-orientd API
        <img src='static/9-70.png' width='600'>
        
        - Slide 9-71: Forward pass
        <img src='static/9-71.png' width='600'>
        
        - Slide 9-72: Loss functions
        <img src='static/9-72.png' width='600'>
        
        - Slide 9-73: Backward pass
        <img src='static/9-73.png' width='600'>
        
        - Slide 9-74: Make gradient step on each model parameter (w/ gradients disabled)
        <img src='static/9-74.png' width='600'>
    - optim Module
        - Slide 9-75: Use an **optimizer** for different update rules
        <img src='static/9-75.png' width='600'>
        
        - Slide 9-76: After computing gradients, use optimizer to update and zero gradients
        <img src='static/9-76.png' width='600'>
    - Define your own nn Module
        - Slide 9-77: Very common to define your own models or layers as custom Modules 
        <img src='static/9-77.png' width='600'>
        
        - Slide 9-78
        <img src='static/9-78.png' width='600'>
        
        - Slide 9-79
        <img src='static/9-79.png' width='600'>
        
        - Slide 9-80
            - No need to define backward - autograd will handle it
        <img src='static/9-80.png' width='600'>
        
        - Slide 9-81: Very common to mix and match custom Module subclasses and Sequential containers
        <img src='static/9-81.png' width='600'>
        
        - Slide 9-82
            - JJ: Not a practical example
        <img src='static/9-82.png' width='600'>
        
        - Slide 9-83: Very easy to quickly build complex network architectures
        <img src='static/9-83.png' width='600'>
    - DataLoaders
        - Slide 9-84
            - A **DataLoader** wraps a **Dataset** and provides minibatching, shuffling, multithreading for you
            - When you need to load custom data, jsut write your own Dataset class
        <img src='static/9-84.png' width='600'>
    - Pretrained Models: Super easy to use pretrained models with torchvision
        - Slide 9-87
        <img src='static/9-87.png' width='600'>
    - Dynamic Computation Graphs
        - Slide 9-97
            - Dynamic graphs let you use regular Python control flow during the forward pass!
        <img src='static/9-97.png' width='600'>
    - Alternative: Static Computation Graphs
        - Slide 9-100
        <img src='static/9-100.png' width='600'>
    - Static Graphs with JIT
        - Slide 9-102: Lots of magic
        <img src='static/9-102.png' width='600'>
    - Static(for) vs Dynamic Graphs: Optimization
        - Slide 9-106
        <img src='static/9-106.png' width='600'>
    - Static(for) vs Dynamic Graphs: Serialization
        - Slide 9-107
        <img src='static/9-107.png' width='600'>
    - Static(downside) vs Dynamic Graphs: Debugging
        - Slide 9-108
        <img src='static/9-108.png' width='600'>
    - Dynamic Graph Applications
        - Model structure depends on the input
            - Recurrent Networks: Slide 9-109
            <img src='static/9-109.png' width='600'>
            
            - Recursive Networks: Slide 9-110
            <img src='static/9-110.png' width='600'>
            
            - Modular Networks: Slide 9-111
            <img src='static/9-111.png' width='600'>
- TensorFlow
    - Versions
        - TensorFlow 1.0
            - Default: static graphs
        - TensorFlow 2.0
            - Default: dynamic graphs
