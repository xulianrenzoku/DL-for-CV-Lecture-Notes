Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 5: Neural Networks

**Where we are:**
1. Use **Linear Models** for image classification problems
2. Use **Loss Functions** to express preferences over different choices of weights
3. Use **Stochastic Gradient Descent** to minimize our loss functions and train the model

**Motivation**
- Problem: Linear Classifiers aren’t that powerful
    - Slide 5-6
    <img src='static/5-6.png' width='600'> 
- One solution: Feature Transforms
    - Slide 5-10
    - Transform to make points linearly separable
    <img src='static/5-10.png' width='600'> 
- Image Features
    - Color Histogram
    - Histogram of Oriented Gradients (HoG)
        - JJ: don’t want to go too much detail
    - Bag of Words (Data-Driven!) Slide 5-17
        - Step 1: Build codebook
            - Extract random patches
            - Cluster patches to form “codebook” of “visual words”
        - Step 2: Encode images
        <img src='static/5-17.png' width='600'> 
- Image Features
    - Slide 5-21
    - Feature Extraction stage
        - Ex: Bag of Words
    - Introduce Neural Networks
        - Motivation: We want to train all stages
    <img src='static/5-21.png' width='600'> 

**Neural Networks**
- Notation
    - Slide 5-23, 5-24, 5-26, 5-27
    - H, hidden layer
    - In practice we will usually add a learnable bias at each layer as well
    - Fully-connected neural network
        - Also “Multi-Layer Perceptron” (MLP)
    <img src='static/5-23.png' width='600'> 
    <img src='static/5-24.png' width='600'> 
    <img src='static/5-26.png' width='600'> 
    <img src='static/5-27.png' width='600'> 
- Neural net
    - Slide 5-28, 5-29
    - First layer is bank of templates; Second layer recombines templates
    - Can use different templates to cover multiple modes of a class
    - “Distributed representation”: Most templates not interpretable!
    <img src='static/5-28.png' width='600'> 
    <img src='static/5-29.png' width='600'> 
- Deep Neural Networks
    - Slide 5-32
    - Depth = number of layers
    - Width: Size of each layer
    <img src='static/5-32.png' width='600'> 
- Activation Functions
    - Slide 5-35
    - ReLU(z) = max(0, z) is called “Rectified Linear Unit”
    - This is called **activation function** of the neural network
    - Q: What happens if we build a neural network with no activation function?
        - S = W2W1x
        - A: We end up with a linear classifier
            - S = W2W1x = W3x
        - **So activation function is a non-linear function**
    <img src='static/5-35.png' width='600'> 
    
    - Slide 5-37
        - Sigmoid
        - Tanh
        - ReLU
        - Leaky ReLU
        - Maxout
        - ELU
    <img src='static/5-37.png' width='600'> 
- Neural Net in < 20 lines
    - Slide 5-42
    - Steps
        - Initialize weights and data
        - Compute loss (sigmoid activation, L2 loss)
        - Compute gradients
        - SGD step
    <img src='static/5-42.png' width='600'> 
- Biological Neuron v. Artificial Neuron
    - Slide 5-48, 5-49
    - Biological Neurons: Complex connectivity patterns
    - Neurons in a neural network: Organized into regular layers for computational efficiency
    - Be very careful with brain analogies!
    <img src='static/5-48.png' width='600'> 
    <img src='static/5-49.png' width='600'> 

**Good**
**Space Warping**
- Slide 5-54, 5-56
    - Consider a linear transform: h = Wx
        - Where x, h are both 2-dimensional
<img src='static/5-54.png' width='600'> 
<img src='static/5-56.png' width='600'> 

- Slide 5-61, 5-65
    - Consider a neural net hidden layer:
        - H = ReLu(Wx) = max(0, wX)
        - Where x, h are both 2-dimensional
<img src='static/5-61.png' width='600'> 
<img src='static/5-65.png' width='600'> 

- Setting the number of layers and their sizes
    - Slide 5-66
    - More hidden units = more capacity
    <img src='static/5-66.png' width='600'> 
- Don’t regularize with size; instead use strong L2
    - Slide 5-67
    <img src='static/5-67.png' width='600'> 

**Universal Approximation**
- A neural network with one hidden layer can approximate any function f: R^N —> R^M with arbitrary precision*
- Example: Approximating a function f: R—> R with a two-layer ReLU network
    - Output is a sum of shifted , scaled ReLUs 
        - Slide 5-73
    <img src='static/5-73.png' width='600'> 
    
    - We can build a “bump function” using four hidden units
        - Slide 5-79
    <img src='static/5-79.png' width='600'> 
    
    - With 4K hidden units we can build a sum of K bumps
        - Slide 5-81
        - Approximate any underlying functions by using a wider and wider neural network
        <img src='static/5-81.png' width='600'> 
        
    - Reality check: Networks don’t really learn bumps
- Realizations
    - Universal approximation tells us:
        - Neural nets an represent any function
    - Universal approximation DOES NOT tell us:
        - Whether we can actually learn any function with SGD
        - How much data we need to learn a function
    - Remember: kNN is also a universal approximator!

**Bad**
**How to optimize Neural Networks**
- Convex Functions
    - Example: f(x) = x^2 is convex
        - Slide 4-88
        <img src='static/5-88.png' width='600'> 
        
    - Example: f(x) = cos(x) is not convex
        - Slide 4-89
        <img src='static/5-89.png' width='600'> 
        
    - Intuition: A convex function is a (multidimensional) bowl
        - Generally speaking, convex functions are **easy to optimize**: can drive theoretical guarantees about **converging to global minimum**
            - Slide 4-91
         <img src='static/5-91.png' width='600'> 
         
        - Linear classifiers optimize a **convex function**! 
            - Slide 4-92
        <img src='static/5-92.png' width='600'> 
        
        - Neural net losses sometimes look convex-ish
            - Slide 4-93
        <img src='static/5-93.png' width='600'> 
        
        - But often clearly nonconvex
            - Slide 4-94
        <img src='static/5-94.png' width='600'> 
        
        - With local minima
            - Slide 4-95
        <img src='static/5-95.png' width='600'> 
        
    - Most neural networks need nonconvex optimization
        - Few or no guarantees about convergence
        - Empirically it seems to work anyway
        - Active area of research
        - JJ: Hot topic of research community

**Summary**
- Feature transform + Linear classifier allows nonlinear decision boundaries
- Neural Networks as learnable feature transforms
- From linear classifiers to fully-connected networks
