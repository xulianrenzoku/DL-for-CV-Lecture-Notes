# DL-for-CV-Lecture-Notes
Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson


**Lecture 1: [Intro](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L1_Intro.md)**  

**Lecture 2: [Image Classification](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L2_Image_Classification.md)**  
- Image Classification: 
  - A core computer vision task
  - A building block for other vision tasks
- Image Classifier:
  - Hard-code: Teach the machine
  - Machine Learning: Feed the machine
    - Data-Driven Approach
- Classification Algorithms
  - Nearest Neighbor
  - K-Nearest Neighbor
    - Hyperparameters
      - Choose hyperparameters using the val set
      - Only run on the test set once at the very end
    - Universal Approximation
  - Nearest Neighbor with ConvNet features works well

**Lecture 3: [Linear Classifiers](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L3_Linear_Classifiers.md)**  
- Linear Classifiers
  - Building blocks of neural network
  - Parametric Approach: f(x, W) = Wx + b
- How to choose W?
  - Use a loss function to quantify how good value
  - Find a W that minimizes the loss function (*Lecture 4: Optimization*)
- Loss Function
  - A loss function tells how good our current classifier is
  - Multiclass SVM Loss
  - Regularization
    - L(W) = Data loss + Regularization
    - Purpose of regularization
  - Cross-Entropy Loss (Multinomial Logistic Regession)
    - Interpret raw classifier scores as probabilities
    - Introduce softmax function

**Lecture 4: [Optimization](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L4_Optimization.md)**  
- Purpose of optimization
  - Find the best W
    - w* = argmin L(w)
    - Imagine to walk towards bottom of a high-dimensional landscape
- How to minimize?
  - Idea #1: Random Search (BAD)
  - Idea #2: Follow the slope
- Gradient Descent 
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
    - Intuition
    - Problems
  - SGD + Momentum
    - Intuition of Momentum
    - First moment
  - AdaGrad
    - Sum of squares
    - Second moment
    - RMSProp
  - Adam: RMSProp + Momentum 
    - Good default choice
    - Bias Correction

**Lecture 5: [Neural Networks](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L5_Neural_Networks.md)**  
- Motivation
  - Problem: Linear Classifiers aren’t that powerful
  - One Solution: Feature Transforms
  - Feature transform + Linear classifier allows nonlinear decision boundaries
- Neural Networks: Learnable Feature Transforms
  - Notation
    - Hidden layer
    - Fully-connected neural network
  - Neural Net (2-layer)
    - First layer is bank of templates; Second layer recombines templates
  - Deep Neural Networks
  - Activation Functions
    - Non-linear functions
    - ReLU, Sigmoid...
    - Q&A
      - Q: What happens if we build a neural network with no activation function?
      - A: We end up with a linear classifier 
- Properties
  - Space Warping
  - Universal Approximation
  - Convex Functions (How to optimize neural networks)
    - Convex functions are easy to optimize
    - Most neural networks need nonconvex optimization

**Lecture 6: [Backpropagation](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L6_Backpropagation.md)**  
- How to compute gradients?
  - (BAD) Idea: Derive on paper
  - Better Idea: Computational Graphs
- Backpropagation (with scalars)
  - Used to compute gradients of computational graphs
    - Forward pass: Compute outputs
    - Backward pass: Compute derivatives/gradients
  - Useful because it is modular
  - Multiple ways to construct backward pass. Choose the simple one
    - Sigmoid example
  - Patterns in Gradient Flow
  - Implementations
    - "flat" gradient code 
      - backward pass looks like forward pass reversed
      - Problem: Not modular
    - Modular API (a set of paired forward/backward functions)
- Backpropagation (vector-valued)
  - Local Jacobian matrices
  - Never explicitly form Jacobian; instead use implicit multiplication
- Backprop with Matrices (or Tensors)
  - "local graident slice" approach
  - Weight matrices
    

**Lecture 7: [Convolutional Networks](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L7_Convolutional_Networks.md)**  
- Convolutional Network
  - Ingredients of a Full-Connected Network
    - Fully-Connected Layers
    - Activation Function
  - Need Three New Components to Build ConvNets
    - Convolution Layers
    - Pooling Layers
    - Normalization
- Convolutional Layer
  - Components
    - Input 
    - Filter
      - Convovle the filter with the image
    - Activation map
    - Non-linear activation function
  - Problem: Feature maps "shrink" with each layer
    - Padding: Add zeros around the input
  - Problem:  For large images we need many layers for each output to “see” the whole image
    - Stride: downsample inside the network
- Pooling Layers
  - Another way to downsample
  - Max Pooling 
    - Another non-linear function technically
    - Why do we choose pooling over strides?
- Normalization
  - Batch Normalization
  - Batch Normalization for ConvNets
  - Structure: Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity
  - Properties
    - Good: Easier, faster, more robust/regularization/zero overhead at test-time 
    - Bad


**Lecture 8: [CNN Architectures](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L8_CNN_Architectures.md)**  
- AlexNet(2012)
  - Architecture
  - Interesting trends
- ZFNet(2013): A Bigger AlexNet
- VGG(2014): Deeper Networks, Regular Design
  - Design rules
- GoogLeNet(2014): Focus on Efficiency
  - Aggresive Stem (downsample)
  - Inception Module
  - Gloabl Average Pooling
  - Auxiliary Classifiers
- ResNet(2015)
  - Batch normalization allows researchers to go deep
  - Architecture
    - Residual Block
    - Bottleneck Block 
  - Takeaway
    - Able to train very deep networks
    - Deeper networks do better than shallow networks (as expected)
    - Still widely used today
- ResNeXt
- Densely Connected Neural Networks
- MobileNets: Tiny Networks (For Mobile Devices)
  - Sacrifice accuracy for low computational cost
- Neural Architecture Search
  - Automate designing
  - Can be used to find efficient CNN architectures
  
  
**Lecture 9: [Hardware and Software](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main//L9_Hardware_and_Software.md)** 
- Deep Learning Hardware
  - Inside a computer: CPU & GPU
  - Inside a GPU
    - Tensor Core
  - Programming GPUs
    - CUDA (NVIDIA only)
    - OpenCL (Runs on anything)
  - Google Tensor Processing Units (TPU)
    - Have to use TensorFlow
- Deep Learning Software
  - Mainstream Frameworks: PyTorch & TensorFlow
  - The points of Deep Learning frameworks
    - Allow rapid prototyping of new ideas
    - Automatically compute gradients
    - Run it all efficiently on GPU
  - PyTorch
    - Fundamental Concepts
      - Tensor: Like a numpy array, but can run on GPU
      - Autograd: Package for building computational graphs out of Tensors, and automatically computing gradients
      - Module: A neural network layer; may store state or learnable weights
    - Dynamic Computation Graphs
      - Applications: Recurrent Networks/Recursive Networks/Modular Networks
    - Static Computation Graphs
    - Static vs Dynamic Computation Graphs
      - For Static
        - Optimization
        - Serialization
      - For Dynamic
        - Debugging
        

**Lecture 9: [Training Neural Networks I](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L10_Training_Neural_Networks_I.md)** 
- Activation Functions
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - Exponential Linear Unit (ELU)
  - Scaled Exponential Linear Unit (SELU)
  - Summary: Just use ReLU. Don't use sigmoid or tanh    
- Data Preprocessing
  - Types
    - Zero-centering & Normalization
    - PCA & Whitening
  - Summary: Use Zero-centering & Normalization. Not common to do PCA or whitening
- Weight Initialization
  - Idea: Small random numbers
    - Gaussian w/ zero mean, std=0.01
    - Gaussian w/ zero mean, std=0.05
    - Xavier Initialization
      - Problem (w/ ReLU)
    - Kaiming/MSRA Initialization
    - Residual Networks
- Regularization
  - A common pattern:
    - Training: Add some kind of randomness
    - Testing: Average out randomness (sometimes approximate)
  - Add term to the loss: L1/L2/Elastic Net(L1+L2)
  - Dropout: How/Why/Problems
  - Data Augmentation
    - Types: Transform/Random Crops & Scales/Color Jitter
  - Other ways to regularize
  - Summary: 
    - Batch normalization and data augmentation almost always a good idea
    - Dropout not used quite much these days
