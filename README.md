# DL-for-CV-Lecture-Notes
University of Michigan

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
