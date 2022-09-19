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
        

**Lecture 10: [Training Neural Networks I](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L10_Training_Neural_Networks_I.md)** 
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
    
    
**Lecture 11: [Training Neural Networks II](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L11_Training_Neural_Networks_II.md)** 
- Training Dynamics
  - Learning Rate Schedules
    - Types of Learning Rate Decay
      - Step
      - Cosine
      - Linear
      - Inverse Sqrt
      - Constant 
    - How long to train? Early Stopping
  - Hyperparameter Optimization
    - Choosing Hyperparameters
      - with tons of GPUs
        - Grid Search
        - Random Search (works better)
      - without tons of GPUs
        - Step 1: Check initial loss
        - Step 2: Overfit a small sample
        - Step 3: Find LR that makes loss go down
        - Step 4: Coarse grid, train for ~1-5 epochs
        - Step 5: Refine grid, train longer
        - Step 6: Look at learning curves
        - GOTO step 5
    - Hyperparameters to play with (like a DJ)
      - Network architecture
      - Learning rate, its decay schedule, update type
      - Regularization (L2/Dropout strength)
    - Track ratio of weight update / weight magnitude
- After Training
  - Model Ensembles
    - Steps
      - Train multiple independent models
      - At test time average their results
    - Tips and Tricks
  - Transfer Learning
    - Steps
    - Guidelines
  - Distributed Training
    - Typical way: Copy Model on each GPU, split data
    - Large-Batch Training
      - Tricks
        - Scale Learning Rates (Linearly)
        - Learning Rate Warmup


**Lecture 12: [Recurrent Neural Networks](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L12_Recurrent_Neural_Networks.md)** 
- Definition
  - Recurrent Neural Networks: Process Sequences
    - One-to-many
      - e.g. Image captioning (Image —> Sequence of words)
    - Many-to-one
      - e.g. Video classification (Sequence of images —> label)
    - Many-to-many
      - e.g. Machine translation (Sequence of words —> sequence of words)
      - e.g. Per-frame video classification (Sequence of images —> Sequence of labels)
  - Key idea
    - RNNs have an “internal state” that is updated as a sequence is processed
    - We can process a sequence of vectors x by applying a recurrence formula at every time step
- (Vanilla) RNN
  - Concept
    - tanh
  - Computational Graph
    - Many-to-many
    - Many-to-one
    - One-to-many
    - seq2seq (many-to-one + one-to-many)
  - Backpropagation Through Time
    - Problem
  - Truncated Backpropagation Through Time
    - Steps
  - Gradient Flow
- Long Short Term Memory (LSTM)
  - Intuition: cell state & hidden state
  - Architecture: input gate(i), forget gate(f), output gate(o), gate gate(g)
  - Gradient Flow: Uninterrupted
- Multi-layer RNN
- Other RNN Variants
  - Gated Recurrent Unit (GRU)


**Lecture 13: [Attention](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L13_Attention.md)** 
- Attention
  - Architecture
    - Sequence-to-Sequence with RNNs
    - Sequence-to-Sequence with RNNs and Attention
      - Steps
        - Alignment scores & alignment weights
        - Context vector
- (Generalized) Self-Attention
  - Changes
    - Use dot product for similarity
    - Multiple query vectors
    - Separate key and value
  - Self-Attention Layer
    - Permutation Equivariant
    - Works on sets of vectors
  - Masked Self-Attention Layer
  - Multihead Self-Attention Layer
  - Example: CNN with Self-Attention
- Transformer
  - Three Ways of Processing Sequences
    - Reccurent Neural Networks
      - Works on Ordered Sequences
        - Good at long sequences
        - Not parallelizable
    - 1D Convolution
      - Works on Multidimensional Grids
        - Bad at long sequences
        - Highly parallel
    - Self-Attention
      - Works on Sets of Vectors
        - Good at long sequences
        - Highly parallel
        - Very memory intensive
  - The Transformer: Only uses attention
    - Steps
    - Transformer Block
    - Transformer: A Transformer is a sequence of transformer blocks
    - Transfer Learning
      - Pretraining
      - Finetuning
      

**Lecture 14: [Visualizing and Understanding](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L14_Visualizing_and_Understanding.md)**  
- Activations
  - First Layers
  - Higher layers
  - Last Layer
    - Nearest Neighbors
    - Dimensionality Reduction
  - Maximally Activating Patches
  - Saliency via Occlusion
  - Saliency via Backprop
- Graidents
  - Saliency Maps
  - Intermediate Features via (guided) backprop
  - Gradient Ascent
  - Feature Inversion
- Fun
  - Deep Dream: Amplify Existing Features
  - Texture Synthesis
    - Texture Synthesis w/ Neural Networks: Gram Matrix
    - Neural Style Transfer: Feature + Gram Reconstruction
      - Problem: Very slow
      - Solution: Train another neural network (Fast Neural Style Transfer)
        - Instance Normalization 


**Lecture 15: [Object Detection](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L15_Object_Detection.md)**  
- Object Detection
  - Task Definition
    - Input: Single RGB Image
    - Output: A set of detected objects; For each object predict:
      a. Category label (from fixed, known set of categories)
      b. Bounding box (four numbers: x, y, width, height)
  - Challenges
    - Multiple outputs
    - Mutiple types of output
    - Large images
  - Detecting a single object
  - Detecting multiple objects
    - Sliding window
- R-CNN: Region-Based CNN
  - Region Proposals
  - Steps
  - Comparing Boxes: Intersection over Union (IoU)
  - Overlapping Boxes: Non-Max Suppression (NMS)
  - Evaluating Object Detectors: Mean Average Precision (mAP)
  - Problems with R-CNN: Very slow
- Fast R-CNN
  - Steps
  - Structure
    - Most of the computation happens in backbone network
  - Examples (backbone): AlexNet, ResNet
  - Cropping Features
    - RoI Pool
    - RoI Align
- Faster R-CNN: Learnable Region Proposals
  - Insert Region Proposal Network (RPN) to predict proposals from features
  - Region Proposal Network (RPN)
    - Steps
    - Challenges
  - Jointly train with 4 losses
    a. RPN classification
    b. RPN regression
    c. Object classification
    d. Object regression
  - Faster R-CNN is a Two-stage object detector
  - Single-Stage Object Detection
    - RPN: Classify each anchor as object / not object
    - Single-Stage Detector: Classify each object as one of C categories (or background)


**Lecture 16: [Detection and Segmentation](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L16_Detection_and_Segmentation.md)**  
- Object Detection Recap
  - Slow R-CNN Training
  - Fast R-CNN
  - Faster R-CNN
  - Cropping Features
    - RoI Pool
    - RoI Align
  - Detection without anchors
- Semantic Segmentation
  - Definition of Task
  - Fully Convolutional Network
    - Problem
    - Solution: Design network as a bunch of convolutional layers, with downsampling and upsampling inside the network
    - In-Network Upsampling
      - No learnable parameter
        - "Unpooling"
        - Bilinear Interpolation
        - Bicubic Interpolation
        - Max Unpooling
      - Learnable parameter
        - Transposed Convolution
- Instance Segmentation
  - Things and Stuff
    - Things: Object categories that can be separated into object instances (e.g. cats, cars, person)
    - Stuff: Object categories that cannot be separated into instances (e.g. sky, grass, water, trees)
  - Instance Segmentation
    - Detects all objects in the image, and identify the pixels that belong to each object (Only things!)
    - Approach: Perform object detection, then predict a segmentation mask for each object
  - Mask R-CNN
  - Beyond Instance Segmentation


**Lecture 17: [3D Vision](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L17_3D_Vision.md)**
- 3D Shape Representations
  - Overview
    - Depth Map
    - Voxel Grid
    - Implicit Surface
    - Pointcloud
    - Mesh
  - Depth Map
    - Depth Map
      - Predicting Depth Maps
      - Problem: Scale/Depth Ambiguity
    - Surface Normals
  - Voxel Grid
    - Definition: Good & Bad
    - Processing Voxel Inputs: 3D Convolution
    - Generating Voxel Shapes:
      - 3D Convolution
      - "Voxel Tubes": Voxel-2 representation
    - Problems: Memory Usage
    - Scaling Voxels
      - Oct-Trees
      - Nested Shape Layers
  - Implicit Surface
    - 3D shape Representations: Implicit Functions
  - Pointcloud
    - Definition: Good & Bad
    - Processing Pointcloud Inputs: PointNet
    - Generating Pointcloud Outputs
  - Mesh
    - Definition: Good
    - Predicting Triangle Meshes
      - Iterative Refinement
      - Graph Convolution
      - Vertex Aligned-Features
      - Chamfer Loss Function
- 3D Metrics
  - Intersection over Union
  - Chamfer Distance
  - F1 Score
- 3D Camera Systems
  - Canonical vs View Coordinates
  - View-Centric Voxel Predictions
- 3D Datasets
  - ShapeNet
  - Pix3D   
- 3D Shape Prediction
  - Mesh R-CNN
    - Hypbrid 3D shape representation  
    - Pipeline
    - Shape Regularizers


**Lecture 18: [Videos](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L18_Videos.md)**
- Overview
  - Video = 2D+Time
  - Problem: Videos are big
  - Train on Clips: Train model to classify short clips with low FPS
- Video Classification Models
  - Single-Frame CNN
  - Late Fusion
    - with FC layers
    - with pooling
  - Early Fusion
  - 3D CNN
  - C3D
  - Two-stream Networks
    - Recognizing actions from motion
    - Measuring motion: optical flow
    - Architecture
      - Spatial stream ConvNet
      - Temporal stream ConvNet
        - Input: Stack of optical flow
  - CNN + RNN
  - Recurrent Convolutional Network
  - Spatio-Temporal Self-Attention (Nonlocal Block)
    - Inflating 2D Networks to 3D
      - Inception block
  - Treating time and space differently: SlowFast Networks
    - Architecture
      - Slow
      - Fast
- Other Studies
  - Temporal Action Localization
  - Spatio-Temporal Detection


**Lecture 19: [Generative Models I](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L19_Generative_Models_I.md)**
- Overview
  - Supervised vs Unsupervised Learning
  - Discriminative vs Generative Models
    - Discriminative Model: Learn a probability distribution p(y|x) 
    - Generative Model: Learn a probability distribution p(x)
    - Conditional Generative Model: Learn p(x|y)
    - Recall Baye's Rule
  - Taxonomy of Generative Models
- Autoregressive Models
  - Explicit Density Estimation
  - PixelRNN
  - PixelCNN
  - Pros & Cons
- Variational Autoencoders
  - Intuition
    - VAE define an intractable density that we cannot explicitly compute or optimize
    - But we will be able to directly optimize a lower bound on the density
  - Term "Autoencoders"
    - (Regular, non-variational) Autoencoders
  - Term "Variational": Probabilistic spin on autoencoders
  - Structure
    - Idea: Jointly train both encoder and decoder
    - Math (where "lower bound" comes from)   


**Lecture 20: [Generative Models II](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L20_Generative_Models_II.md)**
- Generative Adversarial Networks
- Idea
  - Setup: Assume we have data x_i drawn from distribution p_data(x). Want to sample from p_data
  - Intuition: Jointly train G and D. Hopefully p_G converges to p_data
- Training Objective: Jointly train generator and discriminator D with a minimax game
- Optimality
- Interpolation
- Vector Math
- GAN Improvements
  - Improved Loss Functions
  - Higher Resolution
- Conditional GANs
  - Conditional Batch Normalization


**Lecture 21: [Reinforcement Learning](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L21_Reinforcement_Learning.md/)**
- Definition
  - Problems where an agent performs actions in environment, and receives rewards
  - Goal: Learn how to take actions that maximize reward
- Examples
  - Cart-Pole Problem
  - Robot Locomotion
  - Atari Games
  - Go
- Why is RL different from normal supervised learning?
  - Stochasticity
  - Credit assignment
  - Nondifferentiable
  - Nonstationary
- Markov Decision Process
- Find optimal policies
- Value Function and Q Function
- Bellman Equation
- Solving for the optimal policy
  - Value Iteration
  - Deep Q-Learning
- Q-Learning
- Policy Grdients
- Other approaches: Model Based RL
  - Actor-Critic
  - Model-Based
  - Imitation Learnin
  - Inverse Reinforcement Learning
  - Adversarial Learning
- Stochastic Computation Graphs


**Lecture 21: [Conclusion](https://github.com/xulianrenzoku/DL-for-CV-Lecture-Notes/blob/main/L22_Conclusion.md/)**
- Predictions
  - New deep learning models
  - New applications
  - More compute, new hardware
- Problems 
  - Models are biased
  - Need new theory
  - Using less data
  - Understanding the world
