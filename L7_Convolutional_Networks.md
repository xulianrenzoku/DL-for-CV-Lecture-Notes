Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 7: Convolutional Networks

**Last Lecture**
- Status: Stretch pixels into column
- Problem: So far our classifiers don’t respect the spatial structure of images
    - f(x, W) = Wx
    - W_2 max(0, W_1 x)
**This Lecture**
- Solution: Define new computational nodes that operate on images!

**Convolutional Network**
- Components of a Full-Connected Network
    - Fully-Connected Layers
    - Activation Function
    <img src='static/7-7.png' width='600'> 
- Components of a Convolutional Network
    - Fully-Connected Layers
    - Activation Function
    - Convolution Layers
    - Pooling Layers
    - Normalization
    <img src='static/7-8.png' width='600'> 

**Three New Operations**  
**Convolution Layers**
- Fully-Connected Layer: Slide 7-10
<img src='static/7-10.png' width='600'> 
- Convolution Layer: 
    - Slide 7-14
        - Input: 3D tensor
            - 3 * 32 * 32 image: preserve spatial structure
        - Weight matrix 
            - 3 * 5 * 5 filter
            - **Convolve** the filter with the image
            - i.e. “slide over the image spatially, computing dot products”
        - Filter always extend the full depth of the input volunme
        <img src='static/7-14.png' width='600'> 
    - Slide 7-15
        - Output: 1 number
            - The result of taking a dot product b/w the filter and a small 3*5*5 chunk of the image
            - i.e. 3 * 5 * 5 = 75-dimensional dot product + bias
        <img src='static/7-15.png' width='600'> 
    - Slide 7-16
        - Convolve (slide) over all spatial locations
        - Activation map
            - 1 * 28 * 28
        <img src='static/7-16.png' width='600'> 
    - Slide 7-17
        - Consider repeating with a second (green) filter
        <img src='static/7-17.png' width='600'> 
    - Slide 7-18
        - Consider 6 filters
            - Each 3 * 5 * 5
        - 6 activation maps
            - Each 1 * 28 * 28
            - JJ: 32+1-5=28
        - Stack activations to get a 6 * 28 * 28 output image!
        <img src='static/7-18.png' width='600'> 
    - Slide 7-19
        - Also 6-dim bias vector
        <img src='static/7-19.png' width='600'> 
    - Slide 7-21, 7-22
        - Batch of images 
        - Batch of outputs
        <img src='static/7-21.png' width='600'> 
	<img src='static/7-22.png' width='600'> 
- Stacking Convolutions
    - Slide 7-23, 7-25 (Correction: b_1 should be 6 instead of 5)
    - Problem: What if we stack two convolution layers?
    - We get another convolution!
        - Recall y=W2W1x is a linear classifier
    - Solution: ReLU 
        - Non-linear activation function
    <img src='static/7-23.png' width='600'> 
    <img src='static/7-25.png' width='600'> 
- What do convolutional filters learn?
    - Slide 7-29
    - First-layer conv filters: local image templates 
        - Often learns oriented edges, opposing colors
        - Example:L AlexNet
    <img src='static/7-29.png' width='600'> 
- A closer look at spatial dimensions
    - Slide 7-36
        - Example
            - Input: 7 * 7
            - Filter: 3 * 3
            - Output: 5 * 5
        - In general
            - Input: W
            - Filter: K
            - Output: W-K+1
        - Problem: Feature maps “shrink” with each layer!
            - To be specific: Lose at least 2 pixels every time thru a filter
    <img src='static/7-36.png' width='600'> 
    
    - Slide 7-37
        - Solution: **padding**
            - Add zeros around the input (Zero padding)
    <img src='static/7-37.png' width='600'> 
    
    - Slide 7-38
        - In general
            - Input: W
            - Filter: K
            -  Padding: P
            - Output: W-K+1+2P
        - Very common: set P = (K-1) / 2 to make output have same size as input!
    <img src='static/7-38.png' width='600'>  
    
- Receptive Fields
    - Slide 7-39
    	- For convolution with kernel size, each element in the output depends on a K*K **receptive field** in the input
    <img src='static/7-39.png' width='600'>  
    
    - Slide 7-40
    	- Each successive convolution add K-1 to the receptive field size With L layers the receptive field size is 1+L*(K-1)
    <img src='static/7-40.png' width='600'>
    
    - Slide 7-42
    	- Problem: For large images we need many layers for each output to “see” the whole image 
    	- Solution: Downsample inside the network
        	- JJ: Equivalent to adding another hyperparameter
    <img src='static/7-42.png' width='600'>
- **Strided** Convolution
    - Slide 7-43, 7-44, 7-46
    	- Example
    		- Input: 7 * 7
        	- Filter: 3 * 3
        	- Stride: 2 
            		- Skip over 1
        	- Output: 3 * 3
    	- In general: 
    		- Input: W
    		- Filter: K
    		- Padding: P
    		- Stride: S
    		- Output: (W-K+2P)/S+1
    <img src='static/7-43.png' width='600'>
    <img src='static/7-44.png' width='600'>
    <img src='static/7-46.png' width='600'>
- Convolution Example
    - Slide 7-48
    <img src='static/7-48.png' width='600'>
    
    - Slide 7-50
    <img src='static/7-50.png' width='600'>
    
    - Slide 7-52
    <img src='static/7-52.png' width='600'>
    
    - Slide 7-54: “Network in Network” structure
    <img src='static/7-54.png' width='600'>
- Convolution Summary
    - Slide 7-56
    - Common settings
    <img src='static/7-56.png' width='600'>
- Other types of convolution
    - Slide 7-58, 7-59
    - So far: 2D convolution
    - 1D Convolution
        - Input would be 2D
        - JJ: Sometimes used to process textual/audio data
    <img src='static/7-58.png' width='600'>
    
    - 3D Convolution
        - JJ: Sometimes used to process 3D data
    <img src='static/7-59.png' width='600'>
- PyTorch Convolution Layer

**Pooling Layers**
- Pooling Layers: Another way to downsample
    - Slide 7-63
    - Hyperparameters: 
        - Kernel Size
        - Stride
        - Pooling function
    <img src='static/7-63.png' width='600'>
- Max Pooling (Another non-linear function in reality)
    - Slide 7-64
    - Why we choose pooling over strides?
        - Introduces **invariance** to small spatial shifts
        - No learnable parameters! 
    - Other types
        - Average pooling
    <img src='static/7-64.png' width='600'>
- Pooling Summary
    - Slide 7-65
    - Common setting
    <img src='static/7-65.png' width='600'>

**Convolutional Networks**
- Classic architecture
    - Slide 7-67
    - [Conv, ReLU, Pool] —> Flatten —> [Linear, ReLU]
    <img src='static/7-67.png' width='600'>
- Example: LeNet-5
    - Slide 7-68, 7-69, 7-70, 7-71, 7-72, 7-73, 7-74, 7-75, 7-76
        - As we go thru the network
        - Spatial size **decreases** (using pooling or strided conv)
        - Number of channels **increases** (total “volume” is preserved!)
    <img src='static/7-68.png' width='600'>
    <img src='static/7-69.png' width='600'>
    <img src='static/7-70.png' width='600'>
    <img src='static/7-71.png' width='600'>
    <img src='static/7-72.png' width='600'>
    <img src='static/7-73.png' width='600'>
    <img src='static/7-74.png' width='600'>
    <img src='static/7-75.png' width='600'>
    <img src='static/7-76.png' width='600'>
    
    	- Problem: Deep Networks very hard to train!

Solution:  
**Normalization**
- Batch Normalization: 
    - Slide 7-79
        - Idea: “Normalize” the outputs of a layer so they have zero mean and unit variance
        - Why: Helps reduce “internal covariate shift”, improves optimization
        - We can normalize a batch of activations like this:
        - This is a **differentiable function**, so we can use it as an operator in our networks and backprop thru it!
    <img src='static/7-79.png' width='600'>
    
    - Slide 7-81
        - Problem: What if zero-mean, unit variance is too hard of a constraint?
    <img src='static/7-81.png' width='600'>
    
    - Slide 7-83
        - Learnable scale and shift parameters: gamma and beta
        - Leaning gamma = sigma, beta = mu will recover the identity function
        - Problem: Estimates depend on minibatch; can’t do this at test-time!
            - JJ: Gotta make it independent
    <img src='static/7-83.png' width='600'>
    
    - Slide 7-84
        - Introduce (running) average
    <img src='static/7-84.png' width='600'>
    
    - Slide 7-85
        - During test batchnorm becomes a linear operator! 
        - Can be fused with the previous fully-connected or conv layer
            - JJ: Batch Normalization becomes free during test time. Very nice in practice
    <img src='static/7-85.png' width='600'>
- Batch normalization for ConvNets
    - Slide 7-86
    <img src='static/7-86.png' width='600'>
- Structure and properties of Batch normalization
    - Slide 7-87
        - Usually inserted after Fully Connected or Convolutional layers, and before nonlinearity
    <img src='static/7-87.png' width='600'>
    
    - Slide 7-88 (GOOD)
        - Makes deep networks **much** easier to train!
        - Allows higher learning rates, faster convergence
        - Networks become more robust to initialization
        - Acts as regularization during training
        - Zero overhead at test-time: can be fused with conv!
    <img src='static/7-88.png' width='600'>
    
    - Slide 7-89 (BAD)
        - Not well-understood theoretically (yet)
        - Behaves differently during training and testing: this is a very common source of bugs!
    <img src='static/7-89.png' width='600'>
- Variant: 
    - Layer Normalization: Slide 7-90
        - Same behavior at train and test!
        - Used in RNNs, Transformers
    <img src='static/7-90.png' width='600'>
    
    - Instance Normalization: Slide 7-91
        - Same behavior at train and test!
    <img src='static/7-91.png' width='600'>
    
    - Comparison: Slide 7-93
    <img src='static/7-93.png' width='600'>



















