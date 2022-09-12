Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 14: Visualizing and Understanding

**Activations**
- First Layers: Visualize Filters
- Higher Layers: Visualize Filters
    - We can visualize filters at higher layers, but not that interesting
- Last Layer
    - Nearest Neighbors
        - L2 Nearest neighbors in feature space
    - Dimensionality Reduction
        - Visualize the “space” of FC7 feature vectors by reducing dimensionality of vectors from 4096 to 2 dimensions
        - Simple algorithm: PCA
        - More complex: t-SNE
- Visualizing Activations
- Maximally Activating Patches
    - Pick a layer and a channel; e.g. conv5 is 128 * 13 * 13, pick channel 17/128
    - Run many images through the network, record values of chosen channel
    - Visualize image patches that correspond to maximal activations
- Saliency via Occlusion: Slide 14-15
    - Mask part of the image before feeding to CNN, check how much predicted probabilities change
    <img src='static/14-15.png' width='600'> 
- Saliency via Backprop: Slide 14-17
    - Forward pass: Compute probabilities
    - Compute gradient of (unnormalized) class score with respect to image pixels, take absolute value and max over RGB channels
    - JJ: Check the network is looking at the right part of the image
    <img src='static/14-15.png' width='600'> 

**Graidents**
- Saliency Maps: Segmentation without Supervision
    - Use GrabCut on saliency map
- Intermediate Features via (guided) backprop
    - JJ: Use gradient info to pick out pixels
    - Slide 14-21
        - Pick a single intermediate neuron, e.g. one value in 128 * 13 * 13 conv5 feature map
        - Compute gradient of neuron value with respect to image pixels
        - Images come out nicer if you only backprop positive gradients through each ReLU (guided backprop)
    <img src='static/14-21.png' width='600'> 
    
    - Slide 14-22
    <img src='static/14-22.png' width='600'> 
- Gradient Ascent
    - Slide 14-24
        - **(Guided) backprop**: Find the part of an image that a neuron responds to
        - **Gradient ascent**: Generate a synthetic image that maximally activates a neuron
    <img src='static/14-24.png' width='600'> 
    
    - Step 1: Slide 14-25
    <img src='static/14-25.png' width='600'> 
    
    - Simple Regularizer: Slide 14-27
    <img src='static/14-27.png' width='600'> 
    
    - Better Regularizer: Slide 14-30
    <img src='static/14-30.png' width='600'> 
- Adversarial Examples
    - 1. Start from an arbitrary image
    - 2. Pick an arbitrary category
    - 3. Modify the image (via gradient ascent) to maximize the class score
    - 4. Stop when the network is fooled
    - Slide 14-37
    <img src='static/14-37.png' width='600'> 
- Feature Inversion
    - Slide 14-38
        - Given a CNN feature vector for an image, find a new image that
            - Matches the given feature vector
            - “Looks natural” (image prior regularization)
    <img src='static/14-38.png' width='600'> 
    
    - Slide 14-39
        - Reconstructing from different layers of VGG-16
    <img src='static/14-39.png' width='600'> 

**Fun**
- DeepDream: Amplify Existing Features
    - Slide 14-40
    <img src='static/14-40.png' width='600'> 
    
    - Slide 14-45: Code
    <img src='static/14-45.png' width='600'> 
    
    - Slide 14-47: Lower level layer (looking for edges)
    <img src='static/14-47.png' width='600'> 
    
    - Slide 14-48: Higher level layer 
    <img src='static/14-48.png' width='600'> 
    
- Texture Synthesis 
    - Given a sample patch of some texture, can we generate a bigger image of the same texture?
    - Nearest Neighbor
    - Texture Synthesis w/ Neural Networks: Gram Matrix
        - Slide 14-55
        <img src='static/14-55.png' width='600'> 
        
        - Slide 14-56
        <img src='static/14-56.png' width='600'> 
        
        - Slide 14-57
        <img src='static/14-57.png' width='600'> 
        
        - Slide 14-58
        <img src='static/14-58.png' width='600'> 
    - Neural Texture Synthesis
        - Slide 14-65
        <img src='static/14-65.png' width='600'> 
    - Neural Style Transfer: Feature + Gram Reconstruction
        - Slide 14-68
        <img src='static/14-68.png' width='600'> 
        
        - Slide 14-69
        <img src='static/14-69.png' width='600'> 
        
        - Slide 14-70
        <img src='static/14-70.png' width='600'> 
        
        - Problem: Style transfer requires many forward / backward passes through VGG; very slow!
            - JJ: Takes a lot of GPUS
        - Solution: Train another neural network to perform style transfer for us!
            - Fast Neural Style Transfer
            - Instance Normalization: Slide 14-88
            <img src='static/14-88.png' width='600'> 

**Summary**
- Many methods for understanding CNN representations
    - **Activations**: Nearest neighbors, Dimensionality reduction, mammal patches, occlusion
    - **Gradients**: Saliency maps, class visualization, fooling images, feature inversion
    - **Fun**: DeepDream, Style Transfer

