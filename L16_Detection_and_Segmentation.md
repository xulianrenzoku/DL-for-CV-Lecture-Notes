Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 16: Detection and Segmentation  


**Object Detection Recap**
- Slow R-CNN Training
    - Slide 16-10: Categorize each region proposal as positive, negative, or neutral based on overlap with ground-truth boxes
        - JJ: A step we did not discuss last time
    <img src='static/16-10.png' width='600'> 
    
    - Slide 16-11: Crop pixels from each positive and negative proposal, resize to 224 * 224
        - JJ: We don’t use neutral 
    <img src='static/16-11.png' width='600'> 
    
    - Slide 16-12: Run each region through CNN. For positive boxes predict class and box offset; for negative boxes just predict background class
        - JJ: Regression loss only for positive regional proposal
    <img src='static/16-12.png' width='600'> 
    
- Fast R-CNN
    - Slide 16-13: Crop features for each region, use them to predict class and box targets per region
    <img src='static/16-13.png' width='600'> 
- Faster R-CNN
    - Anchor —> Region Proposal —> Object Box
    - Slide 16-15: RPN
    <img src='static/16-15.png' width='600'> 
    
    - Slide 16-16: Stage 2
    <img src='static/16-16.png' width='600'> 
- Cropping Features
    - RoI Pool
        - Problem #1: Misaligned features due to snapping
        - Problem #2: Can’t backprop to box coordinates
    - RoI Align
        - JJ: We want to remove all the snapping in the RoI Pool
        - Instead of snapping, sample features at regularly-spaced points in each subregion using **bilinear interpolation**
        - Example: Slide 16-23, 16-24, 16-25
        <img src='static/16-23.png' width='600'> 
        <img src='static/16-24.png' width='600'> 
        <img src='static/16-25.png' width='600'> 
        
        - After sampling, max-pool in each subregion
        - Output features now aligned to input box! And we an backprop to box coordinates!
- Detection without Anchors: CornerNet
    - Slide 16-32
    <img src='static/16-32.png' width='600'> 

**Semantic Segmentation**
- Definition of Task: Slide 16-35
    - Label each pixel in the image with a category label
    - Don’t differentiate instances, only care about pixles
<img src='static/16-35.png' width='600'> 

- Sliding Window: Slide 16-37
    - Problem: Very slow
<img src='static/16-37.png' width='600'> 

- Fully Convolutional Network: 
    - Slide 16-38
        - Design a network as a bunch of convolutional layers to maker predictions for pixels all at once!
        - Loss function: Per-Pixel cross-entropy
    <img src='static/16-38.png' width='600'> 
    
    - Problems
        - #1: Effective receptive field size is linear in number of conv layers: With L 3 * 3 conv layers, receptive field is 1+2L
        - #2: Convolution on high res images is expensive! Recall ResNet stem aggressively downsamples
    - Slide 16-42
        - Design network as a bunch of convolutional layers, with **downsampling** and **upsampling** inside the network
        - What is upsampling?
    <img src='static/16-42.png' width='600'> 
    
    - In-Network Unpsampling
        - No learnable parameter
            -  “Unpooling”: Slide 16-44
                - Bed of Nails
                - Nearest Neighbor
            <img src='static/16-44.png' width='600'> 
            
            - Bilinear Interpolation: Slide 16-45
            <img src='static/16-45.png' width='600'> 
            
            - Bicubic Interpolation: Slide 16-46
                - JJ: I don’t want to get into details here
            <img src='static/16-46.png' width='600'> 
            
            - Max Unpooling: Slide 16-47
                - Pair each downsampling layer with an upsampling layer
            <img src='static/16-47.png' width='600'> 
            
        - Learnable parameter
            - Transposed Convolution
                - Slide 16-56
                <img src='static/16-56.png' width='600'> 
                
                - Slide 16-57
                <img src='static/16-57.png' width='600'> 
                
                - Slide 16-58
                <img src='static/16-58.png' width='600'> 
                
                - Slide 16-59
                <img src='static/16-59.png' width='600'> 
                
                - 1D Example: Slide 16-60
                <img src='static/16-60.png' width='600'> 

**Instance Segmentation**
- Things and Stuff
    - Things: Object categories that can be separated into object instances (e.g. cats, cars, person)
    - Stuff: Object categories that cannot be separated into instances (e.g. sky, grass, water, trees)
    - Takeaways:
        - Object Detection: Only things!
        - Semantic Segmentation: Both things and stuff
- Instance Segmentation: Slide 16-71
    - Detects all objects in the image, and identify the pixels that belong to each object (Only things!)
    - **Approach**: Perform object detection, then predict a segmentation mask for each object!
<img src='static/16-71.png' width='600'> 

- Mask R-CNN
    - Slide 16-74
    <img src='static/16-74.png' width='600'> 
    
    - Slide 16-75
    <img src='static/16-75.png' width='600'> 
    
    - Slide 16-79: Example Training Targets
    <img src='static/16-79.png' width='600'> 
    
    - Slide 16-80
    <img src='static/16-80.png' width='600'> 
    
- Beyond Instance Segmentation 
    - Panoptic Segmentation
    - Human Keypoints
    - Dense Captioning
    - 3D Shape Prediction 
