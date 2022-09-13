Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 15: Object Detection

**Overview**
- So far: Image Classification
    - JJ: Image in, label out
- Computer Vision Tasks: Slide 15-6
    - Classification
    - Semantic Segmentation
    - Object Detection (Today)
    - Instance Segmentation

**Object Detection**
- Task Definition: Slide 15-8
    - Input: Single RGB Image
    - Output: A set of detected objects; For each object predict:
        - 1. Category label (from fixed, known set of categories)
        - 2. Bounding box (four numbers: x, y, width, height)
<img src='static/15-8.png' width='600'> 

- Challenges
    - **Multiple outputs**: Need to output variable numbers of objects per image
    - **Multiple types of output**: Need to predict “what“ (category label) as well as “where” (bounding box)
    - **Large images**: Classification works at 224 * 224; need higher resolution for detection, often ~800 * 600
        - JJ: It’s a computation issue. Needs fewer images per batch. Train longer. Multi-distributed training…
        - JJ: No.2 core CV problem after image classification
- Detecting a single object
    - Steps
        - Slide 15-10: Get Vector representation 
        <img src='static/15-10.png' width='600'> 
        
        - Slide 15-11: “What”
        <img src='static/15-11.png' width='600'> 
        
        - Slide 15-12: “Where”
            - Treat localization as a regression problem!
        <img src='static/15-12.png' width='600'> 
        
        - Slide 15-14: Multitask Loss
            - Add up losses to make one loss function
        <img src='static/15-14.png' width='600'> 
        
        - Slide 15-15: Often pretrained on ImageNet (Transfer learning)
        <img src='static/15-15.png' width='600'> 
    - Problem: Images can have more than one object!
- Detecting Multiple Objects
    - Slide 15-17: Need different numbers of outputs per image
    <img src='static/15-17.png' width='600'> 
    
    - Sliding Window
        - Slide 15-19
            - Apply a CNN to many different crops of the image, CNN classifies each crop as object or background
        <img src='static/15-19.png' width='600'> 
        
        - Challenge: Slide 15-25
            - Problem: How many possible boxes are there in an image of size H * W
        <img src='static/15-25.png' width='600'> 

**R-CNN: Region-Based CNN**
- Region Proposals: Slide 15-26
    - Find a small set of boxes that are likely to cover all objects
    - Often based on heuristics: e.g. look for “blob-like” image regions
    - Relatively fast to run e.g. Selective Search
<img src='static/15-26.png' width='600'> 

- Steps
    - Slide 15-27: Region Proposal
    <img src='static/15-27.png' width='600'> 
    
    - Slide 15-28
    <img src='static/15-28.png' width='600'> 
    
    - Slide 15-29: Warp
    <img src='static/15-29.png' width='600'> 
    
    - Slide 15-30
    <img src='static/15-30.png' width='600'> 
    
    - Slide 15-31
    <img src='static/15-31.png' width='600'> 
    
    - Slide 15-32: Bounding box regression
    <img src='static/15-32.png' width='600'> 
    
    - Slide 15-33
    <img src='static/15-33.png' width='600'> 
    
- Test-time: Slide 15-34
    - Input Single RGB Image
    - 1. Run region proposal method to compute ~2000 region proposals
    - 2. Resize each region to 224 * 224 and run independently through CNN to predict class scores and box transform
    - 3. Use scores to select a subset of region proposals to output
    - 4. Compare with ground-truth boxes
    - Comments:
        - JJ: we do not use location info as input
        - JJ: in classification, we usually use 224 * 224, then we warp the image to 224 * 224 for image detection as well
<img src='static/15-34.png' width='600'> 

- Comparing Boxes: Intersection over Union (IoU)
    - Slide 15-37
        - Intersection over Union (IoU) (Also called “Jaccard similarity” or “Jaccard index”)
        - Area of Intersection / Area of Union
            - IoU > 0.5 is “decent”
            - IoU > 0.7 is “pretty good”
            - IoU > 0.9 is “almost perfect”
    <img src='static/15-37.png' width='600'> 
- Overlapping Boxes: Non-Max Suppression (NMS)
    - Slide 15-41
        - Problem: Object detectors often output many overlapping detections:
        - Solution: Post-process raw detections using Non-Max Suppression (NMS)
            - 1. Select next highest-scoring box
            - 2. Eliminate lower-scoring boxes with IoU > threshold (e.g. 0.7)
            - 3. If any boxes remain, GOTO 1
        - Example: Slide 15-42, 15-43, 15-44
    <img src='static/15-41.png' width='600'> 
    
    - Failure mode: Slide 15-45
        - JJ: Still ongoing challenge
    <img src='static/15-45.png' width='600'> 
- Evaluating Object Detectors: Mean Average Precision (mAP)
    - 1. Run object detector on all test images (with NMS)
    - 2. For each category, compute Average Precision (AP) = area under Precision vs Recall Curve
        - Steps: 
            - 1. Slide 15-47
            <img src='static/15-47.png' width='600'> 
            
            - 1-1 & 2. Slide 15-48
            <img src='static/15-48.png' width='600'>   
                
            - 1-3. Slide 15-49, 15-50, 15-51, 15-52, 15-53
            <img src='static/15-49.png' width='600'> 
            <img src='static/15-50.png' width='600'> 
            <img src='static/15-51.png' width='600'> 
            <img src='static/15-52.png' width='600'> 
            <img src='static/15-53.png' width='600'> 
                
            - 2. Slide 15-54 
            <img src='static/15-54.png' width='600'> 
    - How to get AP = 1.0: Slide 15-55
        - JJ: All TPs come before All FPs
        - JJ: Why this metric? Trade-off b/w how many objets you hit and how many objects you miss
    <img src='static/15-55.png' width='600'> 
    
    - 3. Mean Average Precision (mAP) = average of AP for each category
        - Slide 15-56
    <img src='static/15-56.png' width='600'> 
    
    - 4. For “COCO mAP”: Compute mAP&thresh for each IoU threshold (0.5, 0.55, 0.6, …, 0.95) and take average
        - Slide 15-57
    <img src='static/15-57.png' width='600'> 
    
- Problems with R-CNN
    - Problem: Very slow! Need to do ~2k forward passes for each image!
    - Solution: Run CNN *before* warping!

**Fast R-CNN**
- Steps
    - Slide 15-62
    <img src='static/15-62.png' width='600'> 
    
    - Slide 15-63: Run whole image thru ConvNet
        - JJ: Most computation
    <img src='static/15-63.png' width='600'> 
    
    - Slide 15-64: RoIs from a proposal method
    <img src='static/15-64.png' width='600'> 
    
    - Slide 15-65: Crop + Resize features
    <img src='static/15-65.png' width='600'> 
    
    - Slide 15-66: Per-Region Network
    <img src='static/15-66.png' width='600'> 
    
    - Slide 15-67: Category and box transform per region
    <img src='static/15-67.png' width='600'> 
    
- Structure
    - Slide 15-68
        - Per-Region network is relatively lightweight
        - Most of the computation happens in backbone network; this saves work for overlapping region proposals
    <img src='static/15-68.png' width='600'> 
    
- Examples
    - AlexNet: Slide 15-69
    <img src='static/15-69.png' width='600'> 
    
    - ResNet: Slide 15-70
    <img src='static/15-70.png' width='600'> 
    
- How to crop features?
- Cropping Features
    - RoI Pool
        - Project proposal onto features: Slide 15-74
        <img src='static/15-74.png' width='600'> 
        
        - “Snap” to grid cells: Slide 15-75
        <img src='static/15-75.png' width='600'> 
        
        - Divide into 2 * 2 grid of (roughly) equal subregions: Slide 15-76
        <img src='static/15-76.png' width='600'> 
        
        - Max-pool within each subregion: Slide 15-77
            - Region features always the same size even if input regions have different sizes!
        <img src='static/15-77.png' width='600'> 
        
        - Problem: Slight misalignment due to snapping, different-sized subregions is weird
    - RoI Align: Slide 15-89
    <img src='static/15-89.png' width='600'> 
    
- Fast R-CNN vs “Slow” R-CNN: 
    - Problem: Slide 15-93
    <img src='static/15-93.png' width='600'> 
    
    - Problem: Slide 15-94
    <img src='static/15-94.png' width='600'> 

**Faster R-CNN: Learnable Region Proposals**
- Slide 15-95
    - Insert **Region Proposal Network (RPN)** to predict proposals from features
    - Otherwise same as Fast R-CNN: Crop features for each proposal, classify each one
<img src='static/15-95.png' width='600'> 

- Region Proposal Network (RPN)
    - Steps
        - Slide 15-96: Run backbone CNN
        <img src='static/15-96.png' width='600'> 
        
        - Slide 15-97: Anchor box
        <img src='static/15-97.png' width='600'> 
        
        - Slide 15-98: Binary classification
        <img src='static/15-98.png' width='600'> 
        
        - Slide 15-99: Also predict a box transform to regress from anchor box to object box
        <img src='static/15-99.png' width='600'> 
        
    - Challenge: Slide 15-100
        - Problem: Anchor box may have the wrong size / shape
        - Solution: Use K different anchor boxes at each point!
    <img src='static/15-100.png' width='600'> 
    
- Loss: Slide 15-101
    - Jointly train with 4 losses
        - 1. RPN classification
        - 2. RPN regression
        - 3. Object classification
        - 4. Object regression
<img src='static/15-101.png' width='600'> 

- Comparison: Slide 15-102
<img src='static/15-102.png' width='600'> 

- Faster R-CNN is a **Two-stage** object detector: Slide 15-103
    - Question: Do we really need the second stage?
<img src='static/15-103.png' width='600'> 

- Single-Stage Object Detection: Slide 15-106
    - RPN: Classify each anchor as object / not object
    - Single-Stage Detector: Classify each object as one of C categories (or background)
    - Sometimes use **category-specific regression**: Predict different box transforms for each category
<img src='static/15-106.png' width='600'> 

**Current Status & Code**
- Object Detection: Lots of variables!
    - Takeaways: Slide 15-110
        - Two stage method (Faster R-CNN) get the best accuracy, but are slower
        - Single-stage methods (SSD) are much faster, but don’t perform as well
        - Bigger backbones improve performance, but are slower
    <img src='static/15-110.png' width='600'> 
    
    - Since 2017: Slide 15-118
        - Train longer!
        - Multiscale backbone: Feature Pyramid Networks
        - Better backbone: ResNeXt
        - Single-Stage methods have improved
        - Very big models work better
        - Test-time augmentation pushes numbers up
        - Big ensembles, more data, etc
    <img src='static/15-118.png' width='600'> 
    
- Open-Source Code: Slide 15-119
    - Detectron2 (PyTorch)
    - Fast/Faster/Mask R-CNN, RetinaNet
<img src='static/15-119.png' width='600'> 

**Summary**
- **”Slow” R-CNN**: Run CNN independently for each region
- **Fast R-CNN**: Apply differentiable cropping to shared image features
- **Faster R-CNN**: Compute proposal with CNN
- **Single-Stage**: Fully convolutional detector
