Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 17: 3D Vision

**Overview**
- Focus on Two Problems today
    - Predicting 3D shapes from single image
        - Input: Image
        - Output: 3D shape
    - Processing 3D input data
        - Input: 3D shape
        - Output: label

**3D Shape Representations**
- 3D Shape Representations: Slide 17-8
    - Depth Map
    - Voxel Grid
    - Implicit Surface
    - Pointcloud
    - Mesh
<img src='static/17-8.png' width='600'> 

- Depth Map
    - Depth Map
        - Definition: Slide 17-10
            - For each pixel, **depth map** gives distance from the camera to the object in the world at that pixel
            - RGB-D Image (2.5D)
        <img src='static/17-10.png' width='600'>
        
        - Predicting Depth Maps: Slide 17-13
            - Per-Pixel Loss (Scale invariant)
        <img src='static/17-13.png' width='600'>
        
        - Problem: Scale/Depth Ambiguity: Slide 17-12
        <img src='static/17-12.png' width='600'>
        
    - Surface Normals
        - Definition: Slide 17-14
            - For each pixel, **surface normals** give a vector giving the normal vector to the object in the world for that pixel
        <img src='static/17-14.png' width='600'>
        
        - Predicting Normals: Slide 17-15
        <img src='static/17-15.png' width='600'>
        
- Voxel Grid
    - Definition: Slide 17-17
        - Good: Conceptually simple: just a 3D grid!
        - Bad: Need high spatial resolution to capture fine structures
        - Bad: Scaling to high resolutions is nontrivial!
        - JJ: A minecraft representation
    <img src='static/17-17.png' width='600'>
    
    - Processing Voxel Inputs
        - 3D Convolution: Slide 17-18
            - JJ: Kernel does not to be binary, will be real-valued
    - Generating Voxel Shapes
        <img src='static/17-18.png' width='600'>
        
        - 3D Convolution: Slide 17-19
            - JJ: This architecture is very very computationally expensive
        <img src='static/17-19.png' width='600'>
        
        - “Voxel Tubes”: Slide 17-20
            - JJ: V * V * V called a voxel 2 representation
            - Q: Do we sacrifice anything by doing this?
            - JJ’s A: We sacrifice translational invariance in Z-dimension, but we still have translational invariance in XY-dimension
        <img src='static/17-20.png' width='600'>
        
    - Voxel Problems: Memory Usage
        - Storing 1024^3 voxel grid takes 4GB of memory!
    - Scaling Voxels: 
        - Oct-Trees: Slide 17-22
            - JJ: Implementing this is tricky
        <img src='static/17-22.png' width='600'>
        
        - Nested Shape Layers: Slide 17-23
            - Predict shape as a composition of positive and negative spaces
        <img src='static/17-23.png' width='600'>
        
- Implicit Surface
    - Definition: Slide 17-27
        - Learn a function to classify arbitrary 3D points as inside/outside the shape
        - Signed distance function (SDF)
    <img src='static/17-27.png' width='600'>
    
    - 3D Shape Representations
        - Implicit Functions: Slide 17-29
    <img src='static/17-29.png' width='600'>
    
- Pointcloud
    - Definition: Slide 17-32
        - Good: Can represent fine structures without huge numbers of points
        - Neutral: Requires new architecture, losses, etc
        - Bad: Doesn’t explicitly represent the surface of the shape: extracting a mesh for rendering or other applications requires post-processing
            - JJ: we kinda need to inflate the point to ball-size (as shown in figures)
    <img src='static/17-32.png' width='600'>
    
    - Processing Pointcloud Inputs
        - PointNet: Slide 17-33
            - Want to process pointclouds as **sets** order should not matter
    - Generating Pointcloud Outputs
        <img src='static/17-33.png' width='600'>
        
        - Architecture: Slide 17-34
        <img src='static/17-34.png' width='600'>
        
        - Loss Function
            - We need a (differentiable) way to compare point clouds as **sets**!
            - Chamfer distance: Slide 17-37, 17-39
            <img src='static/17-37.png' width='600'>
            <img src='static/17-39.png' width='600'>

- Mesh
    - Definition
        - Slide 17-42
            - Good: Standard representation for graphics
            - Good: Explicitly represents 3D shapes
        <img src='static/17-42.png' width='600'>
        
        - Slide 17-43
            - Good: Adaptive: Can represent flat surfaces very efficiently, can allocate more faces to areas with fine detail
        <img src='static/17-43.png' width='600'>
        
        - Slide 17-44
            - Can attach data on verts and interpolate over the whole surface: RGB colors, texture coordinates, normal vectors, etc
        <img src='static/17-44.png' width='600'>
        
    - Predicting Triangle Meshes: Pixel2Mesh: Slide 17-47
        <img src='static/17-47.png' width='600'>
        
        - Iterative Refinement: Slide 17-48
        <img src='static/17-48.png' width='600'>
        
        - Graph Convolution: Slide 17-49, 17-51
        <img src='static/17-49.png' width='600'>
        <img src='static/17-51.png' width='600'>
        
        - Vertex Aligned-Features: Slide 17-52, 17-53
        <img src='static/17-52.png' width='600'>
        <img src='static/17-53.png' width='600'>
        
        - Chamfer Loss Function: Slide 17-55
            <img src='static/17-55.png' width='600'>
            
            - Steps: Slide 17-56, 17-57, 17-60
            <img src='static/17-56.png' width='600'>
            <img src='static/17-57.png' width='600'>
            <img src='static/17-60.png' width='600'>

**3D Metrics**
- Shape Comparison Metrics
    - Intersection over Union: Slide 17-68
    <img src='static/17-68.png' width='600'>
    
    - Chamfer Distance: Slide 17-70
        - JJ: sensitive to outliers since it relies on L2 distance
    <img src='static/17-70.png' width='600'>
    
    - F1 Score: Slide 17-74, 17-75
        - JJ: better loss
    <img src='static/17-74.png' width='600'>
    <img src='static/17-75.png' width='600'>

**3D Camera Systems**
- Cameras: Canonical vs View Coordinates
    - Definition: Slide 17-80
        - Canonical Coordinates
        - View Coordinates
        - JJ: a lot of people use view coordinates since it is easy to implement
    <img src='static/17-80.png' width='600'>
    
    - Problem: Slide 17-81, 17-82
    <img src='static/17-81.png' width='600'>
    <img src='static/17-82.png' width='600'>
    
- View-Centric Voxel Predictions: Slide 17-84
<img src='static/17-84.png' width='600'>

**3D Datasets**
- Object-Centric: Slide 17-88
    - ShapeNet
        - JJ: not a realistic dataset
    - Pix3D
        - JJ: real images!
<img src='static/17-88.png' width='600'>

**3D Shape Prediction**
- Mesh R-CNN
    - Task: Slide 17-90
    <img src='static/17-90.png' width='600'>
    
    - Hybrid 3D shape representation: Slide 17-92
        - Mesh deformation
        - Our approach
    <img src='static/17-92.png' width='600'>
    
    - Pipeline: Slide 17-96
    <img src='static/17-96.png' width='600'>
    
    - Results: Slide 17-97
    <img src='static/17-97.png' width='600'>
    
    - Shape Regularizers: Slide 17-98
    <img src='static/17-98.png' width='600'>
