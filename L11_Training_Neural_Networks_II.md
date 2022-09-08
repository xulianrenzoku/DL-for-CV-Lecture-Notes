Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 11: Training Neural Networks II

**Overview**
- Slide 11-11 
<img src='static/11-11.png' width='600'> 

**Training Dynamics**  
**Learning Rate Schedules**
- Viz about Choice of LR: Slide 11-15
    - SGD, SGD+Momentum, Adagrad, RMSProp, Adam all have **learning rate** as a hyper meter
    - JJ: We want the red curve!
    - Q: Which one of these learning rates is best to use?
    - A: All of them! Start w/ large learning rate and decay over time
        - Quick loss in the beginning
        - Converge to very low
    <img src='static/11-15.png' width='600'> 
- Learning Rate Decay
    - Step: Slide 11-16
        - Reduce learning rate at a few fixed points
            - e.g. for ResNets, multiply LR by 0.1 after epochs 30, 60, and 90.
            - JJ: Require a fair amount of trial and error, due to too many hyperparameters (# of iterations, etc)
    <img src='static/11-16.png' width='600'> 
    
    - Cosine: Slide 11-17
        - alpha_t = 1/2 * alpha_0 * (1 + cos(t * pi / T))
        - JJ: Only two hyperparameters to choice
            - alpha_0: Initial learning rate
            - T: Number of epochs used to train
        - JJ: Easier to tune than step
        - JJ: Training longer tends to better results 
        - JJ: Recently popular
    <img src='static/11-17.png' width='600'> 
    
    - Linear: Slide 11-18
        - alpha_t = alpha_0 * (1 - t / T)
        - JJ: Interesting trends:
            - CV projects often uses cosine
            - NLP projects often uses linear
    <img src='static/11-18.png' width='600'> 
    
    - Inverse Sqrt: Slide 11-19
        - alpha_t = alpha_0 / sqrt(t)
        - JJ: Learning rate drops very quickly
    <img src='static/11-19.png' width='600'> 
    
    - Constant: Slide 11-20
        - alpha_t = alpha_0 
        - JJ: I recommend people to do this if your goal is to get something as quickly as possible
        - JJ: If use Adam, use content will get you pretty far
    <img src='static/11-20.png' width='600'> 
- How long to train? **Early Stopping**
    - Slide 11-21
        - Stop training the model when accuracy on the validation set decreases or train for a long time, but alway keep track of the model snapshot that worked best on val. **Always a good idea to do this!**
    <img src='static/11-21.png' width='600'> 

**Hyperparameter Optimization**
- Choosing Hyperparameters
    - Grid Search: Slide 11-23
        - Choose several values for each hyperparameter (Often space choices log-linearly)
        - Evaluate all possible choices on this **hyperparameter grid**
        - JJ: get very infeasible very quickly
    <img src='static/11-23.png' width='600'> 
    
    - Random Search: Slide 11-24
        - Choose several values for each hyperparameter (Often space choices log-linearly)
        - Run many different trails
    <img src='static/11-24.png' width='600'> 
    
    - Random vs Grid Search
        - Slide 11-25
            - JJ: Random works better in general, since it is easy to get more samples from important parameters
    <img src='static/11-25.png' width='600'> 
- Choosing Hyperparameters (without tons of GPUs)
    - **Step 1**: Check initial loss
        - Turn off weight decay, sanity check loss at initialization
        - e.g. log(C) for softmax with C classes
        - JJ: Depending on your loss function, you can compute analytically what loss you would expect
    - **Step 2**: Overfit a small sample
        - Try to train to 100% training accuracy on a small sample of training data (~5-10 minibatches); fiddle with architecture, learning rate, weight initialization. Turn off regularization
            - Loss not going down? LR too low, bad initialization
            - Loss explodes to Inf or NaN? LR too high, bad initialization
        - JJ: The point of this step is to make sure you don’t have any bugs in your optimization loop
        - JJ: Training loop runs very very fast. 1 GPU would work
    - **Step 3**: Find LR that makes loss go down
        - Use the architecture from the previous step, use all training data, turn on small weight decay, find a learning rate that makes the loss drop significantly within ~100 iterations
            - Good learning rates to try: 1e-1, 1e-2, 1e-3, 1e-4
    - **Step 4**: Coarse grid, train for ~1-5 epochs
        - Choose a few values of learning rate and weight decay around what worked from Step 3, train a few models for ~1-5 epochs
        - Good weight decay to try: 1e-4, 1e-5, 0
        - JJ: Get you a sense beyond training set to see how it performed on val set
    - **Step 5**: Refine grid, train longer
        - Pick best models from Step 4, train them for longer (~10-20 epochs) without learning rate decay
    - **Step 6**: Look at learning curves
        - Slide 11-34
            - JJ: I always plot these two plots 
        <img src='static/11-34.png' width='600'> 
        
        - Slide 11-35
            - JJ: Bad initialization
        <img src='static/11-35.png' width='600'> 
        
        - Slide 11-36
        <img src='static/11-36.png' width='600'> 
        
        - Slide 11-37
            - JJ: Introduce decay too early
        <img src='static/11-37.png' width='600'> 
        
        - Slide 11-38
        <img src='static/11-38.png' width='600'> 
        
        - Slide 11-39
            - An example of overfitting
        <img src='static/11-39.png' width='600'> 
        
        - Slide 11-40
            - JJ: Unhealthy learning curve
        <img src='static/11-40.png' width='600'> 
    - **Step 7**: GOTO step 5
- Hyperparameters to play with
    - Network architecture
    - Learning rate, its decay schedule, update type
    - Regularization (L2/Dropout strength)
    - JJ: You’re like a DJ
- Track ratio of weight update / weight magnitude
    - Slide 11-44
    <img src='static/11-44.png' width='600'> 

**After Training**  
**Model Ensembles**
- Steps
    - 1. Train multiple independent models
    - 2. At test time average their results (Take average of predicted probability distributions, then choose argmax)
    - Enjoy 2% extra performance
- Tips and Tricks
    - Slide 11-47
        - Instead of training independent models, use multiple snapshot of a single model during training
    <img src='static/11-47.png' width='600'> 
    
    - Slide 11-48
        - Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time (Polka averaging)
        - JJ: used pretty commonly in some large-scaled generative models
    <img src='static/11-48.png' width='600'> 

**Transfer Learning**
- Myth: “You need a lot of data if you want to train/use CNNs”
    - JJ: WRONG! Using transfer learning, you can get away using deep learning for a lot of problems, even in the case you don’t have access to very large training sets
- Steps: Slide 11-59
    - 1. Train on ImageNet
    - 2. Use CNN as a feature extractor
        - Freeze these 
        - Remove last layer
    - 3. Bigger Dataset: **Fine-Tuning**
        - Continue training CNN for new task!
        - Some tricks
            - Train w/ feature extraction first before fine-tuning
            - Lower the learning rate: use ~1/10 of LR used in original training
            - Sometimes freeze lower layers to save computation
<img src='static/11-59.png' width='600'> 

- Architecture Matters!
    - Improvements in CNN architectures lead to improvements in many downstream tasks thanks to transfer learning!
- How to use transfer learning: Slide 11-66
<img src='static/11-66.png' width='600'> 

- Transfer learning is pervasive: 
    - Slide 11-69
        - It’s the norm, not the exception
    <img src='static/11-69.png' width='600'> 
    
    - Steps: Slide 11-70
    <img src='static/11-70.png' width='600'> 
    
    - Some very recent results have questions it
        - Slide 11-71
        <img src='static/11-71.png' width='600'> 
        
        - Slide 11-72
        <img src='static/11-72.png' width='600'> 
    - JJ’s take: Slide 11-73
    <img src='static/11-73.png' width='600'> 

**Distributed Training**
- Model Parallelism: Split Model Across GPUs
    - Idea #1: Run different layers on different GPUs: Slide 11-78
        - Problem: GPUs spend lots of time waiting
    <img src='static/11-78.png' width='600'> 
    
    - Idea #2: Run parallel branches of model on different GPUs: Slide 11-80
        - Problem: Synchronizing across GPUs is expensive; Need to communicate **activations** and **grad activations**
    <img src='static/11-80.png' width='600'> 
    
    - Typical way: Copy Model on each GPU, split data: Slide 11-84, 11-87
    <img src='static/11-84.png' width='600'> 
    
    <img src='static/11-87.png' width='600'> 
    
- Large-Batch Training: 
    - Challenge: Slide 11-90
        - Suppose we can train a good model with one GPU
        - **Goal:** Train for same number of epochs, but use larger minibatches. We want model to train K times faster!
        - Q: How to scale up to data-parallel training on K GPUs?
    <img src='static/11-90.png' width='600'> 
    
    - Tricks: 
        - Scale Learning Rates (Linearly): Slide 11-91
        <img src='static/11-91.png' width='600'> 
        
        - Learning Rate Warmup: Slide 11-92
        <img src='static/11-92.png' width='600'> 
