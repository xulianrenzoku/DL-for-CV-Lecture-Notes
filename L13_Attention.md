Michigan Online  
Deep Learning for Computer Vision  
Instructor: Justin Johnson  

Lecture 13: Attention

**Attention**
- Architecture
    - Sequence-to-Sequence with RNNs
        - Slide 13-5
            - Input: Sequence x_1, …, x_T
            - Output: Sequence y_1, …, y_T
            - Encoder: h_t = f_w(x_t, h_t-1)
        <img src='static/13-5.png' width='600'> 
        
        - Slide 13-6
            - From final hidden state predict: 
                - **initial decoder state** s_0
                - **Context vector** c (often c=h_T)
        <img src='static/13-6.png' width='600'> 
        
        - Slide 13-9
            - **Decoder**: s_t = g_U(y_t-1, h_t-1, c)
        <img src='static/13-9.png' width='600'> 
        
        - Slide 13-11
            - Problem: Input sequence bottlenecked through fixed-sized vector. What if T=1000?
            - Idea: use new content vector at each step of decoder!
        <img src='static/13-11.png' width='600'> 
        
    - Sequence-to-Sequence with RNNs and Attention
        - Slide 13-12
            - JJ: Encoder looks exactly the same
        <img src='static/13-12.png' width='600'> 
        
        - Slide 13-13
            - Compute (scaler) alignment scores
            - JJ: Alignment function. Fully-connected network. How much should we attend the encoder given the current state of the decoder
        <img src='static/13-13.png' width='600'> 
        
        - Slide 13-14
            - Normalize alignment scores to get **attention weights**
            - JJ: probability score
        <img src='static/13-14.png' width='600'> 
        
        - Slide 13-15
            - Compute context vector as linear combination of hidden states
        <img src='static/13-15.png' width='600'> 
        
        - Slide 13-16
            - This is all differentiable! Do not supervise attention weights - backprop through everything
        <img src='static/13-16.png' width='600'> 
        
        - Slide 13-17
            - Intuition: Context vector attends to the relevant part of the input sequence
        <img src='static/13-17.png' width='600'> 
        
        - Slide 13-18
            - Repeat: Use s_1 to compute new context vector c_2
        <img src='static/13-18.png' width='600'> 
        
        - Slide 13-20
        <img src='static/13-20.png' width='600'> 
        
        - Slide 13-21
            - Use a different context vector in each tilmestep of decoder
                - Input sequence not bottlenecked through single vector
                - At each tilmestep of decoder, context vector “looks at” different parts of the input sequence
        <img src='static/13-21.png' width='600'> 
        
        - Slide 13-26
            - JJ: Something we can notice is, this mechanism we built does not care the input we have is a sequence
        <img src='static/13-26.png' width='600'> 
        
- Examples
    - Image Captioning with RNNs and Attention
        - Slide 13-27
        <img src='static/13-27.png' width='600'> 
        
        - Slide 13-28
        <img src='static/13-28.png' width='600'> 
        
        - Slide 13-29
        <img src='static/13-29.png' width='600'> 
        
        - Slide 13-30
        <img src='static/13-30.png' width='600'> 
        
        - Slide 13-31
        <img src='static/13-31.png' width='600'> 
        
        - Slide 13-32
        <img src='static/13-32.png' width='600'> 
        
        - Slide 13-33
        <img src='static/13-33.png' width='600'> 
        
        - Slide 13-34
        <img src='static/13-34.png' width='600'> 
        
        - Slide 13-35
        <img src='static/13-35.png' width='600'> 
        
        - Slide 13-36
        <img src='static/13-36.png' width='600'> 
        
        - Slide 13-37
        <img src='static/13-37.png' width='600'> 
        
        - JJ: The structure of the model looks very similar to our seq2seq model
- Paper
    - Slide 13-45: X, Attend, and Y
<img src='static/13-45.png' width='600'> 

**Self-Attention**
- Attention Layer
    - JJ: Generalize from the idea of attention
    - Changes
        - Use dot product for similarity
            - Slide 13-46
            <img src='static/13-46.png' width='600'> 
            
            - Slide 13-47
            <img src='static/13-47.png' width='600'> 
            
            - Slide 13-48
            <img src='static/13-48.png' width='600'> 
            
            - Slide 13-49
            <img src='static/13-49.png' width='600'> 
            
        - Multiple **query** vectors
            - Slide 13-50
            <img src='static/13-50.png' width='600'> 
        
        - Separate **key** and **value**
            - JJ: we do this to use input vectors twice. Give model more flexibility
            - Slide 13-51
            <img src='static/13-51.png' width='600'> 
            
            - Slide 13-57
            <img src='static/13-57.png' width='600'> 
            
- Special case: Self-Attention Layer
    - One query per input vector: Slide 13-65
    <img src='static/13-65.png' width='600'> 
    
    - Consider permuting the input vectors
        - Slide 13-67: Queries and Keys will be the same, but permuted
        <img src='static/13-67.png' width='600'> 
        
        - Slide 13-68: Similarities will be the same, but permuted
        <img src='static/13-68.png' width='600'> 
        
        - Slide 13-69: Attention weights will be the same, but permuted
        <img src='static/13-69.png' width='600'> 
        
        - Slide 13-70: Values will be the same, but permuted 
        <img src='static/13-70.png' width='600'> 
        
        - Slide 13-71: Outputs will be the same, but permuted
        <img src='static/13-71.png' width='600'> 
        
        - Slide 13-72
            - Self-attention layer is **Permutation Equivariant** f(s(x)) = s(f(x))
            - Self-attention layer works on sets of vectors
        <img src='static/13-72.png' width='600'> 
        
        - Slide 13-73: Self attention doesn’t “know” the order of vectors it is processing!
        <img src='static/13-73.png' width='600'> 
        
        - Slide 13-74
            - In order to make processing position-aware, concatenate input with **positional encoding**
            - E can be learned lookup table, or fixed function
        <img src='static/13-74.png' width='600'> 
        
- Masked Self-Attention Layer
    - Slide 13-76: Intuition
        - Don’t let vector “look ahead” in the sequence 
        - Used for language modeling (predict next word)
    <img src='static/13-76.png' width='600'> 
- Multihead Self-Attention Layer
    - Slide 13-77
        - JJ: Actually used in practice commonly
    <img src='static/13-77.png' width='600'> 
- Example: CNN with Self-Attention
    - Slide 13-78
    <img src='static/13-78.png' width='600'> 
    
    - Slide 13-79
    <img src='static/13-79.png' width='600'> 
    
    - Slide 13-80
    <img src='static/13-80.png' width='600'> 
    
    - Slide 13-81
    <img src='static/13-81.png' width='600'> 
    
    - Slide 13-82
    <img src='static/13-82.png' width='600'> 
    
    - Slide 13-83
    <img src='static/13-83.png' width='600'> 

**Transformer**
- Three Ways of Processing Sequences: Slide 13-86
    - Recurrent Neural Networks
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
    - Famous paper: Attention is all you need
<img src='static/13-86.png' width='600'> 

- The Transformer
    - Steps
        - Slide 13-88
        <img src='static/13-88.png' width='600'> 
        
        - Slide 13-89: Self-Attention
        <img src='static/13-89.png' width='600'> 
        
        - Slide 13-90: Residual connection
        <img src='static/13-90.png' width='600'> 
        
        - Slide 13-91: Layer Normalization
        <img src='static/13-91.png' width='600'> 
        
        - Slide 13-92: MLP independently on each vector
        <img src='static/13-92.png' width='600'> 
        
        - Slide 13-93: Residual connection
        <img src='static/13-93.png' width='600'> 
        
        - Slide 13-94 Layer Normalization
        <img src='static/13-94.png' width='600'> 

    - Slide 13-95: Transformer Block
        - Input: Set of vectors x
        - Output: Set of vectors y
        - Self-attention is the only interaction b/w vectors!
        - Layer norm and MLP work independently per vector
        - Highly scalable, highly parallelizable
    <img src='static/13-95.png' width='600'> 
    
    - Slide 13-96: Transformer
        - A **Transformer** is a sequence of transformer blocks
    <img src='static/13-96.png' width='600'> 
    
    - Slide 13-97: Transfer Learning
        - “ImageNet Moment for NLP”
        - Pretraining
            - Download a lot of text from the internet
            - Train a giant Transformer model for language modeling
        - Finetuning
            - Fine-tune the Transformer on your own NLP task
    <img src='static/13-97.png' width='600'> 
    
    - Slide 13-102: Scaling up Transformers
        - JJ: As the model gets bigger, the better you train
    <img src='static/13-102.png' width='600'> 

**Summary**
- Adding **Attention** to RNN models lets them look at different parts of the input at each tilmestep
- Generalized **Self-Attention** is new, powerful neural network primitive
- **Transformers** are a new neural network model that only uses attention

