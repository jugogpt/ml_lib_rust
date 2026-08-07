Hello folks.

In this project, I will be attempting to program an ML library in rust from first principals.


To start, what is fundamentally necessary to creating an ML library?
1. We need to be able to do Math (vectors, matrices, broadly tensors)
2. We need some way to define and find gradients 
3. Helper components to tie everything together 


Close your ears DL enjoyers, but we can first think of ML has just mapping inputs to outputs: x -> y; this could be (like our example) mapping some image of a number to a vector.

We associate fucntions with \Theta- parameters. This increases (crudely) the amount of possibilities we achieve from a set of inputs.

We also need a cost function C(x, y, theta); A cost function quantifies how bad the function/model is using the parameters.


The entire goal of machine learning is to try to minimize the cost function by adjusting the parameters during training.

Even for our simple network, the cost function is in some crazy n dimensional space

the gradient tells us the direction we need to move in order to minimize the gradient. 

The "Helper" part of our library will go through all of our training examples and try to update the gradient in order to aid us in finding this minimum and automatically adjust our gradients

