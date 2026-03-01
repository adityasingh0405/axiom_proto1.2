# Fundamentals of Deep Learning (Assignment 1)

## Key Concepts
- **Deep Learning vs Shallow Machine Learning**: Comparison using real datasets (Iris, MNIST subset, or any CSV dataset) [(Document - Page 1)](http://example.com/fdl_assignment1_page1)
	+ Accuracy
	+ Training time
	+ Overfitting behavior
	+ Feature engineering requirement
- **Gradient Descent and Batch Optimization**: Demonstration of Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent for a simple regression task [(Document - Page 6)](http://example.com/fdl_assignment1_page6)
	+ Convergence speed and stability comparison
- **Biological Neuron vs Artificial Neuron**: Comparison of biological neuron components with artificial neuron components [(Document - Page 3)](http://example.com/fdl_assignment1_page3)
	+ Diagram illustrating similarities/differences

## Visual Elements
- **Decision Tree Model Overfitting** [(FDL_assignment1 - Chunk 0, Page 3)](http://example.com/fdl_assignment1_chunk0_page3): A deep decision tree model trained on raw MNIST pixel data exhibits clear overfitting behavior due to high model complexity.
- **Deep Neural Network Model Performance** [(FDL_assignment1 - Chunk 2, Page 4)](http://example.com/fdl_assignment1_chunk2_page4): A deep neural network model shows better generalization behavior compared to the decision tree, learning hierarchical feature representations directly from data.

## Detailed Explanation
### Deep Learning vs Shallow Machine Learning

*   The shallow machine learning model (decision tree) is prone to overfitting due to its high complexity and training on raw pixel data.
*   The deep neural network model achieves higher accuracy by learning hierarchical feature representations directly from the data, enabling it to generalize effectively to unseen data.

### Gradient Descent and Batch Optimization

*   **Batch Gradient Descent**: Updates the model using the entire dataset at each step, resulting in stable and smooth convergence but slower updates.
*   **Stochastic Gradient Descent**: Updates the model using only one randomly selected data point at a time, leading to faster updates but introducing high variability and fluctuations in the cost function.
*   **Mini-Batch Gradient Descent**: Provides a balanced approach by using small subsets of data for each update, achieving a good compromise between speed and stability.

### Biological Neuron vs Artificial Neuron

*   The artificial neuron is designed to mimic the behavior of a biological neuron, with weighted sum and activation function (ReLU/Sigmoid).
*   The diagram illustrates similarities and differences between biological and artificial neurons.

## Important Formulas/Definitions
- **Weighted Sum**: A mathematical operation used in neural networks to calculate the weighted sum of inputs.
- **Activation Function**: A mathematical function used in neural networks to introduce non-linearity, such as ReLU (Rectified Linear Unit) or Sigmoid.

## Conclusion

*   The two models demonstrate clear differences in learning behavior and generalization performance.
*   The decision tree model exhibits strong overfitting due to its high depth and training on raw pixel data, while the deep neural network model achieves higher accuracy by learning hierarchical feature representations directly from the data.