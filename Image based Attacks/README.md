# Adversarial-Attack-Neural-Networks
Adversarial attacks are performed on neural networks to affect their ability to predict accurately. This could have adverse implications for the self-driving or healthcare industry.

## List of Attacks:
1. Fast Gradient Sign Method (FGSM) Attack
2. Basic Iterative Method (BIM) Attack
3. Projected Gradient Descent (PGD) Attack
4. DeepFool Attack
5. Jacobian-based Saliency Map Attack (JSMA) Attack
6. Boundary Attack
7. Carlini & Wagner (C&W) Attack
8. Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) Attack
9. Momentum Iterative FGSM (MI-FGSM) Attack
10. Zeroth Order Optimization (ZOO) Attack
11. Square Attack
12. HopSkipJump Attack

# Datasets 
## MNIST Dataset
The MNIST dataset is a collection of 70,000 handwritten digit images, split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes (digits 0 through 9).

## Fashion-MNIST Dataset
The Fashion-MNIST dataset is a collection of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

# Algorithms
## Fast Gradient Sign Method (FGSM) Attack on the MNIST Dataset
The Fast Gradient Sign Method (FGSM) is an adversarial attack technique that perturbs input images in a single step to fool a neural network into making incorrect predictions. FGSM works by leveraging the gradients of the loss function concerning the input image to create adversarial examples. Used on the MNIST dataset.

### Attack Implementation
#### Model Training:
- A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the digits.
- The CNN consists of two convolutional layers followed by two fully connected layers.
#### Generating Adversarial Examples:
- The FGSM attack is applied to the test images to generate adversarial examples.
- The attack perturbs the input images by adding a small step in the direction of the gradient of the loss with respect to the input images. This step size is controlled by the parameter epsilon (ε).
#### Visualization:
- The original images and their corresponding adversarial examples are visualized side by side.
- Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.


## Basic Iterative Method (BIM) Attack on MNIST Dataset
The Basic Iterative Method (BIM) is an adversarial attack technique that iteratively perturbs input images to fool a neural network into making incorrect predictions. This attack is an extension of the Fast Gradient Sign Method (FGSM), where multiple small steps are taken in the direction of the gradient to craft adversarial examples. Used on the MNIST dataset

### Attack Implementation
#### Model Training:
- A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the digits.
- The CNN consists of two convolutional layers followed by two fully connected layers.
#### Generating Adversarial Examples:
- The BIM attack is applied to the test images to generate adversarial examples.
- The attack iteratively adjusts the pixel values of the input images to maximize the model's prediction error, while keeping the perturbations within a specified limit (epsilon).
#### Visualization:
- The original images and their corresponding adversarial examples are visualized side by side.
- Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

The BIM attack demonstrates the vulnerability of neural networks to adversarial examples. By iteratively applying small perturbations, the attack effectively deceives the model while keeping the changes to the input images minimal. This highlights the importance of developing robust defenses against such adversarial attacks.

## Projected Gradient Descent (PGD) Attack on the MNIST Dataset
The Projected Gradient Descent (PGD) is an adversarial attack technique that iteratively perturbs input images to fool a neural network into making incorrect predictions. PGD is considered a strong attack as it performs multiple small steps in the direction of the gradient and projects the perturbations back into the allowed epsilon ball, ensuring that the perturbations are within a specified limit. Used on the MNIST dataset.

### Attack Implementation
#### Model Training:
- A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the digits.
- The CNN consists of two convolutional layers followed by two fully connected layers.
#### Generating Adversarial Examples:
- The PGD attack is applied to the test images to generate adversarial examples.
- The attack iteratively adjusts the pixel values of the input images by taking multiple small steps in the direction of the gradient. After each step, the perturbations are projected back into the epsilon ball around the original images.
#### Visualization:
The original images and their corresponding adversarial examples are visualized side by side.
Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

The PGD attack demonstrates the vulnerability of neural networks to adversarial examples by iteratively applying small perturbations and projecting them back into the allowed epsilon ball. This ensures that the perturbations are within a specified limit while effectively deceiving the model. The PGD attack highlights the need for developing robust defenses against such adversarial attacks.

## DeepFool Attack on the Fashion-MNIST Dataset
The DeepFool attack is an adversarial attack technique that iteratively perturbs input images to move them towards the decision boundary of the classifier. This process continues until the image is misclassified by the neural network. Used on the Fashion-MNIST dataset.

### Attack Implementation
#### Model Training:
- A simple Convolutional Neural Network (CNN) is trained on the Fashion-MNIST dataset to classify the images.
- The CNN consists of two convolutional layers followed by two fully connected layers.

#### Generating Adversarial Examples:
- The DeepFool attack is applied to the test images to generate adversarial examples.
- The attack iteratively perturbs the input images to move them towards the decision boundary of the classifier.

#### Visualization:
- The original images and their corresponding adversarial examples are visualized side by side.
- Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

The DeepFool attack demonstrates the vulnerability of neural networks to adversarial examples by iteratively perturbing the input images to move them toward the decision boundary of the classifier. This highlights the importance of developing robust defenses against such adversarial attacks.

## Jacobian-based Saliency Map Attack (JSMA) on the MNIST dataset
The Jacobian-based Saliency Map Attack (JSMA) is an adversarial attack designed to deceive machine learning models, especially those used for image classification tasks. JSMA generates adversarial examples by perturbing input features, like pixels in an image, to maximize the change in the model's output while minimizing the visibility of the perturbation. Used on the MNIST dataset. 

### Attack Implementation
#### Model Training:
- A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the images.
- The CNN consists of two convolutional layers followed by two fully connected layers.

#### Generating Adversarial Examples:
- The JSMA attack is applied to the test images to generate adversarial examples.
- Calculate a saliency map to identify the most important pixels in the image. This is done by computing the gradient of the model's output with respect to the input image.
- Perturb pixels to increase the likelihood of misclassification to the target class while keeping the perturbation visually imperceptible. Applies perturbations iteratively, gradually increasing the difference between the original and perturbed images.
- Stops when either the image is misclassified or a predefined perturbation budget is reached.
- The attack iteratively perturbs the input images to move them towards the target class selected by the user (in this example, it is 2).

#### Visualization:
- The original images and their corresponding adversarial examples are visualized side by side.
- Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

JSMA demonstrates the vulnerability of machine learning models to adversarial attacks and emphasizes the importance of developing robust models that can withstand such attacks. Defending against adversarial examples is critical for ensuring the reliability and security of machine learning systems in practical applications.


## Boundary Attack
The Boundary Attack is an iterative adversarial attack technique that generates adversarial examples by starting from a target image and gradually moving towards the original image while ensuring that the adversarial example remains misclassified. This attack does not require access to the gradients of the model, making it suitable for black-box scenarios. Used on the MNIST dataset.

### Attack Implementation
#### Model Training:
A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the digits. The CNN consists of two convolutional layers followed by two fully connected layers.

#### Generating Adversarial Examples:
The Boundary Attack is applied to the test images to generate adversarial examples.
The attack starts from a target image and iteratively moves towards the original image by taking small steps along the boundary that separates the original class from the target class.
The step size is controlled by the parameter epsilon (ε), ensuring that the perturbations are minimal and the adversarial examples remain visually similar to the original images.

#### Visualization:
Multiple sets of original images, target images, and their corresponding adversarial examples are visualized side by side.
Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

## Carlini & Wagner (C&W) Attack
The Carlini & Wagner (C&W) Attack is a powerful, optimization-based adversarial attack technique. It is designed to generate adversarial examples with minimal perturbation (specifically minimizing the L2 norm of the perturbation) while ensuring the model misclassifies the input to a specific, desired target class with a high degree of confidence. This attack typically requires full white-box access to the model's internal gradients, making it a benchmark for evaluating model robustness.

### Attack Implementation
#### Model Training:
For the provided Python code, a simple MLPClassifier (Multi-layer Perceptron) from scikit-learn is trained on the digits dataset. This model learns to classify grayscale images of handwritten digits (0-9), flattened into a 64-feature vector. The input data is normalized to the [0, 1] range to align with the attack's requirements.

#### Generating Adversarial Examples:
The C&W attack is applied to a selected test image. The core idea is to find a small perturbation that, when added to the original image, makes the model classify it as the target_class. This is achieved by formulating an optimization problem that balances two objectives:

Minimizing the L2 distance between the original and adversarial image.
Ensuring the logit (raw score) of the target_class is sufficiently higher than all other class logits (controlled by the kappa parameter).
The optimization is performed using scipy.optimize.minimize, which works in an unconstrained space after transforming the input using the arctanh function. The c_value parameter controls the trade-off between perturbation size and the misclassification confidence.


# L-BFGS Attack (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
The L-BFGS Attack is another white-box, optimization-based adversarial attack method that aims to generate adversarial examples with minimal L2 perturbation. Similar to the C&W attack, it is a targeted attack, meaning it attempts to force the model to classify the adversarial example into a specific, predetermined class. It leverages the L-BFGS-B algorithm, a quasi-Newton optimization method, to efficiently find these perturbations by approximating the Hessian matrix.

## Attack Implementation
#### Model Training:

In the Python implementation, an MLPClassifier from scikit-learn is trained on the digits dataset. This model learns to recognize handwritten digits. The dataset is preprocessed by scaling the pixel values to the [0, 1] range to ensure compatibility with the attack's mathematical transformations.

#### Generating Adversarial Examples:
The L-BFGS attack operates by minimizing a specially crafted objective function. This function consists of two primary components:

The L2 squared distance between the original input and the perturbed adversarial input, which encourages small perturbations.
A classification loss term that pushes the model's prediction towards the desired target_class with a certain confidence margin (kappa).
The attack transforms the input into an unconstrained space using the arctanh function, allowing the scipy.optimize.minimize function with the L-BFGS-B method to efficiently find the optimal perturbation. The c_value parameter is used to weight the importance of the classification loss relative to the perturbation size.

# Momentum Iterative FGSM (MI-FGSM)
The Momentum Iterative Fast Gradient Sign Method (MI-FGSM) is an iterative, gradient-based adversarial attack technique. It is an extension of the basic Iterative FGSM (BIM) that incorporates a momentum term. This momentum helps to stabilize the gradient updates across iterations and often leads to more robust and transferable adversarial examples. MI-FGSM is typically an untargeted attack, meaning its goal is to simply make the model misclassify the input into any incorrect class, rather than a specific one. It requires white-box access to the model's gradients.

## Attack Implementation
### Model Training:

The Python code utilizes a SimpleCNN (Convolutional Neural Network) implemented in PyTorch, trained on the well-known MNIST dataset. This CNN is designed for classifying grayscale images of handwritten digits. The training process involves standard supervised learning techniques, aiming for high accuracy on the clean MNIST test set.

### Generating Adversarial Examples:
The MI-FGSM attack iteratively perturbs an input image. In each iteration:
1. The model's output for the current perturbed image is obtained.
2. The gradient of the loss (with respect to the true label) is computed relative to the input image.
3. This gradient is normalized by its L1 norm and then integrated into a momentum term, which accumulates gradients over time.
4. A small step (alpha) is taken in the direction of the sign of the accumulated momentum.
5. The total perturbation is carefully clipped by an epsilon budget (L-infinity norm constraint) to ensure the adversarial example remains visually similar to the original.
6. The pixel values of the perturbed image are clamped to stay within the valid [0, 1] range.
7. This iterative process, guided by momentum, drives the image across the decision boundary of the model, leading to misclassification.

# Zeroth-Order Optimization (ZOO) Attack
The Zeroth-Order Optimization (ZOO) Attack is a powerful black-box adversarial attack technique. It is designed to generate adversarial examples by estimating the gradients of the model's loss function without direct access to its internal architecture, weights, or gradients. This makes it highly versatile for real-world scenarios where models are proprietary or only accessible via an API. ZOO is typically a targeted attack, aiming to misclassify an input into a specific desired class.

## Attack Implementation
### Model Training:
For the Python implementation, an MLPClassifier (Multi-layer Perceptron) from scikit-learn is trained on the digits dataset. This model acts as our "black-box" target, simulating a scenario where we can only query its predictions (probabilities) but cannot inspect its internal parameters or compute gradients directly. The digits dataset consists of 8x8 pixel images of handwritten digits, which are flattened into 64-feature vectors and normalized to the [0, 1] range.

### Generating Adversarial Examples:
The ZOO attack iteratively crafts an adversarial example by:

1. Initializing with a copy of the original input.
2. Estimating Gradients: For each feature (dimension) of the input, the attack makes small positive and negative perturbations (controlled by delta). It then queries the black-box model with these perturbed inputs to evaluate an objective function (similar to C&W's classification loss, accounting for target_class and kappa). The gradient for that dimension is approximated using a finite difference formula based on these objective values.
3. Updating: The adversarial example is updated by taking a small step in the negative direction of the estimated gradient (controlled by learning_rate), effectively performing gradient descent in the estimated gradient space.
4. Projecting: After each update, the perturbation is clipped to adhere to an epsilon_budget (L-infinity norm) relative to the original input, and the adversarial example's pixel values are clamped to the valid [0, 1] range.
5. This iterative process continues for total_iterations, progressively refining the adversarial example until it achieves the targeted misclassification.

# Square Attack
The Square Attack is a black-box, query-efficient adversarial attack. It operates without access to the target model's gradients, making it highly practical for real-world applications. This attack generates adversarial examples by iteratively adding small, randomly placed "square" shaped perturbations to the input image. It evaluates the effect of these perturbations by querying the model and accepts them only if they improve the attack's objective, aiming to induce targeted misclassification with a minimal L-infinity perturbation.

## Attack Implementation
### Model Training:
In the Python code, an MLPClassifier from scikit-learn is trained on the digits dataset. This model functions as the black-box classifier we aim to attack. The digits dataset, containing flattened 8x8 grayscale images of digits, is preprocessed by normalizing pixel values to the [0, 1] range, which is standard for image-based adversarial attacks.

### Generating Adversarial Examples:
The Square Attack iteratively creates an adversarial example:

1. Initialization: The attack typically starts with an initial adversarial example that is already misclassified (e.g., random noise within the epsilon budget) or the original input itself.
2. Patch Proposal: In each iteration, a random square-shaped region within the current adversarial image is chosen. The size of this square is influenced by the p_step parameter (a probability that relates to the proportion of features being updated).
3. Perturbation Generation: Random noise, scaled by the epsilon budget, is generated for this specific square region.
4. Candidate Evaluation: A new candidate adversarial example is formed by applying this noise within the square to the current adversarial image, while keeping the rest of the image unchanged.
5. Objective Evaluation: The black-box model is queried with this candidate to evaluate an objective function, which quantifies how well the candidate input is classified towards the target_class (considering a kappa confidence margin).
6. Acceptance Rule: If the candidate improves the objective (i.e., leads to a stronger misclassification towards the target), the candidate replaces the current adversarial example. Otherwise, it is discarded.
7. Clipping: All pixel values are clipped to the valid [0, 1] range after each update.
This iterative "trial-and-error" process efficiently explores the adversarial space, eventually finding a small, localized perturbation that fools the model into the desired target class.

# HopSkipJump Attack
The HopSkipJump Attack is a black-box, boundary-based adversarial attack that aims to find minimal perturbations (L2 norm) to misclassify an input. Unlike gradient-based methods, it does not require access to the model's internal gradients. Instead, it estimates the normal vector to the decision boundary through numerous model queries. This allows it to "hop" towards the original image and "jump" along the decision boundary, making it efficient in finding adversarial examples. It is typically used for untargeted attacks (forcing any misclassification) but can be adapted for targeted scenarios.

## Attack Implementation
### Model Training:
For the Python code, an MLPClassifier from scikit-learn is trained on the digits dataset. This model serves as the black-box target whose decision boundaries the attack will explore. The digits dataset's features (pixel values) are scaled to the [0, 1] range to ensure proper handling by the attack algorithm.

### Generating Adversarial Examples:
The HopSkipJump Attack works in an iterative fashion, maintaining an adversarial example that is already misclassified and continuously moving it closer to the original input while staying on the "wrong" side of the decision boundary:

1. Initialization: The attack begins by finding an initial adversarial example that is far from the original input but is already misclassified by the model. This is typically done by adding large random noise and performing a binary search to land on the decision boundary.
2. Normal Estimation: At the current adversarial point, the attack estimates the normal vector to the decision boundary. This is achieved by making multiple random queries around the current point and observing which queries cross the boundary (i.e., change classification). The average direction of these boundary crossings gives an approximation of the normal vector.
3. Hop Step: The current adversarial example is moved along the estimated normal vector towards the original input. This "hops" the adversarial example closer to the original.
4. Jump Step (Binary Search): After the "hop", a binary search is performed along the line segment connecting the current adversarial point and the original input. This "jumps" the adversarial example back onto the decision boundary, ensuring it remains misclassified while minimizing the distance to the original input.
5. Refinement: Small random steps (gamma) are sometimes added to help escape local minima and ensure convergence.
This iterative process continuously reduces the distance between the adversarial example and the original input, eventually yielding a subtle perturbation that causes misclassification.


## OBSERVATION:
while ZOO and Square attacks are valuable black-box techniques, their reliance on direct gradient approximation or random patch sampling can be less effective or require more tuning and queries compared to boundary-based attacks like HopSkipJump, especially in scenarios with complex decision boundaries or limited epsilon budgets. HopSkipJump's explicit strategy of traversing the decision boundary makes it particularly well-suited for finding minimal adversarial perturbations in black-box settings.
