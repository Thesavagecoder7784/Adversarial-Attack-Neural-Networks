# Adversarial-Attack-Neural-Networks
Repository aimed at documenting adversarial attacks on neural networks and potential ways to stop them

## List of Attacks:
1. Fast Gradient Sign Method (FGSM) Attack
2. Basic Iterative Method (BIM) Attack
3. Projected Gradient Descent (PGD) Attack
4. DeepFool Attack


## Fast Gradient Sign Method (FGSM) Attack on the MNIST Dataset
The Fast Gradient Sign Method (FGSM) is an adversarial attack technique that perturbs input images in a single step to fool a neural network into making incorrect predictions. FGSM works by leveraging the gradients of the loss function concerning the input image to create adversarial examples.

### MNIST Dataset
The MNIST dataset is a collection of 70,000 handwritten digit images, split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes (digits 0 through 9).

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
The Basic Iterative Method (BIM) is an adversarial attack technique that iteratively perturbs input images to fool a neural network into making incorrect predictions. This attack is an extension of the Fast Gradient Sign Method (FGSM), where multiple small steps are taken in the direction of the gradient to craft adversarial examples.

### MNIST Dataset
The MNIST dataset is a collection of 70,000 handwritten digit images, split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes (digits 0 through 9).

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
The Projected Gradient Descent (PGD) is an adversarial attack technique that iteratively perturbs input images to fool a neural network into making incorrect predictions. PGD is considered a strong attack as it performs multiple small steps in the direction of the gradient and projects the perturbations back into the allowed epsilon ball, ensuring that the perturbations are within a specified limit.

### MNIST Dataset
The MNIST dataset is a collection of 70,000 handwritten digit images, split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes (digits 0 through 9).

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
The DeepFool attack is an adversarial attack technique that iteratively perturbs input images to move them towards the decision boundary of the classifier. This process continues until the image is misclassified by the neural network.

### Fashion-MNIST Dataset
The Fashion-MNIST dataset is a collection of Zalando's article images, consisting of 60,000 training examples and 10,000 test examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

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
