# Adversarial-Attack-Neural-Networks
Repository aimed at documenting adversarial attacks on neural networks and potential ways to stop them

## BIM Attack on MNIST Dataset
The Basic Iterative Method (BIM) is an adversarial attack technique that iteratively perturbs input images to fool a neural network into making incorrect predictions. This attack is an extension of the Fast Gradient Sign Method (FGSM), where multiple small steps are taken in the direction of the gradient to craft adversarial examples.

MNIST Dataset
The MNIST dataset is a collection of 70,000 handwritten digit images, split into 60,000 training images and 10,000 test images. Each image is 28x28 pixels and belongs to one of 10 classes (digits 0 through 9).

Attack Implementation
Model Training:

A simple Convolutional Neural Network (CNN) is trained on the MNIST dataset to classify the digits.
The CNN consists of two convolutional layers followed by two fully connected layers.
Generating Adversarial Examples:

The BIM attack is applied to the test images to generate adversarial examples.
The attack iteratively adjusts the pixel values of the input images to maximize the model's prediction error, while keeping the perturbations within a specified limit (epsilon).
Visualization:

The original images and their corresponding adversarial examples are visualized side by side.
Despite minimal visual differences, the adversarial examples are crafted to deceive the model into making incorrect predictions.

The BIM attack demonstrates the vulnerability of neural networks to adversarial examples. By iteratively applying small perturbations, the attack effectively deceives the model while keeping the changes to the input images minimal. This highlights the importance of developing robust defenses against such adversarial attacks.
