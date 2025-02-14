# Vision Transformer for Image Classification with Transfer Learning
This project demonstrates how to perform image classification using a Vision Transformer (ViT) and transfer learning. We leverage a pre-trained ViT model and fine-tune it on the Oxford-IIIT Pets Dataset to classify different breeds of cats and dogs.
## About Transfer Learning
Transfer learning is a powerful technique in machine learning where a model trained on one task is reused as a starting point for a related task. In this project, we utilize a ViT model pre-trained on a massive image dataset (ImageNet) and adapt it for our pet breed classification task.

**Benefits of Transfer Learning:**
* **Reduced training time and resources:** Leveraging pre-trained models significantly reduces the computational cost and time required to train a new model from scratch.
* **Improved performance:** Pre-trained models have learned rich feature representations from vast amounts of data, which can be beneficial for related tasks, even with limited training data.
* **Faster convergence:** Fine-tuning a pre-trained model often leads to quicker convergence during training compared to training a model from scratch.

## Here is a detailed breakdown of the steps:
This notebook demonstrates transfer learning for image classification using a Vision Transformer (ViT) and the Oxford-IIIT Pets Dataset. Here's a breakdown of the steps:
1. Setup: Installs necessary libraries (evaluate, transformers, datasets), imports modules, and logs into Hugging Face Hub.
2. Data Loading: Loads the Oxford-IIIT Pets Dataset from Hugging Face Datasets. This dataset contains images of cats and dogs of different breeds.
3. Data Preprocessing:
- Splits the dataset into training, validation, and test sets.
- Creates mappings between label names and integer IDs.
- Defines an image processor using AutoImageProcessor to resize, normalize, and prepare images for the ViT model.
- Applies transformations to the dataset using with_transform.
- Defines a data collator to batch the data correctly for the model.
4. Metric Definition: Defines the accuracy metric using the evaluate library.
5. Model Loading:
- Loads a pre-trained ViT model (google/vit-base-patch16-224) using ViTForImageClassification.
- Modifies the final classification layer to match the number of labels in the dataset.
- Freezes the weights of all layers except the classification layer to perform transfer learning.
6. Training:
- Defines training arguments using TrainingArguments, including batch size, learning rate, and number of epochs.
- Creates a Trainer instance to manage the training process.
- Trains the model using trainer.train().
7. Evaluation:
- Evaluates the trained model on the test set using trainer.evaluate().
8. Prediction Visualization:
- Makes predictions on a subset of the test set.
- Displays the images along with their predicted and actual labels.
9. Model Saving:
- Saves the trained model locally.
- Pushes the model to the Hugging Face Model Hub for sharing and reuse.

In essence, the notebook utilizes a pre-trained ViT model, adapts it for a specific image classification task (pet breed classification), fine-tunes it on the provided dataset, and finally evaluates its performance. The use of transfer learning significantly reduces the training time and resources required, while achieving good accuracy. 

## Dataset
The **Oxford-IIIT Pets Dataset** is used for this project. It contains images of 37 different cat and dog breeds. You can access the dataset on Hugging Face Datasets: [pcuenq/oxford-pets](https://huggingface.co/datasets/pcuenq/oxford-pets)

## Model
We use the **google/vit-base-patch16-224** model from Hugging Face Model Hub as our base model. This is a Vision Transformer pre-trained on ImageNet.

## Results
The fine-tuned ViT model achieves high accuracy on the Oxford-IIIT Pets Dataset, demonstrating the effectiveness of transfer learning for image classification.


## Acknowledgments
* Hugging Face for providing the pre-trained model, dataset, and libraries.
* Google Colab for providing a free cloud-based environment for running this project.

## License
This project is licensed under the MIT License.
