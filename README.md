

# **Fashion MNIST Classifier**

![Texte alternatif…](https://res.cloudinary.com/practicaldev/image/fetch/s---fNWEeWA--/c_imagga_scale,f_auto,fl_progressive,h_420,q_auto,w_1000/https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)



## **Overview**
Welcome to the **Fashion MNIST Classifier**, a deep learning project that demonstrates the power of **neural networks** for real-world applications. This project involves training a neural network to accurately classify fashion items such as shirts, trousers, and sneakers from the **Fashion MNIST** dataset. Built with **Keras** and **TensorFlow**, this project showcases my proficiency in implementing and deploying neural network architectures.

### **Key Highlights:**
- **Neural Network Architecture**: A carefully designed feed-forward neural network (Multi-Layer Perceptron) for classifying fashion images.
- **Real-World Application**: Built to demonstrate how deep learning models can be applied to practical tasks, such as image classification.
- **Interactive Web Application**: Users can upload images and receive instant predictions through a user-friendly **Streamlit** web app interface.
- **Model Performance**: Achieved a test accuracy of over **88%**, highlighting the model's ability to generalize on unseen data.

## Fashion MNIST
![LeNet](image/LeNet_Original_Image.jpg)
Yann LeCun introduced Convolutional Neural Network (CNN for short) through [his paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), namely **LeNet-5**, and shows its effectiveness in hand-written digits. The dataset used his paper is called ["Modified National Institute of Standards and Technology"](http://yann.lecun.com/exdb/mnist/)(or MNIST for short), and it is widely used for validating the neural network performance. 

![MNIST](image/220px-MnistExamples.png)

Each image has 28x28 shapes, and is grayscaled (meaning that each pixel value has a range from 0 to 255). But as you notice from original image, features for each digits are almost clear, so most of neural network in now can easily learn its dataset. And also the task cannot represent the complicated task. So there are many trials to formalize its baseline dataset. One of these is [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist), presented by Zalando research. Its dataset also has 28x28 pixels, and has 10 labels to classify. So main properties are same as Original MNIST, but it is hard to classify it. 

![FashionMNIST](image/fashionMNIST.png)

In this post, we will use Fashion MNIST dataset classification with tensorflow 2.x. For the prerequisite for implementation, please check the previous posts.

## **Screenshots**
Here are some screenshots of the project in action:

### **1. Training Process**
![Training Process](assets/training_process.png) *(Replace with your actual screenshot)*

### **2. Web App Interface**
![Web App Interface](assets/web_app_interface.png) *(Replace with your actual screenshot)*

### **3. Prediction Results**
![Prediction Results](assets/prediction_results.png) *(Replace with your actual screenshot)*

## **Features**
- **Deep Learning**: Utilizes a **feed-forward neural network** to classify images into one of 10 fashion categories.
- **Data Preprocessing**: Normalizes image data for improved model training and performance.
- **Web Deployment**: Built with **Streamlit**, making it easy to deploy and share with others.
- **Interactive UI**: Users can upload their own images to see real-time predictions.

## **Installation**

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Fashion-MNIST-Classifier.git
    cd Fashion-MNIST-Classifier
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Train the model** (Optional):
    If you want to train the model from scratch, run:
    ```bash
    python src/train.py
    ```

4. **Run the web app**:
    ```bash
    streamlit run app/app.py
    ```

## **Technical Details**

### **Neural Network Architecture**
The neural network consists of the following layers:
- **Input Layer**: Flattens 28x28 greyscale images into a single vector of 784 features.
- **Hidden Layer**: Dense layer with 128 neurons and ReLU activation to capture complex patterns.
- **Output Layer**: Dense layer with 10 neurons (one for each class) and softmax activation for multi-class classification.

### **Dataset**
The **Fashion MNIST** dataset consists of **70,000 greyscale images** divided into **60,000 training images** and **10,000 test images**. Each image is 28x28 pixels, representing one of 10 fashion categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

### **Model Training**
The model was trained using:
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 40
- **Batch Size**: 128

The training process was monitored, and the model was fine-tuned using validation data to ensure it performs well on unseen data.

## **Results**
The model achieved an accuracy of **88%** on the test set, demonstrating its ability to generalize well. Here is a confusion matrix showcasing the performance:
![Confusion Matrix](assets/confusion_matrix.png) *(Replace with your actual screenshot)*

## **Usage**
1. **Upload an Image**: Use the web interface to upload an image of a fashion item.
2. **Receive Prediction**: The app will display the predicted class label based on the trained neural network.
3. **Evaluate Model**: Experiment with different inputs to see how well the model performs.

## **Real-World Applications**
This project showcases how **neural networks** can be applied to real-world problems. Image classification is a fundamental task in computer vision, with applications ranging from e-commerce (automated tagging of products) to healthcare (classifying medical images). The techniques demonstrated here can be extended to other domains requiring accurate and efficient classification.

## **Contributions**
I welcome contributions! If you have suggestions to improve this project or want to add new features, please feel free to **fork** the repository and create a **pull request**.

## **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## **Contact**
For any inquiries or collaboration opportunities, please reach out to:
- **Name**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

