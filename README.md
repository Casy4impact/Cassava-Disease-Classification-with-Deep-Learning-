# Cassava-Disease-Classification-with-Deep-Learning-
Computer Vision for Cassava Diseases Detection, Identify and Classify Disease Symptoms

Agricultural Capacity is one of the most important benchmarks for any thriving economy. As such, it is almost aways a high-investment activity. The investment made in agriculture acan be rendered void due to the acternal factors that may be beyond easy detection and control.

This project will aim to employ machine learning for task of predicting and detecting illness in cassava crop for the purpose of better and improved agrocultural profit
Learn to apply computer vision for the detection of diseases and infections affecting the cassava crop. During the course of this project, you will learn to utilize Python libraries such as: OpenCV, Pandas, and PyTorch, etc.

## Business Problem and Rationale
### Challenges:
- Manual detection of cassava diseases is labor-intensive, subjective, and prone to errors, leading to significant crop losses.
- Delayed detection can result in the rapid spread of diseases like Cassava Mosaic Disease, Cassava Brown Streak Disease, and Cassava Bacterial Blight, which have devastating effects on yield.
- Farmers often lack the expertise to accurately diagnose cassava diseases.
- 
### Rationale:
- Cassava is a staple crop with global importance.
- Early disease detection is crucial to prevent losses that can reach up to 90% of the expected yield.
- Early detection can help safeguard food security, support sustainable farming practices, and reduce diagnostic costs.

## Project Aims
- Develop Disease Detection Models: Build computer vision models that can accurately detect diseases in cassava crops based on leaf and spectral images.
- Model Engineering and Postprocessing: Optimize the size of the models to enable fast inference without sacrificing accuracy.
- Model Deployment: Deploy the trained models in a practical setting to support real-time detection by farmers in the field.

## Tech Stack and Tools
1. Programming Language: Python
2. Key Libraries:
- matplotlib: Visualization
- Pandas: For data preparation and handling.
- Nympy: For data manipulation
- OpenCV: For image loading and manipulation.
- PyTorch: For building deep learning models to detect cassava diseases.

## Data Description
- Cassava Leaf Images: Large dataset of healthy and diseased cassava leaf images.
- Cassava Spectral Images: Spectral images for detailed disease diagnosis.
- Disease Labels: Accurate labels specifying the disease type and severity.

## Project Scope
1. Image Preparation and Model Development:
- Data Exploration: Handle any quirks like data imbalance and process the data for model training.
- Model Training: Build and train deep learning models (likely convolutional neural networks) using the image data.

3. Model Evaluation:
- Use metrics like accuracy, precision, recall, and F1-score to evaluate model performance.
- Select the best model for deployment after testing generalizability and robustness.

4. Model Engineering and Postprocessing:
- Apply optimization techniques such as quantization and model distillation to improve performance and reduce model size.
- Aim for faster inference times to make the model suitable for real-time disease detection.

5. Model Deployment:
- Deploy the optimized model for use in Novel Farms' operations, where it can assist farmers in detecting diseases in real time.

This project has immense potential, both technically and for the positive social and economic impact it could have. Once deployed, it will likely save Novel Farms significant resources and support smaller farms through efficient, data-driven disease detection.

## Methodology
This project will be carried out using the Cross Industry Standard Process for Data Mining (CRISP-DM) Methodology. This is one of the most popular data science methodologies and it's characterised by six important phases:

1. Business Understanding,
2. Data Understanding,
3. Data Preparattion,
4. Data Modeling,
5. Model Evaluation, and
6. Model Deployment.
   
It should be noted that these phases are recurrent in nature (i.e. some phases may be repeated). As such, they do not necessarily follow a linear progression.

## Project Implementation via CRIP-DM
we aim to compare four models to determine the best fit for detecting and classifying cassava diseases through images. The models under evaluation are:
1. VGG-13 Fine-Tuned
2. VGG-13 Frozen
3. ResNet-18 Fine-Tuned
4. ResNet-18 Frozen

To determine which model best suits the project, we’ll compare them using the following metrics:
1. Loss: A measure of how well the model predicts the correct class. Lower values indicate better performance.
2. Accuracy: The percentage of correct predictions. Higher accuracy reflects better model performance.
3. Signs of Overfitting: This occurs when the model performs well on the training data but poorly on the test data, which indicates it has memorized the training data rather than generalizing to new data. Less overfitting is better for robust model performance.
   
1. VGG-13 Fine-Tuned
- Training Accuracy: 99% by epoch 20.
- Test Accuracy: Peaks at 73.9% but declines to 70.4% by epoch 20.
- Overfitting: This model suffers from significant overfitting, with training accuracy nearing 99% but test accuracy falling to 70.4%. The increasing test loss further indicates overfitting.
- Strengths: Fine-tuning VGG-13 leads to very high training accuracy, which suggests that the model has the capacity to learn intricate patterns from the training data.
- Weaknesses: The high gap between training and test accuracy signals that the model is overfitting to the training data. The generalization to unseen test data is poor, making it unreliable in real-world applications.

2. VGG-13 Frozen
- Training Accuracy: 99% by epoch 20.
- Test Accuracy: Peaks at 74%, ending around 71.3%.
- Overfitting: While there is still overfitting, it is less severe than the fine-tuned version. Test accuracy remains relatively stable, but the increasing test loss signals some degree of overfitting.
- Strengths: VGG-13 Frozen performs slightly better than the fine-tuned version, with more stability in test accuracy and less pronounced overfitting. Freezing the pre-trained layers helps retain the learned general features.
- Weaknesses: Despite better performance, the model still suffers from overfitting. Freezing layers limits its adaptability to specific features of cassava diseases, which might be causing the lower test accuracy compared to other models.

3. ResNet-18 Fine-Tuned
- Training Accuracy: 97.6% by epoch 20.
- Test Accuracy: Peaks at 78.1% (epoch 12) and drops to 74.4% by epoch 20.
- Overfitting: ResNet-18 Fine-Tuned achieves higher test accuracy than the VGG-13 models, with a smaller gap between training and test accuracy. While there is mild overfitting, it is more controlled.
- Strengths: Fine-tuning ResNet-18 provides the highest test accuracy (78.1%), outperforming both VGG models. The model's deeper architecture enables it to capture more complex features, which helps in better classification of cassava disease symptoms.
- Weaknesses: Although ResNet-18 Fine-Tuned shows less overfitting, there is still a noticeable gap between training and test performance, suggesting that some generalization issues remain.

4. ResNet-18 Frozen
- Training Accuracy: Improves to 69.8% by epoch 3.
- Test Accuracy: Peaks at 72.2% and remains consistent without large fluctuations.
- Overfitting: ResNet-18 Frozen exhibits minimal overfitting, with consistent test accuracy throughout training. However, its overall accuracy is lower compared to the fine-tuned version.
- Strengths: ResNet-18 Frozen has stable performance, with no significant overfitting. This makes it a reliable option for tasks where stability is preferred over the highest possible accuracy.
- Weaknesses: The test accuracy (72.2%) is lower compared to the fine-tuned version. Freezing the model limits its ability to adapt to the specific nuances of the cassava disease dataset.

## Comparison and Conclusion
Based on the comparison:
- VGG-13 Fine-Tuned shows significant overfitting, with a large gap between training and test accuracy.
- VGG-13 Frozen performs better, with more stable test accuracy, but still suffers from increasing test loss.
- ResNet-18 Fine-Tuned achieves the best test accuracy (78.1%) and exhibits better control over overfitting.
- ResNet-18 Frozen has minimal overfitting but lags behind in accuracy compared to the fine-tuned version.

## Best Model: ResNet-18 Fine-Tuned
Among the four models, ResNet-18 Fine-Tuned stands out as the best fit for this project. It strikes the right balance between high test accuracy (78.1%) and manageable overfitting, making it the most effective model for detecting and classifying cassava disease symptoms.


