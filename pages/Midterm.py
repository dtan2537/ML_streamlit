import streamlit as st

st.title("Modelling Student Stress")

st.header("Midterm Checkpoint")

st.subheader("Introduction and Problem Definition")
st.markdown("""
Student stress significantly affects academic performance, mental health, and well-being. 
It arises from multiple factors, including anxiety, sleep quality, peer pressure, and academic performance. 
To provide targeted support, understanding these interconnected influences is essential. 
We aim to develop a predictive model that accurately determines student stress levels during a semester, enabling proactive interventions to enhance well-being.

Prior research highlights factors contributing to student stress, including psychological (anxiety, depression), physiological (sleep issues, headaches), and social influences (peer pressure, bullying). 
Machine learning models like decision trees, SVMs, and neural networks have been employed to predict stress levels effectively. 
For instance, Choi et al. (2019) used random forests with physiological data, and Wijaya et al. (2021) applied logistic regression to explore academic and social stress factors, showing promising results for intervention strategies.
""")


st.subheader("Methods")
st.markdown("#### Data Preprocessing")
st.markdown("""
The preprocessing methods that our group chose to initially focus on were data standardization and dimensionality reduction. 
We require data standardization to maintain numerical stability throughout the training of our models as the varying feature magnitudes in our dataset span from very small to quite large, creating the potential for overflow or underflow. 
Additionally, since many algorithms we apply later on do not inherently handle scaling, handling this standardization during preprocessing helps our models converge later on. 
We then focus on dimensionality reduction to help the generalizability of our model, reducing the likelihood of overfitting and eliminating features that are very highly correlated. 
These methods are essential to our preprocessing and significantly improve our model results and training time.
            
First, we discuss our approach for data standardization.
""")
st.image("visuals/stress_dataset.png")
st.markdown("""
A first look at the data shows how there are varying magnitudes. We can further see that from the distribution of the features:
""")
st.image("visuals/dataset_distribution.png")
st.markdown("""
Thus, we would first need to **standardize** the data. 
This would ensure that features of higher magnitudes do not dominate features of lower magnitudes. 
For example, depression has a much higher magnitude compared to bullying. 
Hence, we use **sklearn.preprocessing.StandardScaler** to scale the features. 
""")
st.image("visuals/StandardScaler.png")
st.markdown("""
After appropriate scaling, each feature will be centered with a mean of 0 and of 1.

Next, we decide to use Principal Component Analysis (PCA) to perform our feature reduction. 

First, we perform One-Hot Encoding on the target variable. Let's look at stress. 
It takes the discrete values 0, 1 and 2. To make the model more probabilistic and prevent ordinality issues, we decide to use One-Hot Encoding. 
For that we have **sklearn.preprocessing.OneHotEncoder**.

With this encoding, we can then run the following to find a correlation matrix for all features.
""")
st.image("visuals/corr_matrix_code.png")
st.image("visuals/feature_correlation_matrix.png")
st.markdown("""
Looking at the feature correlation matrix, we can see that some features such as anxiety level and future career concerns are highly correlated, and may be redundant. 
This is where PCA comes in. 
We want to combine these 20+ features down to a few engineered features that are better able to capture the overall variance exhibited by the dataset. 
We call the best of these features **Principle Components**, and we can use just the first few to represent the entirety of our dataset.
""")
st.image("visuals/PCA_Rep.png")
st.markdown("""
Here, we can see that just the first two principal components are sufficient for an accurate classification of a large proportion of our test data points. 
By looking at the following chart, we notice that over 60% of the dataset’s variance is explainable by just these two principal components, serving as an effective method for significant dimensionality reduction.
""")
st.image("visuals/Expected_Variance_graph.png")
st.markdown("""
To further improve future performance, we are now going to look at polynomial features. 
We will accomplish this by checking metrics on each degree polynomial. 
Additionally, we will work on feature selection and potentially try to create new features by examining relationships between existing features. 
""")

st.markdown("#### Models")
st.markdown("""
For our first model, we decided to use a Support Vector Machine (SVM). Specifically, we implemented a linear kernel with a One-vs-One decision function shape. We chose this model because we felt it was best suited for the problem at hand: identifying the stress level of students given a few features relating to social behavior, academic performance, and physical health. There are a few specific reasons for choosing an SVM over other popular models:

1. SVM’s have good performance for high-dimensional spaces like ours.
2. Overfitting is mitigated due to the SVM’s regularization parameter
3. SVM’s are also less sensitive to the outliers in the stress dataset due to the support vectors used to implement it
4. SVM’s are well suited for multi-class problems like identifying stress level

Looking at the combined use of SVM and PCA, a few key problems are addressed. Irrelevant or redundant features are not weighted as heavily and training speed is increased. Overall, the model and preprocessing choice worked well in conjunction as proven by our performance metrics.

""")

st.subheader("Results and Discussion")
st.markdown("#### Confusion Matrix Analysis")
st.image("visuals/Confusion_Matrix.png")
st.markdown("""
The confusion matrix visualization shows the model's prediction performance across different classes using a blue-scale heatmap. Key observations:

- The diagonal elements show strong prediction accuracy, indicating the model performs well at classifying all three classes
- The intensity of blue squares along the diagonal suggests balanced performance across classes
- There appears to be minimal confusion between classes, with relatively low off-diagonal values
- The symmetrical nature of misclassifications suggests no systematic bias toward any particular class
""")
st.markdown("#### Classification Report Metrics")
st.image("visuals/Report.png")
st.markdown("""
The detailed classification report reveals:

- Overall accuracy: 90% across all classes
- Per-class metrics:
    - Class 0: Precision 0.87, Recall 0.94, F1-score 0.90
    - Class 1: Precision 0.92, Recall 0.88, F1-score 0.90
    - Class 2: Precision 0.91, Recall 0.88, F1-score 0.90

The consistent F1-scores around 0.90 across all classes indicates well-balanced performance without significant bias toward any particular class.
""")

st.markdown("#### ROC Curve Analysis")
st.image("visuals/ROC_Curve.png")
st.markdown("""
The ROC curves plot demonstrates the model's discrimination ability:

- All three class-specific ROC curves show strong performance, with AUC values consistently above 0.90
- The curves rise sharply toward the upper-left corner, indicating good true positive rates with low false positive rates
- The similar shapes and AUC values across classes further supports balanced classification performance
- The significant separation from the diagonal reference line confirms the model's strong predictive power
""")

st.markdown("#### Key Insights")
st.markdown("""
1. The model achieves strong overall performance with 90% accuracy
2. Performance is remarkably balanced across all three classes
3. High precision and recall values indicate both good positive prediction and good coverage
4. ROC curves demonstrate excellent discrimination ability for all classes
""")

st.subheader("References")
st.markdown("""
[1] H. Choi, S. Lee, and J. Kim, "Predicting Student Stress Levels Using Random Forest Algorithm with Physiological and Environmental Data," *Journal of Educational Data Science*, vol. 14, no. 3, pp. 123-134, 2019.

[2] S. S. Hudd, J. Dumlao, D. Erdmann-Sager, D. Murray, E. Phan, N. Soukas, and N. Yokozuka, "Stress at College: Effects on Health Habits, Health Status, and Self-Esteem," *College Student Journal*, vol. 34, no. 2, pp. 217-227, 2000.

[3] M. Pörhölä, S. Karhunen, and S. Rainivaara, "Bullying and Social Exclusion Among University Students: The Role of Group Dynamics," *Social Psychology of Education*, vol. 22, no. 1, pp. 189-206, 2019.

[4] D. J. Taylor, C. E. Gardner, A. D. Bramoweth, J. M. Williams, B. M. Roane, and E. A. Grieser, "Insomnia and Mental Health in College Students," *Behavioral Sleep Medicine*, vol. 9, no. 2, pp. 107-116, 2010.

[5] A. Wijaya, D. Nugroho, and R. Putra, "Assessing the Impact of Academic and Social Factors on Student Stress Using Logistic Regression," *International Journal of Educational Research and Development*, vol. 17, no. 4, pp. 456-465, 2021.
""")

st.header("Charts")

st.subheader("Gantt Chart")
st.markdown("Link: [Gantt Chart](https://docs.google.com/spreadsheets/d/1scnAtyIJyvz2WJbtioxEO4HTIOQOQ7Kt/edit?usp=sharing&ouid=101934921339602004791&rtpof=true&sd=true)")
st.subheader("Contribution Chart")

chart = {
    "Name" : ["Jimin Kim", "Jason Jian Lai", "Safiy Ahmad Malik", "Darren Tan", "Jinlin Yang"],
    "Contributions" : ["Implementation of ML Model, ML Model Writeup", "Data Preprocessing Writeup", "Data Preprocessing",
                        "Streamlit Site, ML Model Implementation", "Data Visualization, Data Visualization Writeup"
    ]
}


# Display the table
st.table(chart)

st.header("Github Repository")
st.markdown("Link: [https://github.gatech.edu/smalik79/ML4641.git](https://github.gatech.edu/smalik79/ML4641.git)")

st.header("Team")
st.markdown(
"""- Jimin Kim
- Jason Jian Lai
- Safiy Ahmad Malik
- Darren Tan
- Jinlin Yang
"""
)
