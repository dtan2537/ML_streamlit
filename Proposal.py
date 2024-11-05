import streamlit as st

st.title("Modelling Student Stress")

st.header("Research Proposal")

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
st.image("visuals/stress_dataset.png")
st.write("A first look at the data shows how there are varying magnitudes. We can further see that from the distribution of the features:")
st.image("visuals/dataset_distribution.png")
st.markdown("""
Thus, we would first need to **standardize** the data. This would ensure that features of higher magnitudes do not dominate features of lower magnitudes. 
For example, depression has a much higher magnitude compared to bullying. Hence, we use **sklearn.preprocessing.StandardScaler** to scale the features.

Another thing we need to do is perform One-Hot Encoding on the target variable. 
Let’s look at stress. It takes the discrete values 0, 1 and 2. 
To make the model more probabilistic and prevent ordinality issues, we decide to use One-Hot Encoding. 
For that we have **sklearn.preprocessing.OneHotEncoder**.
""")
st.image("visuals/feature_correlation_matrix.png")
st.markdown("""
Looking at the feature correlation matrix, we can see that some features such as anxiety level and future career concerns are highly correlated,
 and may be redundant. We can use dimensionality reduction, especially Principal Component Analysis (PCA) to combat this.

""")

st.markdown("#### Models")
st.markdown("""
Since our dataset contains labeled data, we will explore the following supervised models:
Support Vector Machines (SVM), Neural Networks, and Gradient Boosting Machines (GBM).
            
SVM is a powerful regression and classification model that works by finding an optimal
hyperplane separating different output values. They are known for managing complex decision
boundaries in high dimensions, which is applicable given the diverse feature sets in our data.
            
Neural Networks are a family of machine learning models characterized by a network of
connected neuron layers that can effectively capture complex, non-linear relationships between
features. CNNs in particular are effective with tabularized data that our dataset is formatted in,
where local patterns between feature groups can be captured through convolutional filters.
            
Finally, GBM is a learning technique that sequentially builds a series of self-correcting
decision trees. This model is also highly effective for capturing non-linear relationships and
provides high predictive accuracy that is necessary for our application.
""")

st.subheader("Results and Discussion")
st.markdown("#### Metrics")
st.markdown("""
Accuracy: Measures the proportion of correct predictions out of all predictions, providing an overall sense of model performance.
            
Precision: The proportion of true positives among all positive predictions, crucial when false positives are costly.

Recall (Sensitivity): The proportion of actual high-stress students correctly identified by the model, reducing false negatives.

F1 Score: Balances precision and recall for a comprehensive view of the model’s accuracy.
""")
st.markdown("#### Goals")
st.markdown("""
Accuracy, Precision, Recall, F1 Score: Aim for at least 80% to ensure high performance across all metrics.
""")

st.markdown("#### Expected Results")
st.markdown("""
Strong factor correlations, high model performance, early stress identification, and positive impact on student well-being.
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
    "Contributions" : ["Intro/ Literature Review, Problem Motivation, References, Video", "Models", "Idea/ Dataset Discovery, Data Preprocessing",
                        "Github Pages, Streamlit Site, Gantt Chart, Contribution Chart", "Metrics, Goals, Expected Results"
    ]
}


# Display the table
st.table(chart)

st.header("Video Presentation")
st.markdown("Link: [https://youtu.be/XdXegoL8W64](https://youtu.be/XdXegoL8W64)")

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
