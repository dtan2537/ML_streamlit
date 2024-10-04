import streamlit as st

st.title("Modelling Student Stress")

st.header("Research Proposal")

st.subheader("Introduction")

st.subheader("Problem Definition")

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
st.markdown("#### Goals")
st.markdown("#### Expected Results")

st.subheader("References")

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
