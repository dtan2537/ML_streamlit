import streamlit as st

st.title("ML Project Title")

st.header("Research Proposal")
st.subheader("Introduction")
st.subheader("Problem Definition")
st.subheader("Methods")
st.subheader("Results and Discussion")
st.subheader("References")

st.header("Charts")
st.subheader("Gantt Chart")
st.markdown("Link: [Gantt Chart](https://docs.google.com/spreadsheets/d/1scnAtyIJyvz2WJbtioxEO4HTIOQOQ7Kt/edit?usp=sharing&ouid=101934921339602004791&rtpof=true&sd=true)")
st.subheader("Contribution Chart")

chart = {
    "Name" : ["Jimin Kim", "Jason Jian Lai", "Safiy Ahmad Malik", "Darren Tan", "Jinlin Yang"],
    "Contributions" : ["Intro/ Literature Review, Problem Motivation, References, Video", "Methods", "Idea/ Dataset Discovery, Methods",
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
