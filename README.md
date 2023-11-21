<h1 align="center">
             Spam Analysis Web App 💬 📝 ✍️
</h1>

![image](https://github.com/goyal-rohit/sms-spam-classifer/blob/64e5f4efc6e59b088d90f9486357383c48c69a70/text-analysis.jpg)

This app is used to perform spam analysis of a text/sms

## Tech Stacks Used
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

## Libraries Used

<img src="https://img.shields.io/badge/numpy%20-%2314354C.svg?&style=for-the-badge&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/pandas%20-%2314354C.svg?&style=for-the-badge&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/streamlit%20-%2314354C.svg?&style=for-the-badge&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/nltk%20-%2314354C.svg?&style=for-the-badge&logo=nltk&logoColor=white"/> <img src="https://img.shields.io/badge/scikitlearn%20-%2314354C.svg?&style=for-the-badge&logo=scikitlearn&logoColor=white"/>

## Structure Of The Project

- The prediction page is connected with a Machine Learning Model that uses Multinomial Naive Bayes Algorithms to predict the results.
- We have only 1 relevant feature taken into consideration which is the text and then the text is preprocessed and vectorized with the help of TF-IDF Vectorizer to fit into the model and train it.

The text is preprocessed and then fed to the model.

## Deployment Of The Project

After the modeling part, the model is deployed using the Streamlit library on Streamlit Share so that the app is available for usage for everyone.

## Link To My Web Application -

https://sms-spam-classifer-lxeod8wph4d58vr9whaf2p.streamlit.app/

