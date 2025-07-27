# 🏠 House Price Prediction Web App using Django & Machine Learning

This project predicts house prices based on various input features like area income, house age, number of rooms, bedrooms, population, and address. It uses a trained **Linear Regression** model and presents the result through an interactive and modern Django web interface, including dynamic bar charts using **Chart.js**.

---

## 📦 Dataset

**Kaggle Dataset URL:**
[Housing Dataset on Kaggle](https://www.kaggle.com/datasets/huyngohoang/housingcsv/data)

This dataset contains housing data used to train the model to predict prices.

---

## 🔽 Download Dataset using `opendatasets`

To directly download the dataset from Kaggle into your Google Colab or local machine:

### Step 1: Install libraries

```bash
pip install opendatasets numpy pandas joblib scikit-learn
```

### Step 1: Install libraries

```Download Dataset
import opendatasets as od
od.download("https://www.kaggle.com/datasets/huyngohoang/housingcsv/data")
```
