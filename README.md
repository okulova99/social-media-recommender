# 🚀 Social Media Post Recommendation System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.88.0-green)](https://fastapi.tiangolo.com/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)](https://catboost.ai/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

**Production-grade recommendation system** for personalized post suggestions in a student social network. This solution analyzes user behavior, content features, and temporal patterns to deliver relevant content recommendations in real-time.

## 📊 Business Overview

### Problem Statement
KarpovCourses social network needs to improve user engagement by replacing random posts in users' feeds with **personalized recommendations**. The challenge is to leverage user profiles, post content, and interaction history to predict which posts a user is most likely to engage with.

### Key Objectives
- ✅ Predict top-5 posts a user will like at any given time
- ✅ Achieve high hitrate@5 metric
- ✅ Handle production load (response < 0.5s per request)
- ✅ Efficient memory usage (< 4GB RAM)


## 📈 Evaluation Metric: HitRate@5

The core performance metric is defined as:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;\text{HitRate@5}=\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\mathbb{1}\left(\bigcup_{j=1}^{5}a_j^{(u)}=1\right)" />
</p>

Where:
- ![\mathcal{U}](https://latex.codecogs.com/svg.latex?\mathcal{U}) = Test user set
- ![a_j^{(u)}](https://latex.codecogs.com/svg.latex?a_j^{(u)}\in\{0,1\}) = Like indicator for j-th recommendation
- ![\mathbb{1}(\cdot)](https://latex.codecogs.com/svg.latex?\mathbb{1}(\cdot)) = Indicator function

**Optimization Challenge**:
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\max_{\text{model}}\text{HitRate@5}\quad\text{subject to}\quad\begin{cases}\mathbb{E}[\text{latency}]<0.5s\\\text{memory}<4\text{GB}\end{cases}" />
</p>