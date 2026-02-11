SkyGeni Sales Data Analysis & Predictive Modeling
This project provides a comprehensive end-to-end data science pipeline to analyze sales performance and predict deal outcomes (Won/Lost) using a Random Forest Classifier. It is designed to provide actionable insights for a Chief Revenue Officer (CRO) to optimize sales cycles and regional resource allocation.

📋 Table of Contents
Installation

Project Structure

Exploratory Data Analysis

Feature Engineering

Model Performance

Key Insights

🛠 Installation
Clone the repository:

Bash
git clone https://github.com/Parimal108/skygeni-sales-analysis.git
Navigate to the directory:

Bash
cd skygeni-sales-analysis
Install the required dependencies:

Bash
pip install -r requirements.txt
📂 Project Structure
main_script.py: The primary Python script for data processing, EDA, and modeling.

skygeni_sales_data.csv: The raw sales dataset used for analysis.

Executive_Summary.pdf: A business-focused report detailing findings and recommendations.

requirements.txt: List of Python libraries needed for reproducibility.

📊 Exploratory Data Analysis
The analysis includes:

Win/Loss Distribution: Analyzing success rates across different lead sources.

Sales Cycle Analysis: Measuring the average duration of Won vs. Lost deals.

Revenue Density: Identifying top-performing regions by total deal value.

Time-Series Trends: Tracking monthly revenue growth to identify seasonality.

⚙️ Feature Engineering
To improve model accuracy, we developed custom metrics:

Deal Momentum: Calculated as Deal Amount / Sales Cycle Days to identify high-velocity opportunities.

Rep Historical Win Rate: A metric capturing the efficiency of individual sales representatives.

🤖 Model Performance
The Random Forest Classifier was trained using a stratified 80/20 split with class weight balancing to handle the inherent imbalance in sales outcomes.

Primary Metric: The model focuses on Precision and Recall for the "Won" class to ensure high-confidence predictions.

Interpretability: Feature importance analysis reveals the top drivers of deal success, allowing for data-driven strategy adjustments.
