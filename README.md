# Football Player Similarity and Valuation Model

This repository contains code and models developed for the **Modelling Player Similarity and Valuation for Enhanced Scouting and Recruitment in Football** project. The goal of this project is to streamline football player identification and valuation through data science techniques.

## Project Summary

This project integrates **K-Means Clustering** and **Random Forest Regression** to provide clubs with tools that help identify similar players to a given player and estimate the market value of those players. The project is especially beneficial for clubs operating under financial constraints such as those imposed by **UEFA Financial Fair Play** regulations or clubs with low budgets.

## Objectives

- **Player Similarity:** Group football players based on their performance metrics using K-Means clustering.
- **Player Valuation:** Estimate the market value of players using a Random Forest Regression model trained on player data.

## Data Sources

- **Football Manager (FM):** For player statistics and performance metrics.
- **Transfermarkt:** For player market values and financial data.

## Models

- **K-Means Clustering Model**: This model groups players based on their positional attributes and key performance indicators like passing, shooting, and defending and more.
- **Random Forest Regression Model**: Estimates the player market value based on factors like age, contract, and release clauses.

## Files

### Python Scripts
- **K Means Model for Player Similarity.py**: Contains the code for clustering players based on performance metrics.
- **Random Forest Regressor For Valuation.py**: Contains the code for predicting player market value using Random Forest Regression.

### Report
- **S267562.docx**: A detailed report on the methodologies, models, and results used in this project.

## Prerequisites

To run the models, you’ll need the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## How to Use

1. **Player Similarity Model**: Run `K Means Model for Player Similarity.py` to cluster football players based on their performance attributes.
   - Input: Player statistics dataset (e.g., from FM).
   - Output: Clusters of similar players.
   
2. **Player Valuation Model**: Run `Random Forest Regressor For Valuation.py` to estimate player market values.
   - Input: Player data (age, contract, wage, etc.).
   - Output: Predicted market value for each player.

## Model Performance

- **K-Means Clustering**: Achieved a reasonable balance between clustering quality and practical relevance, with a **Silhouette Score** of 0.14.
- **Random Forest Regressor**: High predictive accuracy with an **R² score** of approximately 0.93.

## Future Improvements

- Add AI-generated scouting reports.
- Develop a "Team Builder" feature for player recommendations.
- Integrate the model into an interactive web application.

## License

This project is licensed under the MIT License.
