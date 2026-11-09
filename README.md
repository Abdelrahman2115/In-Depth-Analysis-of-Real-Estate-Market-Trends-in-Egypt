# In-Depth Analysis of Real Estate Market Trends in Egypt

A data science project analyzing Egypt's real estate market by merging multiple datasets, performing data cleaning and EDA, and building machine learning models for property price prediction. It also includes a rule-based property recommendation system.

## Data sources

All three datasets are public on Kaggle. Download them and put the CSV files in the project root.

| File | Dataset | Rows | Origin |
|---|---|---|---|
| `Egypt_Houses_Price.csv` | [Egypt_Houses_price](https://www.kaggle.com/datasets/mohammedaltet/egypt-houses-price) (mohammedaltet) | 27,361 | OLX Egypt listings |
| `properties.csv` | [Egypt Cairo Housing Prices](https://www.kaggle.com/datasets/iyadelwy/egypt-housing-prices) (iyadelwy) | 11,418 | Cairo property listings |
| `real_estate_data_bayut_full.csv` | [Real Estate Bayot Egypt](https://www.kaggle.com/datasets/ameribrahim/real-estate-bayot-egypt) (ameribrahim) | 50,000 | Bayut Egypt listings |

With the Kaggle CLI:

```bash
kaggle datasets download -d mohammedaltet/egypt-houses-price --unzip
kaggle datasets download -d iyadelwy/egypt-housing-prices --unzip
kaggle datasets download -d ameribrahim/real-estate-bayot-egypt --unzip
```

## How to run

1. Run the cleaning notebooks (each one writes a `*_cleaned_data.csv`):
   - `Egypt_Houses_Price_Cleaning.ipynb`
   - `Egypt_Cairo_Housing_Prices_Cleaning.ipynb`
   - `real_estate_data_bayut_full_Cleaning.ipynb`
2. Run `FinalCode.ipynb` to merge the data, run the EDA, train the price prediction models, and use the recommendation system.

Requirements: Python 3, Jupyter, pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, mlxtend, and tensorflow.
