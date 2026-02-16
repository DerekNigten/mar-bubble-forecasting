## Getting Started

### Prerequisites
- R (≥ 4.0) with RStudio
- Python (≥ 3.10)
- Git

### Installation

**1. Clone the repository**
git clone https://github.com/DerekNigten/mar-bubble-forecasting.git
cd mar-bubble-forecasting

**2. Install R dependencies**
Open RStudio and run:
install.packages("devtools")
devtools::install_github("cran/MARX")
install.packages("mFilter")

**3. Install Python dependencies**
pip install -r requirements.txt

### Data
Download monthly Nickel prices (January 1980 – September 2019) from the
World Bank Global Economic Monitor commodity price database and place the
file at:
data/raw/nickel_prices_1980_2019.csv

The file should have two columns: Date (format: 1980M01) and
Nickel_Price_USD_per_MT.

### Running the Project

**Step 1 — R estimation (run once)**
Open r/marx_estimation.R in RStudio and run the script.
This produces two files in data/processed/:
- mar_parameters.csv   (estimated MAR(1,1) parameters)
- nickel_filtered.csv  (HP-filtered series and noncausal component)

**Step 2 — Verify Python setup**
python src/preprocessing.py

Expected output:
Parameters : MAR(1,1) | φ=0.617, ψ=0.777, df=1.4947, σ=402.94
Series     : 476 obs [1980-02 → 2019-09]

**Step 3 — Run notebooks in order**
Launch Jupyter and run notebooks in sequence:
jupyter notebook
01_data_exploration.ipynb
02_estimation_results.ipynb
03_forecasting_methods.ipynb
04_empirical_application.ipynb

### Project Structure
(your existing structure here)

### Note on Data Sources
The original paper (Hecq & Voisin, 2021) uses IMF Primary Commodity Prices
data. This replication uses World Bank Global Economic Monitor data for the
same series and period. Minor differences in estimated parameters are
attributed to this source difference.
