# Berth Allocation Plan With ML

Berth Allocation Plan With ML is a full-stack system designed to assist port authorities in dynamically allocating berths to vessels based on real-time data and machine learning predictions. The model learns from human decision-making and historical port activity to suggest optimal berth schedules, improving efficiency and reducing delays.

## Link to documentation: https://docs.google.com/document/d/1vb3uau422N3TEv_cRgWFavyMdnqGPm1yvpjquigVAmo/edit?usp=sharing

## Link to slides: https://docs.google.com/presentation/d/1zzWxjqRZm5vUCjnd470LImS5rYhCPTde7E4QuwMJTF8/edit?usp=sharing

# Running the App

1. Clone the repo:
```bash
git clone https://github.com/aleksandrazec/Digital-Twin-of-berth-planning-process.git
cd Digital-Twin-of-berth-planning-process
```

# Libraries that we used:

## PyTorch

PyTorch is an open source machine learning (ML) framework based on the Python programming language and the Torch library

### Installation

Use the package manager https://pytorch.org/get-started/locally/ to install PyTorch library.

```bash
pip install torch torchvision torchaudio
```

## Pandas

Pandas is an open-source data analysis and manipulation library for Python

### Installation

```bash
pip install pandas
```

## NumPy

NumPy is an open source library for numerical computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays

### Installation

```bash
pip install numpy
```

## BeautifulSoup

BeautifulSoup is a Python library used for parsing HTML and XML documents. It is commonly used for web scraping and extracting data from web pages

### Installation

```bash
pip install beautifulsoup4
```

## scikit-learn

scikit-learn is an open source machine learning library for the Python programming language. It provides simple and efficient tools for data mining and data analysis, built on top of NumPy, SciPy, and matplotlib

### Installation

```bash
pip install scikit-learn
```

## How It Works

- Scrape XML files from Hong Kong Marine Department using BeautifulSoup.
- Convert them into structured tabular data using pandas.
- Normalize features and prepare training/validation sets.
- Train the ML model to mimic expert decisions.
- Deploy model and serve predictions via a full-stack interface.

# How we collected data? 
## Our realistic data was scraped:
### Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day0/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day0/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day0/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day0/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day1/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day1/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day1/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day1/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day2/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day2/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day2/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day2/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day3/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day3/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day3/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day3/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day4/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day4/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day4/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day4/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day5/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day5/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day5/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day5/RP05505.XML
    Due To Enter Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day6/RP04005.XML
    Intend To Depart Hong Kong Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day6/RP04505.XML
    Enter Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day6/RP05005.XML
    Depart Hong Kong Water Report,https://www.mardep.gov.hk/e_files/en/pub_services/report_day6/RP05505.XML


## Authors

- Filip Trajkoski - https://github.com/FT1E
- Aleksandra Zec - https://github.com/aleksandrazec
- Simona Cholakova — https://github.com/simona-cholakova
- Nade Belovinova - https://github.com/bel-n















