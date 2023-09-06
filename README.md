# Earthquake Impact on Buildings

An algorithm that predicts the impact of the earthquake on buildings.

## Table of Contents

- [Installation](#a-id-installation-installation)
- [Motivation behind the Project](#motivation-behind-the-project)
- [File Description](#file-description)
- [Results](#results)
- [Licensing, Authors, Acknowledgements](#licensing-authors-and-acknowledgements)

## Installation

The code requires Python 3 and general libraries available through the Anaconda package.

## Motivation behind the Project

We all know how deadly an earthquake can be. We have to make predictions based on the data we've got 
and use it to gain better knowledge about what to change and what to improve.

To find the results we use:

- XGBClassifier

## File Description

This project includes one Jupyter Notebook with all code required for analyzing the data and creating a supervised 
machine learning algorithm. The csv file contains 308854 people along with the following features:

- geo_levels (from largest (level_1) to most specific sub-region (level_3))
- age (of the building)
- area_percentage
- height_percentage
- land_surface_condition
- foundation_type
- roof_type
- ground_floor_type
- other_floor_type
- position
- plan_configuration
- superstructure_types
- legal_ownership_status
- count_families
- has_secondary_use (y/n)

## Results

We get:

- **94.89%** accuracy by using SMOTE and 10K-Fold