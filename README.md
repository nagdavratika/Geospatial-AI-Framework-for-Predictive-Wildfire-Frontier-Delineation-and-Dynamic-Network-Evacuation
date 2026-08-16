# Spatiotemporal Wildfire Spread Prediction & Dynamic Evacuation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Geoinformatics & Spatial AI](https://img.shields.io/badge/Domain-Geoinformatics%20%7C%20Spatial%20AI-green.svg)](#)
[![Stack: XGBoost | GeoPandas | NetworkX | DBSCAN](https://img.shields.io/badge/Stack-XGBoost%20%7C%20GeoPandas%20%7C%20NetworkX%20%7C%20DBSCAN-orange.svg)](#)
[![Data: NASA FIRMS | OSM | DEM Terrain](https://img.shields.io/badge/Data-NASA%20FIRMS%20%7C%20DEM%20Terrain-blueviolet.svg)](#)

An enterprise-grade Spatiotemporal Geospatial AI/ML platform that integrates supervised machine learning fire propagation forecasting with dynamic road network pathfinding. The engine trains an **XGBoost Classifier** on top of terrain topography (slope, aspect), fuel moisture, NDVI vegetation density, and wind vectors to forecast fire spread probabilities at $t + 1\text{ hr}$, dynamically restructuring **OpenStreetMap/NetworkX** transit graph impedances to route emergency evacuation vehicles safely around advancing fire fronts.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Problem Statement & Background](#problem-statement--background)
- [System Architecture](#system-architecture)
- [Mathematical & Algorithmic Methodology](#mathematical--algorithmic-methodology)
- [Feature Store Data Dictionary](#feature-store-data-dictionary)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Execution & Usage](#execution--usage)
- [Benchmark Results & Performance](#benchmark-results--performance)
- [License](#license)

---

## Project Overview

Conventional disaster routing algorithms are strictly reactive: they only account for currently burning hotspots. During fast-moving wildfire incidents, evacuees dispatched along currently clear corridors frequently encounter active fire fronts because the fire spreads faster than vehicles can transit.

This project solves this bottleneck with an **AI/ML-Driven Predictive Framework**:
1. **Machine Learning Wildfire Spread Forecaster:** Trains a gradient-boosted classifier (**XGBoost**) on multi-source terrain, meteorological, and thermal telemetry to predict the spatial probability of fire spread for future time horizons ($t + 1\text{ hr}$).
2. **DBSCAN Plume Clustering & Spatial Buffering:** Clusters high-risk propagation cells ($\hat{p} \ge 0.60$) and constructs a metric buffer safety zone ($500\text{ m}$) via `GeoPandas` and `Shapely`.
3. **Predictive Dynamic Dijkstra Pathfinding:** Re-weights network edge impedances based on predicted hazard intersections, ensuring routes evade both current fires and impending fire fronts.

---

## Key Features

- **Environmental Feature Engineering:** Models non-linear interactions between slope degree, wind velocity vectors, fuel moisture deficiency, and vegetation canopy ($NDVI$).
- **Supervised Spread Classification:** Employs `XGBoost` with optimized hyper-parameters, achieving **$>0.92$ ROC-AUC** on fire propagation prediction.
- **Density-Based Spatial Clustering:** Implements DBSCAN with a geodesic Haversine distance metric to delineate discrete fire fronts.
- **Predictive Dynamic Routing:** Custom topological graph solver that avoids predicted burn perimeters.
- **Production-Ready OOP Design:** Modular architecture suitable for integration into emergency dispatch dashboards.

---

## Problem Statement & Background

Emergency management during extreme wildfire events encounters two primary failures:

1. **Reactive Navigation Blindspots:** Traditional GIS Dijkstra/A* routing calculates the shortest physical distance at time $t_0$. If an active flank spreads at $5\text{ km/h}$, a route calculated as "clear" will lead directly into active flames by $t_0 + 30\text{ min}$.
2. **Coupled Non-Linear Environmental Dynamics:** Fire spread is not isotropic. It accelerates exponentially up steep slopes and along wind azimuths while decelerating across high fuel moisture barriers. Supervised gradient boosting captures these multivariate non-linearities.

---

## System Architecture

```text
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │  NASA FIRMS VIIRS Active Fires  │        │  DEM Terrain & NOAA Wind Vectors │
  │  (Thermal Radiative Power - FRP)│        │   (Slope, Aspect, Wind Speed)    │
  └────────────────┬────────────────┘        └─────────────────┬────────────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │    Environmental Feature Store Assembly   │
                   │    [Slope | Wind_Vec | Fuel_Moist | NDVI] │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Supervised XGBoost Fire Spread Predictor │
                   │       P(Spread_t+1 | Environmental_X)     │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   DBSCAN Clustering on High-Risk Fronts   │
                   │   (Haversine Geodesic Density Grouping)   │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   Spatiotemporal Hazard Buffer Envelope   │
                   │         (EPSG:3857 Metric Buffering)      │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │  Dynamic Network Graph Edge Re-Weighting  │
                   │    Cost_dyn = Distance * ML_Risk_Penalty  │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   AI Hazard-Aware Dijkstra Pathfinding    │
                   │       (Safe Evacuation Corridor)          │
                   └───────────────────────────────────────────┘
