import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. LOAD DATA
# =====================================================================
df = pd.read_csv("data/carrier.csv")
df = df[~df["Airline"].str.contains("Total", case=False, na=False)]
df_clean = df.dropna(subset=[
    "Passenger Number", "Aircraft Kilometres",
    "Seat Kilometers", "Passenger Load Factor", "Airline"
])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape before cleaning: {df.shape}")
print(f"Shape after cleaning: {df_clean.shape}")
print(f"\nColumns: {df_clean.columns.tolist()}")

# =====================================================================
# 2. SUMMARY STATISTICS
# =====================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS (numeric columns)")
print("=" * 60)
print(df_clean[["Passenger Number", "Aircraft Kilometres",
                 "Seat Kilometers", "Passenger Load Factor"]].describe())

# =====================================================================
# 3. MISSING VALUES CHECK
# =====================================================================
print("\n" + "=" * 60)
print("MISSING VALUES (original data)")
print("=" * 60)
print(df.isnull().sum())

# =====================================================================
# 4. AIRLINE FREQUENCY
# =====================================================================
print("\n" + "=" * 60)
print("RECORDS PER AIRLINE")
print("=" * 60)
print(df_clean["Airline"].value_counts())

# =====================================================================
# 5. VISUALIZATIONS (saved as image files)
# =====================================================================

# 5a. Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["Passenger Number"], bins=40, kde=True, color="#378ADD")
plt.title("Distribution of Passenger Number")
plt.xlabel("Passenger Number")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_target_distribution.png")
plt.close()

# 5b. Correlation heatmap
plt.figure(figsize=(7, 5))
corr = df_clean[["Passenger Number", "Aircraft Kilometres",
                  "Seat Kilometers", "Passenger Load Factor"]].corr()
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.close()

# 5c. Passengers by airline (top 10, boxplot)
top_airlines = df_clean["Airline"].value_counts().head(10).index
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_clean[df_clean["Airline"].isin(top_airlines)],
    x="Airline", y="Passenger Number"
)
plt.xticks(rotation=45, ha="right")
plt.title("Passenger Number by Airline (Top 10 by Record Count)")
plt.tight_layout()
plt.savefig("eda_passengers_by_airline.png")
plt.close()

# 5d. Load factor vs passengers (scatter)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x="Passenger Load Factor", y="Passenger Number",
                 alpha=0.4, color="#0F6E56")
plt.title("Passenger Load Factor vs Passenger Number")
plt.tight_layout()
plt.savefig("eda_loadfactor_vs_passengers.png")
plt.close()

print("\nEDA complete. 4 chart images saved in your project folder.")