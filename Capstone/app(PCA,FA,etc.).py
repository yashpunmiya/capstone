import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def main():
    # Set page config
    st.set_page_config(page_title="Data Analyzer", page_icon="📊", layout="centered")
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
            .stApp { background-color: #f0f2f6; }
            h1 { color: #4CAF50; text-align: center; }
            .css-1d391kg { background-color: #ffffff; border-radius: 10px; padding: 20px; }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("📊 CSV Data Analyzer")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df  # Store dataframe in session state
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    if 'df' in st.session_state:
        df = st.session_state.df

        st.subheader("Basic Data Analysis")
        
        st.write("### Head of the dataset")
        st.dataframe(df.head())

        st.write("### Tail of the dataset")
        st.dataframe(df.tail())

        st.write("### Summary Statistics")
        st.write(df.describe())

        # Handling Missing Values
        st.subheader("Handling Missing Values")
        st.write(df.isnull().sum())

        selected_columns = st.multiselect("Select column(s) to handle outliers", df.columns)

        null_action = st.radio("How do you want to handle missing values?", ("None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode"))
        
        if null_action == "Drop Rows":
            df = df.dropna()
            st.success("Dropped rows with missing values!")
        elif null_action == "Fill with Mean":
            df = df.fillna(df.mean(numeric_only=True))
            st.success("Filled missing values with column mean!")
        elif null_action == "Fill with Median":
            df = df.fillna(df.median(numeric_only=True))
            st.success("Filled missing values with column median!")
        elif null_action == "Fill with Mode":
            df = df.fillna(df.mode().iloc[0])
            st.success("Filled missing values with column mode!")

        # Outlier Detection & Handling
        st.subheader("Outlier Detection & Handling")
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:

            # Compute Outliers using IQR Method
            outlier_counts = {}
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_counts[col] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()

            # Display as Table
            outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Column", "Outlier Count"])
            st.write("### Number of Outliers per Column (IQR Method):")
            st.dataframe(outlier_df, hide_index=True)

            # User Selects Column(s) for Outlier Handling
            selected_columns = st.multiselect("Select column(s) to handle outliers", numeric_cols)

            if selected_columns:
                outlier_action = st.radio("How do you want to handle outliers?", 
                                        ("None", "Remove Outliers", "Replace with Mean", "Replace with Median"))

                for col in selected_columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    if outlier_action == "Remove Outliers":
                        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]
                        st.success(f"Removed outliers from {col}!")
                    
                    elif outlier_action == "Replace with Mean":
                        mean_value = df[col].mean()
                        df[col] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), mean_value, df[col])
                        st.success(f"Replaced outliers in {col} with mean value!")
                    
                    elif outlier_action == "Replace with Median":
                        median_value = df[col].median()
                        df[col] = np.where((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)), median_value, df[col])
                        st.success(f"Replaced outliers in {col} with median value!")
            
        # Boxplot for Outlier Visualization
        if selected_columns:
            st.write("### Boxplot for Selected Columns")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df[selected_columns], ax=ax)
            st.pyplot(fig)


        # Data Visualization
        st.subheader("Data Visualization")

        # Select multiple columns
        vis_cols = st.multiselect("Select columns to visualize", df.columns)

        # Select graph type
        graph_type = st.selectbox("Choose a graph type", ["Histogram", "Boxplot", "Scatterplot", "Bar Chart", "Line Chart"])

        if vis_cols:
            fig, ax = plt.subplots(figsize=(8, 4))

            if graph_type == "Histogram":
                for col in vis_cols:
                    if df[col].dtype in [np.int64, np.float64]:
                        sns.histplot(df[col], kde=True, ax=ax, label=col)
                ax.legend()

            elif graph_type == "Boxplot":
                sns.boxplot(data=df[vis_cols], ax=ax)

            elif graph_type == "Scatterplot":
                if len(vis_cols) >= 2:
                    sns.scatterplot(x=df[vis_cols[0]], y=df[vis_cols[1]], ax=ax)
                else:
                    st.warning("Select at least two columns for scatterplot!")

            elif graph_type == "Bar Chart":
                for col in vis_cols:
                    if df[col].dtype == object:
                        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
                ax.set_ylabel("Categories")
            
            elif graph_type == "Line Chart":
                if len(vis_cols) >= 2:
                    for col in vis_cols[1:]:
                        sns.lineplot(x=df[vis_cols[0]], y=df[col], ax=ax, label=col)
                    ax.legend()
                else:
                    st.warning("Select at least two columns for line chart!")

            st.pyplot(fig)
        else:
            st.warning("Select at least one column for visualization.")


        # Regression Analysis
        st.subheader("Regression Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 1:
            target = st.selectbox("Select the dependent (target) variable", numeric_cols)
            features = st.multiselect("Select independent (predictor) variables", [col for col in numeric_cols if col != target])

            if features:
                X = df[features]
                y = df[target]

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                st.write("### Regression Model Summary")
                st.write(f"R² Score: {model.score(X, y):.4f}")

                # Residual Plot
                fig, ax = plt.subplots()
                sns.scatterplot(x=y, y=y - y_pred, ax=ax)
                ax.axhline(y=0, color="r", linestyle="--")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)

        # Cluster Analysis
        st.subheader("Cluster Analysis (K-Means)")
        cluster_cols = st.multiselect("Select numeric columns for clustering", numeric_cols)
        if cluster_cols:
            num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df["Cluster"] = kmeans.fit_predict(df[cluster_cols])

            st.write("### Cluster Visualization")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[cluster_cols[0]], y=df[cluster_cols[1]], hue=df["Cluster"], palette="viridis", ax=ax)
            ax.set_xlabel(cluster_cols[0])
            ax.set_ylabel(cluster_cols[1])
            st.pyplot(fig)

        # Time Series Analysis
        st.subheader("Time Series Analysis")
        time_col = st.selectbox("Select the time column", df.columns)
        value_col = st.selectbox("Select the value column for time series analysis", numeric_cols)
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df = df.sort_values(by=time_col)

        st.line_chart(df.set_index(time_col)[value_col])

        # Moving Average Forecasting
        window = st.slider("Select moving average window size", min_value=2, max_value=30, value=5)
        df["Moving_Avg"] = df[value_col].rolling(window=window).mean()
        st.line_chart(df.set_index(time_col)[["Moving_Avg", value_col]])

        # PCA (Principal Component Analysis)
        st.subheader("Principal Component Analysis (PCA)")
        pca_cols = st.multiselect("Select numeric columns for PCA", numeric_cols)
        if pca_cols:
            num_components = st.slider("Number of PCA components", 1, len(pca_cols), min(2, len(pca_cols)))
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[pca_cols])

            pca = PCA(n_components=num_components)
            pca_result = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(num_components)])
            st.write("### PCA Results")
            st.dataframe(pca_df.head())

            fig, ax = plt.subplots()
            sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"])
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

            st.write("### Explained Variance Ratio")
            st.bar_chart(pca.explained_variance_ratio_)

        # Factor Analysis
        st.subheader("Factor Analysis")
        factor_cols = st.multiselect("Select numeric columns for Factor Analysis", numeric_cols)
        if factor_cols:
            num_factors = st.slider("Number of Factors", 1, len(factor_cols), min(2, len(factor_cols)))
            factor_analysis = FactorAnalysis(n_components=num_factors)
            factors = factor_analysis.fit_transform(df[factor_cols])
            factor_df = pd.DataFrame(factors, columns=[f"Factor{i+1}" for i in range(num_factors)])
            st.write("### Factor Analysis Results")
            st.dataframe(factor_df.head())

if __name__ == "__main__":
    main()
