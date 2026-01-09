from flask import Flask, render_template, send_from_directory, request
import pandas as pd
import io
import os
import boto3

app = Flask(__name__)
s3 = boto3.client("s3", region_name="eu-north-1")
BUCKET_NAME = "trendverse-forecasting"

df = None

def load_dataframe():
    global df
    if df is not None:
        return df
    
    try:
        # Try to load from S3
        obj = s3.get_object(Bucket=BUCKET_NAME, Key="trending_schedule_2026.csv")
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        # Fallback to local file
        print(f"Warning: Could not load from S3 ({e}). Loading from local file.")
        local_path = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts", "trending_schedule_2026.csv")
        if os.path.exists(local_path):
            df = pd.read_csv(local_path)
        else:
            raise FileNotFoundError(f"Could not find trending_schedule_2026.csv in S3 or local path: {local_path}")
    return df


content_categories = {
    "Beauty": "content_category_Beauty",
    "Comedy": "content_category_Comedy",
    "Fashion": "content_category_Fashion",
    "Fitness": "content_category_Fitness",
    "Food": "content_category_Food",
    "Lifestyle": "content_category_Lifestyle",
    "Music": "content_category_Music",
    "Photography": "content_category_Photography",
    "Technology": "content_category_Technology",
    "Travel": "content_category_Travel"
}

content_types = {
    "Carousel": "media_type_Carousel",
    "Photo": "media_type_Photo",
    "Reel": "media_type_Reel",
    "Video": "media_type_Video"
}


@app.route("/")
def home():
    plots = [
        {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
        {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
    ]
    return render_template("home.html", plots=plots, results=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_interest = request.form.get("interest", "").strip()
        media_choice = request.form.get("media_type", "").strip()
        top_n = int(request.form.get("top_n", 3))

        if not user_interest or not media_choice:
            plots = [
                {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
                {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
            ]
            return render_template("home.html", plots=plots, results=[{"error": "Please select both category and media type"}])

        category_key = content_categories.get(user_interest)
        media_key = content_types.get(media_choice)

        if not category_key or not media_key:
            plots = [
                {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
                {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
            ]
            return render_template("home.html", plots=plots, results=[{"error": "Invalid category or media type"}])

        data = load_dataframe()
        
        # Filter by content_category and content_type (matching your CSV columns)
        filtered = data[(data["content_category"] == category_key) & (data["content_type"] == media_key)]

        if filtered.empty:
            plots = [
                {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
                {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
            ]
            return render_template("home.html", plots=plots, results=[{"error": f"No trends found for {user_interest} + {media_choice}. Try a different combination!"}])

        # Sort by yhat (trend score) and get top results
        top_results = filtered.sort_values("yhat", ascending=False).head(top_n)
        
        # Format results for display
        results_list = []
        for _, row in top_results.iterrows():
            results_list.append({
                "Trend_Name": row["label"],
                "Trend_Score": round(row["yhat"], 2),
                "Date": row["ds"]
            })

        plots = [
            {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
            {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
        ]
        return render_template("home.html", plots=plots, results=results_list)
    
    except Exception as e:
        plots = [
            {"name": "Previous Forecast", "filename": "plot_html_prev.html"},
            {"name": "2026 Forecast", "filename": "plot_html_2026.html"}
        ]
        return render_template("home.html", plots=plots, results=[{"error": f"An error occurred: {str(e)}"}])

@app.route("/plot/<filename>")
def show_plot(filename):
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=filename)
        plot_html = obj["Body"].read().decode("utf-8")
        return render_template("index.html", plot_html=plot_html, name=filename)
    except Exception as e:
        print(f"Warning: Could not fetch {filename} from S3 ({e}). Trying local file.")
        local_path = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts", filename)
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                plot_html = f.read()
            return render_template("index.html", plot_html=plot_html, name=filename)
        else:
            return f"Error: Could not find {filename} in S3 or local path", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
