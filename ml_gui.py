import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
import pandas as pd

# Load trained KNN model
model = joblib.load("knearest_neighbors_k=3,batch_learning.pkl")  # Your trained model file

def predict_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, header=None)
        if df.shape[1] != 60:
            raise ValueError("CSV must contain exactly 60 features (columns).")
        prediction = model.predict(df)[0]
        return prediction
    except Exception as e:
        return f"Error: {str(e)}"

def browse_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select CSV file",
        filetypes=[("CSV Files", "*.csv")])
    if file_path:
        result = predict_from_csv(file_path)
        messagebox.showinfo("Prediction Result", f"Prediction for {file_path}:\n{result}")

# GUI setup
root = tk.Tk()
root.title("Sonar Rock vs Mine Prediction")

# Button to select CSV and predict
tk.Button(root, text="Select CSV and Predict", command=browse_and_predict).pack(pady=10)

# Custom input section
tk.Label(root, text="\nOr enter 60 comma-separated values:").pack()
entry = tk.Entry(root, width=120)
entry.pack()

def predict_custom():
    try:
        values = list(map(float, entry.get().split(',')))
        if len(values) != 60:
            raise ValueError("Must enter exactly 60 comma-separated numeric values.")
        prediction = model.predict([values])[0]
        messagebox.showinfo("Prediction", f"The object is: {prediction}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Predict Custom Input", command=predict_custom).pack(pady=10)

root.mainloop()
