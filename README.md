#  House Price Prediction (End-to-End Machine Learning System)
> **ITI AI Track — Level 1 Capstone Project**  
> An end-to-end real estate valuation system featuring data preprocessing pipelines, regression models, FastAPI backend service, and interactive frontend dashboard.

---

##  Project Overview & Architecture
This project implements a complete, production-ready machine learning system that predicts property prices based on key attributes such as location, carpet area, floor number, furnishing status, bathrooms, balconies, and transaction type.

###  Dataset Information (Kaggle)
* **Dataset Name:** [House Price Dataset (Magicbricks)](https://www.kaggle.com/datasets/juhibhojani/house-price)
* **Kaggle Source:** [https://www.kaggle.com/datasets/juhibhojani/house-price](https://www.kaggle.com/datasets/juhibhojani/house-price)
* **Dataset Author:** Juhi Bhojani
* **Raw Records:** 187,531 rows × 21 columns
* **Cleaned Records:** 173,042 rows after price/area normalization and outlier filtering
* **Key Features:** `location`, `Carpet Area` / `Super Area`, `Floor`, `Furnishing`, `Transaction`, `Bathroom`, `Balcony`, `Amount(in rupees)`

---

##  Machine Learning Models & Evaluation

The system evaluates two core regression approaches trained with a target transformation `log1p(y)` to stabilize right-skewed pricing distributions and evaluated using `expm1(y_pred)` on the test set:

###  Model Performance Comparison Table (جدول مقارنة ودقة النماذج)

| Model Name | $R^2$ Score (Test) | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | Evaluation / Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (Baseline)** | `< 0.00` (-4.61) | ₹4,245,530 (~42.45 Lacs) | ₹27,240,792 (~2.72 Cr) | ❌ Baseline (Underfitting non-linear relationships) |
| **Random Forest Regressor** | **0.9310 (93.1%)** | **₹981,881 (~9.81 Lacs)** | **₹3,020,816 (~30.20 Lacs)** | 🏆 **Best Model (Selected for Production API)** |

> ** Key Takeaway:** The **Random Forest Regressor** achieved high accuracy ($R^2 = 93.1\%$) by effectively capturing non-linear interactions between prime property locations, floor levels, and unit configurations, reducing the Mean Absolute Error to less than ₹9.8 Lacs.

---

##  End-to-End System Components

1. **Machine Learning Pipeline (`notebooks/house_price_model.ipynb` & `train_model.py`)**:
   - **Data Cleaning**: Parses messy text amounts (e.g. `42 Lac`, `1.4 Cr`), unifies multi-unit areas (`sqm` to `sqft`), extracts floor numbers, handles outliers.
   - **EDA**: 4 comprehensive visualizations analyzing price distribution, area correlation, top locations, and furnishing impact.
   - **Scikit-Learn Pipeline**: `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder(handle_unknown='ignore')`.
   - **Target Transformation**: `log1p(y)` scaling to handle skewed pricing distributions and `expm1` inverse transformation for predictions.
   - **Artifact Export**: Exported trained pipeline to `models/house_price.pkl` and supported cities to `models/locations.json`.

2. **FastAPI Backend Service (`backend/`)**:
   - High-performance asynchronous REST API.
   - **Endpoints**:
     - `GET /health`: Model status & health check.
     - `GET /locations`: List of 50+ supported locations for UI dropdowns.
     - `POST /predict`: Real-time valuation estimation.
     - `GET /docs`: Interactive Swagger documentation.
   - Full input validation using **Pydantic v2**.
   - Structured logging, CORS support, and Docker containerization.

3. **Modern Frontend Dashboard (`frontend/`)**:
   - Responsive, dark-mode Glassmorphism user interface.
   - Real-time prediction display in Lacs/Crores and exact Rupees.
   - Preset buttons (1 BHK Budget, 2 BHK Standard, 3 BHK Luxury) for 1-click live demo during project defense.

---

##  API Testing & Request Demonstration (أمر تجربة الـ API للمشرف)

Below are instructions and sample commands to test the FastAPI backend directly via command line, cURL, or Python:

### 1️⃣ Health Check (`GET /health`)
Verify that the FastAPI server is running and the trained model pipeline is loaded:

```bash
# Using cURL
curl -X GET http://127.0.0.1:8000/health
```

**Sample Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "locations_count": 51
}
```

---

### 2️⃣ Predict Property Price (`POST /predict`)

#### 🔹 Using cURL (Bash / Linux / macOS / Git Bash):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "location": "thane",
       "carpet_area_sqft": 1200.0,
       "floor_num": 3,
       "furnishing": "Semi-Furnished",
       "transaction": "Resale",
       "bathrooms": 2,
       "balconies": 1
     }'
```

#### 🔹 Using cURL (Windows PowerShell):
```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{\"location\": \"thane\", \"carpet_area_sqft\": 1200.0, \"floor_num\": 3, \"furnishing\": \"Semi-Furnished\", \"transaction\": \"Resale\", \"bathrooms\": 2, \"balconies\": 1}'
```

#### 🔹 Using Python (`requests`):
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "location": "thane",
    "carpet_area_sqft": 1200.0,
    "floor_num": 3,
    "furnishing": "Semi-Furnished",
    "transaction": "Resale",
    "bathrooms": 2,
    "balconies": 1
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())
```

####  Sample Request Payload:
```json
{
  "location": "thane",
  "carpet_area_sqft": 1200.0,
  "floor_num": 3,
  "furnishing": "Semi-Furnished",
  "transaction": "Resale",
  "bathrooms": 2,
  "balconies": 1
}
```

####  Sample Response Payload:
```json
{
  "predicted_price_rupees": 12450800.0,
  "formatted_price": "1.25 Cr",
  "currency": "INR",
  "status": "success"
}
```

---

##  Project Directory Structure
```
house-price-project/
├── data/
│   └── house_prices.csv           # Kaggle Dataset (Juhi Bhojani House Price Dataset)
├── notebooks/
│   └── house_price_model.ipynb    # Clean Jupyter Notebook with Arabic explanations
├── models/
│   ├── house_price.pkl            # Exported Scikit-Learn Pipeline (Preprocessor + RF Model)
│   └── locations.json             # Exported Top Locations List
├── backend/
│   ├── app/
│   │   ├── api/routes/prediction.py
│   │   ├── core/config.py
│   │   ├── schemas/prediction.py
│   │   ├── services/inference.py
│   │   ├── services/preprocessing.py
│   │   ├── utils/logging_config.py
│   │   └── main.py
│   ├── tests/
│   │   └── test_prediction.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train_model.py                 # Standalone script to train & export models
└── README.md
```

---

##  How to Run the Project

### 1️⃣ Run the Jupyter Notebook
```bash
# Navigate to notebooks directory
cd notebooks
jupyter notebook house_price_model.ipynb
```

### 2️⃣ Run the FastAPI Backend
```bash
# Navigate to backend directory and install dependencies
cd backend
pip install -r requirements.txt

# Start the uvicorn server
uvicorn app.main:app --reload --port 8000
```
* **Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Docs:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 3️⃣ Run the Frontend
Simply open `frontend/index.html` in any web browser (or use VS Code Live Server / Python HTTP server):
```bash
cd frontend
python -m http.server 3000
```
Then visit [http://localhost:3000](http://localhost:3000).

---

##  Project Defense Q&A Cheatsheet (أسئلة المناقشة المتوقعة)

### 1. ليه استخدمنا التحويل اللوغاريتمي `np.log1p(y)` على السعر؟
> **الإجابة:** لأن توزيع أسعار العقارات في الواقع مائل لليمين بشدة (Right-Skewed Distribution) وفيه عقارات أسعارها عالية جداً مقارنة بالمتوسط. استخدام الـ Log Transform بيقرب التوزيع للشكل الطبيعي (Normal Distribution) وبيمنع القيم الكبيرة من التأثير السلبي على دقة تدريب الموديل، وبعد التوقع بنرجع السعر لأصله باستخدام `np.expm1`.

### 2. إزاي اتعاملنا مع المناطق الكثيرة جداً (High Cardinality) في عمود الـ `location`؟
> **الإجابة:** الداتا فيها مئات المناطق المختلفة وأغلبها متكرر مرات قليلة جداً. خدنا أعلى 50 منطقة تكراراً، وأي منطقة تانية نادرة جمعناها تحت اسم `'other'`، وطبقنا عليها `OneHotEncoder(handle_unknown='ignore')` عشان لو جه أي اسم جديد في الـ API ميحصلش Error.

### 3. ليه بنستخدم `Pipeline` و `ColumnTransformer` مع بعض ونحفظهم في ملف `.pkl` واحد؟
> **الإجابة:** عشان نمنع تسريب البيانات (Data Leakage) ونضمن إن خطوات الـ Preprocessing (زي الـ Imputation والـ Scaling والـ One-Hot Encoding) اللي اتعلمناها من بيانات التدريب تتطبق بنفس الدقة والقيم بالظبط على أي بيانات جديدة تدخل للـ API في مرحلة الـ Inference.

### 4. إيه الفرق بين نتائج الموديل البسيط (Linear Regression) والموديل المتقدم (Random Forest)؟
> **الإجابة:** الـ Linear Regression بيعتمد على علاقة خطية فقط فكان الـ $R^2$ بتاعه ضعيف جداً، بينما الـ Random Forest بيقدر يتعلم العلاقات المعقدة وغير الخطية (Non-linear Interactions) بين المساحة والموقع والدور ونوع الفرش، وبالتالي حقق دقة عالية جداً ($R^2 = 93.1\%$) وخطأ أقل بكثير (Lower MAE: ₹9.8 Lacs & Lower RMSE).
