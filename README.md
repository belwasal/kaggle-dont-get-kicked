# 🚗 Car Defect Classification

Классификация автомобилей на вторичном рынке для выявления машин с повышенным риском скрытых дефектов.

Датасет: [Don't Get Kicked!](https://www.kaggle.com/competitions/DontGetKicked/overview) (Kaggle).  
Цель — построить аналитический классификационный пайплайн для оценки факторов, влияющих на риск покупки проблемного автомобиля.

---

## Структура репозитория

* **`src/`** — директория с модулями кастомных реализаций (алгоритмы классификации, ансамбли, метрики).
* **`data/`** — исходные данные (train.csv)
* **`ML4_Classification.ipynb`** — Линейные и вероятностные модели классификации, расчет метрик.
* **`ML5_Tree_based_models.ipynb`** — Решающие деревья, ансамблевые методы и бустинг.
* **`requirements.txt`** — зависимости проекта

---

## Описание ноутбуков

### 1. ML4: Binary Classification, Metrics & Model Implementation
* Обучение стандартных классификаторов (`LogisticRegression`, `GaussianNB`, `KNN`) из библиотеки `sklearn` и их сопоставление с собственными реализациями (SGD-логистическая регрессия, NaiveBayes, KNN).
* Реализация и расчет метрик Precision, Recall, F1, AUC PR.
* Расчет коэффициента Gini на разных выборках и проверка моделей на переобучение.

### 2. ML5: Decision Trees, Random Forests & Gradient Boosting
* Реализация деревьев решений и ансамблей (`DecisionTree`, `RandomForest`, `ExtraTrees`, `GBDT`).
* Сравнение собственных моделей с реализациями `sklearn`, `LightGBM`, `CatBoost` и `XGBoost`.
* Оценка качества по коэффициенту Gini и анализ переобучения.
