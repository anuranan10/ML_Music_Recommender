# ML Music Recommender

A basic Machine Learning-based music recommender system that predicts a user's preferred music genre based on demographic data. This project demonstrates the end-to-end ML pipeline — from data preprocessing to model training and visualization using decision trees.

---

## Project Structure

```
.
├── HelloWorld.ipynb           # Jupyter notebook with data loading, preprocessing, model training, and predictions
├── music.csv                  # Dataset containing user demographics and corresponding music preferences
├── musics-recommender.joblib # Trained model serialized using joblib
├── music-recommender.dot     # Decision tree model exported as a DOT file
└── vgsales.csv                # Unused file (possibly included by mistake)
```

---

## Description

This recommender predicts a music genre using a decision tree classifier trained on user attributes like:
- **Age**
- **Gender**

Target labels include:
- Classical
- Acoustic
- Dance
- HipHop
- Jazz

The model is visualized as a decision tree using the `.dot` format, allowing you to inspect the learned logic.

---

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ML_Music_Recommender
   ```

2. **Install dependencies**
   You’ll need:
   - Python 3.8+
   - `pandas`
   - `scikit-learn`
   - `joblib`
   - `graphviz` (for visualizing the tree)

   Use pip:
   ```bash
   pip install pandas scikit-learn joblib graphviz
   ```

3. **Run the notebook**
   Launch the notebook in Jupyter:
   ```bash
   jupyter notebook HelloWorld.ipynb
   ```

---

## Model Visualization

The decision tree is saved as a Graphviz `.dot` file (`music-recommender.dot`). You can render it using Graphviz:

```bash
dot -Tpng music-recommender.dot -o tree.png
```

Example Tree Logic:
- If age > 30.5 → Classical
- If age ≤ 25.5 and gender = Male → Dance
- If age ≤ 25.5 and gender = Female → HipHop
(And so on...)

---

## Model Inference

You can load the pre-trained model using:

```python
from joblib import load
model = load("musics-recommender.joblib")
prediction = model.predict([[age, gender]])  # Example input
```

---

## Technologies Used

- **Language:** Python
- **Libraries:** pandas, scikit-learn, joblib
- **Model:** Decision Tree Classifier
- **Visualization:** Graphviz

---

## Acknowledgments

This project is an educational example for practicing basic ML concepts like:
- Supervised classification
- Decision tree logic
- Model serialization and visualization

---

## TODO

- Expand dataset with more features (e.g., listening history)
- Integrate with a front-end UI
- Implement alternative classifiers (e.g., Random Forest, SVM)
