# Credit Card Fraud Detector

## Overview

**What did you build and why?**
Built a logistic regression model to detect fraudulent credit card transactions using a dataset of 284,807 real transactions, of which only 492 were fraud.

**What was the core challenge?**
The core challenge was the class imbalance. The gap between my frauds and normal cases were 284,315 normal vs 492 fraud, which is a massive gap. Due to the large amount of normal data my model trained mostly on normal data learns to call everything normal. This gives 99.8% accuracy but catches no fraud, which is not of any help to the situation of successfully catching frauds. I fixed it with class_weight="balanced" which forced the model to treat fraud cases as equally important despite being rare.

**Why did you choose recall over accuracy?**
I place more importance on recall compared to accuracy as false negatives can be more detrimental than false positives, due to how it actually causes damage that banks need to control through reimbursements and other long process actions, compared to false positives where the situation can be resolved much faster.

**What were your results?**
Of all transactions flagged as fraud, only 6% were actually fraud (precision). However the model successfully caught 92% of all real fraud cases (recall). It missed 8% of real fraud cases. My low F1 of 0.12, comes from my low precision rate, therefore although my model is good at catching fraud, it created too many false alarms.

**What do the top features tell you?**
My model mainly listed the transactions as fraud by the three main features (amount, V1 and V14), therefore the key traits the model looked at to determine whether something was fraud was through unnatural transaction amounts and two other anonymous features such as location, last transaction, and merchant category.

## Tools Used
Python, pandas, scikit-learn, matplotlib, seaborn

## Files
- fraud.py — main model script
- confusion_matrix.png — model evaluation matrix
- feature_importance.png — top 10 features driving fraud detection