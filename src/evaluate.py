from .train_model import *

# Évaluer les trois modèles
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("LightGBM", y_test, y_pred_lgb)
evaluate_model("CatBoost", y_test, y_pred_cat)

# Afficher les meilleurs hyperparamètres
print("\nMeilleurs paramètres RF:", grid_rf.best_params_)
print("Meilleurs paramètres LGBM:", grid_lgb.best_params_)
print("Meilleurs paramètres CatBoost:", grid_cat.best_params_)
print("Fin du script")