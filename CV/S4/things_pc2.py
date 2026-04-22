def base_evaluator(X, y_labels, modelo_base, parametros, nombre_modelo, k=5):
    """
    Función oculta que hace el trabajo pesado de Cross Validation
    para no repetir el código (Normalización, Búsqueda de Hyperparams, etc.)
    """
    inicio_tiempo = time.time()

    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n--- Inicializando k-fold (k={k}) para {nombre_modelo} ---")

    y_true_totales, y_pred_totales = [], []
    fold_accs, fold_f1s = [], []
    mejores_params_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalizar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Convertir a DF explícito para LightGBM
        nombres_cols = [f"f_{i}" for i in range(X_train_scaled.shape[1])]
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=nombres_cols)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=nombres_cols)

        mejor_score = -1
        mejor_modelo_fold = None
        mejor_config_fold = None

        # Búsqueda manual de hiperparámetros (Grid Search)
        for config in ParameterGrid(parametros):
            modelo_tmp = modelo_base.set_params(**config)
            modelo_tmp.fit(X_train_scaled, y_train)
            val_preds = modelo_tmp.predict(X_val_scaled)
            f1_val = f1_score(y_val, val_preds, average="macro", zero_division=0)

            if f1_val > mejor_score:
                mejor_score = f1_val
                mejor_modelo_fold = modelo_tmp
                mejor_config_fold = config

        mejores_params_folds.append(mejor_config_fold)

        # Predicción Final de este fold
        mejores_preds = mejor_modelo_fold.predict(X_val_scaled)
        fold_accs.append(accuracy_score(y_val, mejores_preds))
        fold_f1s.append(
            f1_score(y_val, mejores_preds, average="macro", zero_division=0)
        )

        y_true_totales.extend(y_val)
        y_pred_totales.extend(mejores_preds)

    # Calcular promedios
    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    cm = confusion_matrix(y_true_totales, y_pred_totales)
    reporte = classification_report(
        y_true_totales, y_pred_totales, target_names=le.classes_, zero_division=0
    )

    fin_tiempo = time.time()
    tiempo_total = fin_tiempo - inicio_tiempo

    print(
        f"Resultados {nombre_modelo}: Accuracy_CV={mean_acc:.3f} | F1_Macro_CV={mean_f1:.3f} | Tiempo={tiempo_total:.2f}s"
    )

    return {
        "modelo": nombre_modelo,
        "accuracy_medio": mean_acc,
        "f1_score_medio": mean_f1,
        "matriz_confusion": cm,
        "reporte_clases": reporte,
        "etiquetas_raw": le.classes_,
        "mejores_parametros_folds": mejores_params_folds,
        "tiempo_ejecucion_segundos": tiempo_total,
    }


def evaluar_knn(X, y_labels, k_fold=5):
    """Evalúa k-NN buscando el mejor hiperparámetro de vecinos"""
    modelo = KNeighborsClassifier()
    parametros = {"n_neighbors": [3, 5, 7]}
    return base_evaluator(X, y_labels, modelo, parametros, "k-NN", k_fold)


def evaluar_svm(X, y_labels, k_fold=5):
    """Evalúa Support Vector Machine buscando el mejor Kernel"""
    modelo = SVC(random_state=RANDOM_STATE)
    parametros = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10]}
    return base_evaluator(X, y_labels, modelo, parametros, "SVM", k_fold)


def evaluar_lightgbm(X, y_labels, k_fold=5):
    """Evalúa LightGBM iterando sus estimadores (n_estimators)"""
    modelo = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    parametros = {"n_estimators": [100, 200], "max_depth": [3, 6]}
    return base_evaluator(X, y_labels, modelo, parametros, "LightGBM", k_fold)


def evaluar_randomforest(X, y_labels, k_fold=5):
    """Evalúa Random Forest iterando sus estimadores y profundidad"""
    modelo = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    parametros = {"n_estimators": [100, 200], "max_depth": [None, 10]}
    return base_evaluator(X, y_labels, modelo, parametros, "RandomForest", k_fold)
