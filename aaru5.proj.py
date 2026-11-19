            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))
        ])
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Cross-validate candidates, select by CV ROC-AUC, and test once."""
    rows = []
    fitted = {}

    for name, pipeline in models.items():
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # The positive class for this dataset is explicitly set to 'malignant'.
        classes = list(pipeline.named_steps["model"].classes_)
        malignant_index = classes.index("malignant")
        probabilities = pipeline.predict_proba(X_test)[:, malignant_index]
        y_test_binary = (y_test == "malignant").astype(int)

        row = {
            "model": name,
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision_malignant": float(
                precision_score(y_test, predictions, pos_label="malignant", zero_division=0)
            ),
            "test_recall_malignant": float(
                recall_score(y_test, predictions, pos_label="malignant", zero_division=0)
            ),
            "test_f1_malignant": float(
                f1_score(y_test, predictions, pos_label="malignant", zero_division=0)
            ),
            "test_roc_auc_malignant": float(roc_auc_score(y_test_binary, probabilities))
        }
        rows.append(row)
        fitted[name] = pipeline

        with open(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_report.txt",
                  "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, predictions, zero_division=0))
            f.write("\nConfusion matrix (labels: benign, malignant):\n")
            f.write(str(confusion_matrix(y_test, predictions,
                                         labels=["benign", "malignant"])))

    results = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    return results, best_name, best_model


def save_best_model(best_name, best_model, X_test, y_test):
    """Persist the selected model and produce final diagnostic plots."""
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")

    predictions = best_model.predict(X_test)
    classes = list(best_model.named_steps["model"].classes_)
    malignant_index = classes.index("malignant")
    probabilities = best_model.predict_proba(X_test)[:, malignant_index]
    y_test_binary = (y_test == "malignant").astype(int)

    cm = confusion_matrix(y_test, predictions, labels=["benign", "malignant"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["benign", "malignant"],
                yticklabels=["benign", "malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_model_confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test_binary, probabilities)
    plt.title(f"ROC Curve — {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_model_roc_curve.png", dpi=150)
    plt.close()

    metadata = {
        "selected_model": best_name,
        "features": list(X_test.columns),
        "target_labels": ["benign", "malignant"],
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    print("1/5 Loading dataset...")
    X, y = load_data()

    print("2/5 Running exploratory analysis...")
    exploratory_analysis(X, y)

    print("3/5 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("4/5 Training and evaluating models...")
    results, best_name, best_model = evaluate_models(
        build_models(), X_train, X_test, y_train, y_test
    )

    print("5/5 Saving best model and visualizations...")
    save_best_model(best_name, best_model, X_test, y_test)

    print("\nModel comparison:")
    print(results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"All reports, plots, and model saved in: {OUTPUT_DIR.resolve()}")
