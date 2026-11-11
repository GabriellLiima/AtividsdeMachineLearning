"""

O script:
1. carrega o heart.csv e identifica a coluna alvo (HeartDisease ou target);
2. remove linhas duplicadas (a versão do Kaggle tem muitas) para evitar vazamento
   entre treino e teste;
3. ajusta uma Regressão Logística (padronização + one-hot);
4. interpreta os coeficientes (coeficiente e razão de chances);
5. avalia com acurácia, precisão, recall e F1-score;
6. salva o modelo (modelo_heart.joblib), os coeficientes (coeficientes.csv)
   e as métricas (metricas.json).
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ALVOS_POSSIVEIS = ["HeartDisease", "target"]

# Na versão numérica do dataset (cp, restecg, slope, thal) são códigos de
# categorias, não quantidades; por isso entram como categóricas.
CODIGOS_CATEGORICOS = ["cp", "restecg", "slope", "thal"]


def carregar(caminho):
    df = pd.read_csv(caminho)
    alvo = next((c for c in ALVOS_POSSIVEIS if c in df.columns), None)
    if alvo is None:
        raise SystemExit(f"Coluna alvo não encontrada. Esperado: {ALVOS_POSSIVEIS}")
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Linhas: {n0} -> {len(df)} após remover {n0 - len(df)} duplicadas")
    print(f"Valores ausentes: {int(df.isna().sum().sum())}")
    print(f"Distribuição do alvo:\n{df[alvo].value_counts(normalize=True).round(3)}\n")
    return df, alvo


def separar_colunas(X):
    categoricas = [c for c in X.columns
                   if X[c].dtype == object or c in CODIGOS_CATEGORICOS]
    resto = [c for c in X.columns if c not in categoricas]
    # binárias 0/1 (sexo, glicemia, angina) não são padronizadas: assim o
    # coeficiente compara diretamente 1 contra 0.
    binarias = [c for c in resto if set(X[c].unique()) <= {0, 1}]
    numericas = [c for c in resto if c not in binarias]
    return numericas, binarias, categoricas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="heart.csv")
    args = ap.parse_args()

    df, alvo = carregar(args.csv)
    X, y = df.drop(columns=[alvo]), df[alvo]
    numericas, binarias, categoricas = separar_colunas(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)

    prep = ColumnTransformer([
        ("num", StandardScaler(), numericas),
        ("bin", "passthrough", binarias),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricas),
    ])
    modelo = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    modelo.fit(X_tr, y_tr)

    # --- Avaliação no conjunto de teste ---
    pred = modelo.predict(X_te)
    metricas = {
        "acuracia": accuracy_score(y_te, pred),
        "precisao": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "matriz_confusao": confusion_matrix(y_te, pred).tolist(),
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
    }
    # Com poucos pacientes no teste, uma única divisão é instável; a validação
    # cruzada estratificada (5 folds) dá uma estimativa mais robusta.
    cv = cross_validate(modelo, X, y, scoring=["accuracy", "precision", "recall", "f1"],
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    metricas["validacao_cruzada_5folds"] = {
        k: {"media": float(cv[f"test_{k}"].mean()), "desvio": float(cv[f"test_{k}"].std())}
        for k in ["accuracy", "precision", "recall", "f1"]}

    print("=== MÉTRICAS (conjunto de teste) ===")
    for k in ["acuracia", "precisao", "recall", "f1"]:
        print(f"{k:>9}: {metricas[k]:.4f}")
    print("\nMatriz de confusão [[VN, FP], [FN, VP]]:")
    print(np.array(metricas["matriz_confusao"]))
    print("\n" + classification_report(y_te, pred, digits=3))

    print("=== VALIDAÇÃO CRUZADA (5 folds, média ± desvio) ===")
    for k, v in metricas["validacao_cruzada_5folds"].items():
        print(f"{k:>9}: {v['media']:.3f} ± {v['desvio']:.3f}")
    print()

    # --- Interpretação dos coeficientes ---
    nomes = modelo.named_steps["prep"].get_feature_names_out()
    coef = modelo.named_steps["clf"].coef_[0]
    tab = pd.DataFrame({
        "variavel": [n.split("__", 1)[1] for n in nomes],
        "coeficiente": coef,
        "razao_de_chances": np.exp(coef),
    })
    dp = dict(zip(numericas, modelo.named_steps["prep"].named_transformers_["num"].scale_))
    tab["desvio_padrao"] = tab["variavel"].map(dp)
    tab["rc_por_unidade"] = np.exp(tab["coeficiente"] / tab["desvio_padrao"])
    tab["efeito"] = np.where(tab["coeficiente"] > 0, "aumenta a chance", "reduz a chance")
    tab = tab.reindex(tab["coeficiente"].abs().sort_values(ascending=False).index)
    print("=== COEFICIENTES (ordenados por magnitude) ===")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nLeitura: variáveis contínuas foram padronizadas, então 'razao_de_chances' é "
          "por +1 desvio-padrão ('rc_por_unidade' converte para +1 unidade original); "
          "binárias e dummies comparam com 0 / com a categoria de referência.")

    # --- Persistência ---
    categorias = {c: sorted(X[c].unique().tolist()) for c in categoricas + binarias}
    intervalos = {c: [float(X[c].min()), float(X[c].max())] for c in numericas}
    joblib.dump({
        "modelo": modelo, "alvo": alvo, "numericas": numericas,
        "categoricas": categoricas, "binarias": binarias, "categorias": categorias,
        "intervalos": intervalos, "ordem_colunas": list(X.columns),
    }, "modelo_heart.joblib")
    tab.to_csv("coeficientes.csv", index=False)
    with open("metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("\nSalvos: modelo_heart.joblib, coeficientes.csv, metricas.json")


if __name__ == "__main__":
    main()