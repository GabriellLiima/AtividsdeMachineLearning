"""


O script pergunta cada variável clínica, calcula a probabilidade com o modelo
de Regressão Logística salvo em modelo_heart.joblib e mostra a previsão.
s
"""
import joblib
import pandas as pd

ARQUIVO_MODELO = "modelo_heart.joblib"


def pedir_numero(nome, intervalo):
    minimo, maximo = intervalo
    while True:
        texto = input(f"{nome} (valores no treino: {minimo:g} a {maximo:g}): ").strip().replace(",", ".")
        try:
            valor = float(texto)
        except ValueError:
            print("  Digite um número válido.")
            continue
        if not (minimo <= valor <= maximo):
            aviso = input("  Valor fora da faixa vista no treino; a previsão pode ser menos confiável. Usar mesmo assim? (s/n): ")
            if aviso.strip().lower() != "s":
                continue
        return valor


def pedir_categoria(nome, opcoes):
    texto_opcoes = ", ".join(str(o) for o in opcoes)
    while True:
        texto = input(f"{nome} (opções: {texto_opcoes}): ").strip()
        for o in opcoes:
            if texto == str(o):
                return o
        print("  Escolha uma das opções listadas.")


def main():
    pacote = joblib.load(ARQUIVO_MODELO)
    modelo = pacote["modelo"]

    print("=== Previsão de doença cardíaca (Regressão Logística) ===\n")
    entrada = {}
    for col in pacote["ordem_colunas"]:
        if col in pacote["categorias"]:
            entrada[col] = pedir_categoria(col, pacote["categorias"][col])
        else:
            entrada[col] = pedir_numero(col, pacote["intervalos"][col])

    X = pd.DataFrame([entrada], columns=pacote["ordem_colunas"])
    prob = modelo.predict_proba(X)[0, 1]
    classe = int(prob >= 0.5)

    print("\n--- Resultado ---")
    print(f"Probabilidade estimada de doença cardíaca: {prob:.1%}")
    if classe == 1:
        print("Previsão: PRESENÇA de doença cardíaca (classe 1)")
    else:
        print("Previsão: AUSÊNCIA de doença cardíaca (classe 0)")
    print("Aviso: resultado de um modelo acadêmico, não é diagnóstico médico.")


if __name__ == "__main__":
    main()
