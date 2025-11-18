import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# nacteni excel souboru
excel_file = "tlbo_results.xlsx"

if not Path(excel_file).exists():
    print(f"Soubor {excel_file} neexistuje!")
    exit(1)

# nacteni vsech sheetu
xl = pd.ExcelFile(excel_file)
sheet_names = xl.sheet_names

print(f"Nacteno {len(sheet_names)} funkci: {sheet_names}\n")

# priprava dat pro grafy
algorithms = ["DE", "PSO", "SOMA", "FA", "TLBO"]
functions = []
means = {algo: [] for algo in algorithms}
stds = {algo: [] for algo in algorithms}

# nacteni dat z kazdeho sheetu
for sheet in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet, index_col=0)
    functions.append(sheet)

    # ziskani mean a std hodnot (posledni dva radky)
    mean_row = df.loc["Mean"]
    std_row = df.loc["Std. dev"]

    for algo in algorithms:
        means[algo].append(mean_row[algo])
        stds[algo].append(std_row[algo])

# vytvoreni grafu
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle(
    "Porovnani algoritmu na testovacich funkcich", fontsize=16, fontweight="bold"
)

axes = axes.flatten()

# pro kazdou funkci vytvor graf
for idx, func_name in enumerate(functions):
    ax = axes[idx]

    # nacteni dat pro tuto funkci
    df = pd.read_excel(excel_file, sheet_name=func_name, index_col=0)

    # odstraneni mean a std radku pro boxplot
    df_experiments = df.iloc[:-2]

    # boxplot pro vsechny algoritmy
    ax.boxplot(
        [df_experiments[algo].values for algo in algorithms],
        labels=algorithms,
        showmeans=True,
        meanline=True,
    )

    ax.set_title(func_name, fontweight="bold")
    ax.set_ylabel("Fitness hodnota")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("comparison_boxplots.png", dpi=300, bbox_inches="tight")
print("Ulozeno: comparison_boxplots.png")
plt.show()

# graf s prumernymi hodnotami a smerodatnymi odchylkami
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(functions))
width = 0.15

for i, algo in enumerate(algorithms):
    offset = (i - 2) * width
    bars = ax.bar(
        x + offset,
        means[algo],
        width,
        label=algo,
        yerr=stds[algo],
        capsize=3,
        alpha=0.8,
    )

ax.set_xlabel("Testovaci funkce", fontweight="bold")
ax.set_ylabel("Prumerna fitness hodnota", fontweight="bold")
ax.set_title("Porovnani prumernych vysledku algoritmu", fontweight="bold", fontsize=14)
ax.set_xticks(x)
ax.set_xticks_labels(functions, rotation=45, ha="right")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("comparison_means.png", dpi=300, bbox_inches="tight")
print("Ulozeno: comparison_means.png")
plt.show()

# tabulka s mean hodnotami
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis("tight")
ax.axis("off")

# priprava dat pro tabulku
table_data = []
for func in functions:
    row = [func]
    for algo in algorithms:
        idx = functions.index(func)
        row.append(f"{means[algo][idx]:.2e}")
    table_data.append(row)

# pridani radku s prumerem pres vsechny funkce
avg_row = ["Prumer"]
for algo in algorithms:
    avg_row.append(f"{np.mean(means[algo]):.2e}")
table_data.append(avg_row)

table = ax.table(
    cellText=table_data,
    colLabels=["Funkce"] + algorithms,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# zvyrazneni header
for i in range(len(algorithms) + 1):
    table[(0, i)].set_facecolor("#40466e")
    table[(0, i)].set_text_props(weight="bold", color="white")

# zvyrazneni posledniho radku
for i in range(len(algorithms) + 1):
    table[(len(table_data), i)].set_facecolor("#d3d3d3")
    table[(len(table_data), i)].set_text_props(weight="bold")

plt.title("Tabulka prumernych vysledku", fontweight="bold", fontsize=14, pad=20)
plt.savefig("comparison_table.png", dpi=300, bbox_inches="tight")
print("Ulozeno: comparison_table.png")
plt.show()

# convergence graf pro vybranou funkci (napr. Sphere)
selected_func = sheet_names[0]
print(f"\nKonvergencni graf pro funkci: {selected_func}")

df = pd.read_excel(excel_file, sheet_name=selected_func, index_col=0)
df_experiments = df.iloc[:-2]

fig, ax = plt.subplots(figsize=(12, 7))

for algo in algorithms:
    values = df_experiments[algo].values
    ax.plot(
        range(1, len(values) + 1),
        values,
        marker="o",
        markersize=3,
        label=algo,
        alpha=0.7,
    )

ax.set_xlabel("Cislo experimentu", fontweight="bold")
ax.set_ylabel("Fitness hodnota", fontweight="bold")
ax.set_title(
    f"Vysledky jednotlivych experimentu - {selected_func}",
    fontweight="bold",
    fontsize=14,
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"convergence_{selected_func}.png", dpi=300, bbox_inches="tight")
print(f"Ulozeno: convergence_{selected_func}.png")
plt.show()

print("\nVizualizace dokoncena!")
