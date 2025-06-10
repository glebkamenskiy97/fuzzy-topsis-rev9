
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Fuzzy TOPSIS App", layout="wide")
st.title("Приложение для поддержки принятия решения по выбору рационального варианта верхнего строения морской нефтегазопромысловой платформы")

# --- Ввод данных ---
st.header("Ввод исходных данных")
num_alternatives = st.number_input("Количество альтернатив", min_value=2, value=3)
criteria = ["ЧДД (млн руб.)", "ВНД (%)", "Масса ВСП (т)", "Срок строительства ВСП (мес.)", "Пиковая мощность (МВт)", "Выбросы CO2 (т)"]
benefit_criteria = [True, True, False, False, False, False]

weights_linguistic = {
    "Очень низкая": (0.0, 0.0, 0.1),
    "Низкая": (0.0, 0.1, 0.3),
    "Средняя": (0.2, 0.5, 0.8),
    "Высокая": (0.7, 0.9, 1.0),
    "Очень высокая": (0.9, 1.0, 1.0)
}

st.subheader("Выбор весов критериев (лингвистические оценки)")
weights = []
for crit in criteria:
    sel = st.selectbox(f"Вес критерия '{crit}'", list(weights_linguistic.keys()), index=2)
    weights.append(weights_linguistic[sel])
weights = np.array(weights)

st.subheader("Ввод оценок альтернатив (таблица: l,m,u)")
alt_labels = [f"Альт {i+1}" for i in range(num_alternatives)]
rows = []
for alt in alt_labels:
    for crit in criteria:
        rows.append({"Альтернатива": alt, "Критерий": crit, "Оценка (l,m,u)": "0,0,0"})

data_input = pd.DataFrame(rows)
data_input = st.data_editor(data_input, num_rows="fixed")

# Проверка: наличие пустых или некорректных ячеек TFN
missing_entries = data_input[
    data_input["Оценка (l,m,u)"].apply(
        lambda x: not isinstance(x, str) or x.strip() == '' or x.strip().lower() in ['nan', 'none']
    )
]
if not missing_entries.empty:
    st.warning("⚠️ Обнаружены незаполненные или некорректные TFN-оценки:")
    for _, row in missing_entries.iterrows():
        st.error(f"❌ Альтернатива: {row['Альтернатива']} | Критерий: {row['Критерий']} — значение отсутствует")
    for idx in missing_entries.index:
        data_input.at[idx, "Оценка (l,m,u)"] = "0,0,0"
    st.info("⚠️ Пустые значения TFN заменены на 0,0,0 автоматически.")

# Проверка: соответствие количества строк
expected_rows = num_alternatives * len(criteria)
if len(data_input) != expected_rows:
    st.error(f"Ошибка: ожидается {expected_rows} строк (альтернатива × критерий), но получено {len(data_input)}. Проверьте ввод данных.")
    st.stop()

# Формирование массива оценок
valid_input = True
data = []
for alt in alt_labels:
    row = []
    for crit in criteria:
        val = data_input[(data_input["Альтернатива"] == alt) & (data_input["Критерий"] == crit)]["Оценка (l,m,u)"].values[0]
        try:
            l, m, u = map(float, val.strip().split(","))
            if l > u:
                st.info(f"⚠️ Подозрительная TFN для {alt}, {crit}: левый конец больше правого (l > u)")
            row.append((l, m, u))
        except:
            st.error(f"Неверный формат для {alt}, {crit}. Ожидается: l,m,u")
            row.append((0.0, 0.0, 0.0))
            valid_input = False
    data.append(row)
data = np.array(data)

if data.shape[1] != len(criteria):
    st.error(f"Ошибка: количество столбцов в массиве данных ({data.shape[1]}) не совпадает с числом критериев ({len(criteria)}).")
    st.stop()

# --- Fuzzy TOPSIS функции ---
def normalize(data, benefit):
    norm = [[] for _ in range(data.shape[0])]  # для каждой альтернативы
for j in range(data.shape[1]):  # по каждому критерию
    col = data[:, j]
    if benefit[j]:
        max_val = max([u for l, m, u in col])
        norm_col = [(l/max_val, m/max_val, u/max_val) if max_val != 0 else (0, 0, 0) for l, m, u in col]
    else:
        min_val = min([l for l, m, u in col if l > 0]) if any(l > 0 for l, _, _ in col) else 1
        norm_col = [(min_val/u if u != 0 else 0, min_val/m if m != 0 else 0, min_val/l if l != 0 else 0) for l, m, u in col]
    for i, val in enumerate(norm_col):
        norm[i].append(val)
return np.array(norm)

def weighted_fuzzy_decision(norm_data, weights):
    return np.array([[(r[0]*w[0], r[1]*w[1], r[2]*w[2]) for r, w in zip(row, weights)] for row in norm_data])

def ideal_solutions(weighted_data):
    pis = [(max(weighted_data[:, j, 2]), max(weighted_data[:, j, 1]), max(weighted_data[:, j, 0])) for j in range(weighted_data.shape[1])]
    nis = [(min(weighted_data[:, j, 0]), min(weighted_data[:, j, 1]), min(weighted_data[:, j, 2])) for j in range(weighted_data.shape[1])]
    return pis, nis

def distance(a, b):
    return np.sqrt((1/3)*((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))

def closeness(weighted_data, pis, nis):
    cc = []
    for i in range(weighted_data.shape[0]):
        d_pos = np.sqrt(sum([distance(weighted_data[i, j], pis[j])**2 for j in range(weighted_data.shape[1])]))
        d_neg = np.sqrt(sum([distance(weighted_data[i, j], nis[j])**2 for j in range(weighted_data.shape[1])]))
        cc.append(d_neg / (d_pos + d_neg) if (d_pos + d_neg) != 0 else np.nan)
    return cc

# --- Визуализация ---
st.subheader("Выбор визуализаций")
show_table = st.checkbox("Показать таблицу", value=True)
show_bar = st.checkbox("Показать столбчатый график", value=True)
show_tfn = st.checkbox("Показать TFN графики", value=True)
show_radar = st.checkbox("Показать радар-график", value=True)

if st.button("Выполнить расчет и визуализацию") and valid_input:
    norm_data = normalize(data, benefit_criteria)
    if norm_data.shape[1] != len(criteria):
        st.error(f"Ошибка: после нормализации данных получено {norm_data.shape[1]} критериев, ожидается {len(criteria)}. Проверьте ввод данных.")
        st.stop()

    weighted_data = weighted_fuzzy_decision(norm_data, weights)
    pis, nis = ideal_solutions(weighted_data)
    cc_scores = closeness(weighted_data, pis, nis)

    if len(cc_scores) != len(alt_labels):
        st.error("Ошибка: количество коэффициентов близости не соответствует числу альтернатив.")
    else:
        results_df = pd.DataFrame({"Альтернатива": alt_labels, "Коэффициент близости": cc_scores})
        if results_df["Коэффициент близости"].isnull().any():
            st.error("Ошибка: один или несколько коэффициентов близости не определены (NaN). Проверьте ввод данных.")
        else:
            results_df["Ранг"] = results_df["Коэффициент близости"].rank(ascending=False).astype(int)
            results_df = results_df.sort_values("Ранг")
            if show_table:
                st.dataframe(results_df)
            if show_bar:
                fig = px.bar(results_df, x="Альтернатива", y="Коэффициент близости", color="Альтернатива")
                st.plotly_chart(fig)
            if show_tfn:
                for j, crit in enumerate(criteria):
                    fig, ax = plt.subplots()
                    for i in range(num_alternatives):
                        tri = data[i, j]
                        ax.plot([tri[0], tri[1], tri[2]], [0, 1, 0], label=f"Альт {i+1}")
                    ax.set_title(f"TFN по критерию: {crit}")
                    ax.legend()
                    st.pyplot(fig)
            if show_radar:
                radar_df = pd.DataFrame(columns=["Альтернатива"] + criteria)
                for i in range(num_alternatives):
                    row = [alt_labels[i]]
                    for j in range(norm_data.shape[1]):
                        try:
                            row.append(norm_data[i, j][1])
                        except Exception as e:
                            st.warning(f"⚠️ Ошибка в данных: Альт {i+1}, критерий {j}: {e}")
                            row.append(np.nan)
                    radar_df.loc[len(radar_df)] = row
                fig = px.line_polar(radar_df, r=criteria, theta=criteria, line_close=True, color=radar_df["Альтернатива"])
                st.plotly_chart(fig)
