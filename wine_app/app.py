import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
FEATURE_LABELS = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный диоксид серы',
    'total_sulfur_dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино',
}


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predictions(frame, model, feature_columns, threshold):
    missing = [name for name in feature_columns if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('В CSV нет строк с данными.')
    try:
        X = frame.loc[:, feature_columns].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все признаки вина должны содержать числа.') from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 или 1.')
    result = frame.copy()
    result['probability_good'] = model.predict_proba(X)[:, 1]
    result['prediction'] = (result['probability_good'] >= threshold).astype(int)
    return result


def main():
    st.set_page_config(page_title='Качество вина', page_icon='🍷', layout='wide')
    model = load_model()
    test = load_test_data()
    config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
    feature_columns = list(model.feature_names_in_)
    X_test = test.loc[:, feature_columns]
    y_test = test['target'].to_numpy()
    probabilities = test_proba(model, X_test)

    st.title('Качество вина')
    st.caption('Хорошее вино — экспертная оценка не ниже 7 из 10.')
    threshold = st.sidebar.slider(
        'Порог рекомендации',
        min_value=0.0,
        max_value=1.0,
        value=float(config['threshold']),
        step=0.001,
        format='%.6f',
        key='threshold',
    )
    st.sidebar.caption('Один порог применяется ко всем вкладкам.')

    single_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
        'Оценка одного вина',
        'Модельные метрики',
        'Пакетное предсказание',
        'Что выучила модель',
    ])

    with single_tab:
        values = {}
        columns = st.columns(3)
        numeric_features = [name for name in feature_columns if name != 'is_red']
        for i, name in enumerate(numeric_features):
            with columns[i % 3]:
                values[name] = st.number_input(
                    FEATURE_LABELS.get(name, name),
                    min_value=float(X_test[name].min()),
                    max_value=float(X_test[name].max()),
                    value=float(X_test[name].median()),
                    format='%.5f',
                    key=f'feature_{name}',
                )
        values['is_red'] = int(st.toggle('Красное вино', value=False))
        sample = pd.DataFrame([values], columns=feature_columns)
        probability = float(model.predict_proba(sample)[0, 1])
        st.metric('Вероятность хорошего вина', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Рекомендовать: модель считает вино хорошим.')
        else:
            st.info('Не включать в подборку при текущем пороге.')

    with metrics_tab:
        predictions = (probabilities >= threshold).astype(int)
        precision = metrics.precision_score(y_test, predictions, zero_division=0)
        recall = metrics.recall_score(y_test, predictions, zero_division=0)
        f1 = metrics.f1_score(y_test, predictions, zero_division=0)
        columns = st.columns(3)
        columns[0].metric('Precision', f'{precision:.3f}')
        columns[1].metric('Recall', f'{recall:.3f}')
        columns[2].metric('F1', f'{f1:.3f}')

        cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        figure, axis = plt.subplots(figsize=(5, 4))
        metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['Остальные', 'Хорошие'],
        ).plot(ax=axis, colorbar=False, cmap='Blues')
        axis.set_xlabel('Предсказание')
        axis.set_ylabel('Истинный класс')
        axis.set_title('Матрица ошибок')
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        fpr, tpr, _ = metrics.roc_curve(y_test, probabilities)
        pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, probabilities)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, probabilities):.3f}')
        axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Случайная модель')
        axes[0].scatter(fp / (fp + tn), tp / (tp + fn), color='red', zorder=5, label=f'Порог {threshold:.3f}')
        axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[0].legend()
        axes[1].plot(pr_recall, pr_precision, label=f'PR-AUC = {metrics.average_precision_score(y_test, probabilities):.3f}')
        axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Случайная модель')
        axes[1].scatter(recall, precision, color='red', zorder=5, label=f'Порог {threshold:.3f}')
        axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая', xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[1].legend()
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        st.write(
            'При повышении порога подборка становится меньше: модель пропускает больше хороших вин, '
            'а доля хороших среди рекомендаций обычно растёт. При снижении порога подборка расширяется: '
            'в неё попадает больше хороших вин и больше посредственных. '
            'На конкретной выборке precision может изменяться немонотонно.'
        )
        if not predictions.any():
            st.caption('При этом пороге нет рекомендаций; precision принят равным 0.')

    with batch_tab:
        st.write('Загрузите CSV с 12 признаками вина, без столбца target.')
        st.caption('Нужные столбцы: ' + ', '.join(feature_columns))
        uploaded = st.file_uploader('CSV-файл', type=['csv'])
        if uploaded is not None:
            try:
                frame = pd.read_csv(uploaded)
                result = batch_predictions(frame, model, feature_columns, threshold)
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(f'Не удалось обработать CSV: {exc}')
            else:
                st.dataframe(result, hide_index=True)
                st.download_button(
                    'Скачать предсказания',
                    data=result.to_csv(index=False).encode('utf-8-sig'),
                    file_name='wine_predictions.csv',
                    mime='text/csv',
                )

    with coefficients_tab:
        coefficients = pd.Series(
            model.named_steps['logisticregression'].coef_[0],
            index=feature_columns,
        ).sort_values()
        figure, axis = plt.subplots(figsize=(9, 5))
        colors = np.where(coefficients.to_numpy() >= 0, '#2a9d8f', '#c85555')
        axis.barh([FEATURE_LABELS.get(name, name) for name in coefficients.index], coefficients.to_numpy(), color=colors)
        axis.axvline(0, color='black', linewidth=0.8)
        axis.set_xlabel('Коэффициент после стандартизации')
        axis.set_title('Влияние признаков на решение модели')
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        positive = coefficients[coefficients > 0].sort_values(ascending=False).head(3)
        negative = coefficients[coefficients < 0].head(3)
        st.write('Сильнее всего за хорошее вино: ' + ', '.join(FEATURE_LABELS.get(name, name) for name in positive.index) + '.')
        st.write('Сильнее всего против: ' + ', '.join(FEATURE_LABELS.get(name, name) for name in negative.index) + '.')
        st.caption('Знак коэффициента показывает изменение логарифма шансов при росте признака и фиксированных остальных признаках; это не доказательство причинной связи.')


if __name__ == '__main__':
    main()
