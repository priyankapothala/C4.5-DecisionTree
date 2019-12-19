import pandas as pd
import DecisionTreeClassifier


def get_attribute_list(data):
    return [col for col in data.columns]


def attributeValues(data):
    columns = [col for col in data.columns]
    attr_values = {}
    for column in columns:
        attr_values[column] = list(data[column].value_counts().keys())
    return attr_values


def set_datatype(data):
    data_types = {
        "Temp": 'int64',
        "Humidity": 'int64'}
    data.astype(data_types)


def set_categorical_type(data):
    col_list = ['Outlook', 'Wind', 'Decision']
    for col in get_attribute_list(data):
        if col in col_list:
            data[col] = pd.Categorical(data[col])


if __name__ == "__main__":
    data = pd.read_csv('weather.csv', encoding='utf-8')
    columns = [col for col in data.columns if col != 'Decision']
    set_datatype(data)
    set_categorical_type(data)
    attr_values = attributeValues(data)
    majority_label = data['Decision'].value_counts().idxmax()
    root = DecisionTreeClassifier.buildTree(
        data, columns, attr_values, "Decision", majority_label)

    results = []
    for index, row in data.iterrows():
        result = DecisionTreeClassifier.predict_label(row, root)
        results.append(result)
    print(results)
