import pandas as pd

def _onehotencode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    vc = list(df[column].value_counts().keys())
    df[column] = df[column].map(lambda x: vc.index(x))
    df = pd.get_dummies(df, columns=[column])
    return df

def onehotencode(df: pd.DataFrame, columns: list[str]):
    for column in columns:
        df = _onehotencode(df, column)
    return df