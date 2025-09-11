import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_df(dff):
    df= dff.copy()

    

    # mode_val = df[['ca','thal']].mode()[0]
    # df[['ca','thal']].fillna(mode_val, inplace=True)

    # df['ca'].fillna(df['ca'].mode()[0], inplace=True)
    # df['thal'].fillna(df['thal'].mode()[0], inplace=True)

    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])

    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)


    df['thal_label'] = df['thal'].map({
    3: 'unknown',
    6: 'fixed_defect',
    7: 'reversible_defect'
    })

    df = pd.get_dummies(df, columns=['thal_label'], prefix='thal')

    categorical_cols= ['cp', 'slope', 'ca', 'restecg']

    df[categorical_cols] = df[categorical_cols].astype('category')

    
    df['cp'] = df['cp'].cat.reorder_categories(
    new_categories=[1, 2, 3, 4], 
    ordered=True
    )
    
    
    # df['cp'] = df['cp'].cat.set_categories(
    # new_categories=[1, 2, 3, 4], 
    # ordered=True
    # )

    df['slope'] = df['slope'].cat.reorder_categories(
        new_categories=[1, 2, 3], 
        ordered=True
    )
    
    # df['slope'] = df['slope'].cat.set_categories(
        # new_categories=[1, 2, 3], 
        # ordered=True
    # )
    
    df['ca'] = df['ca'].cat.reorder_categories(
        new_categories=[0, 1, 2, 3], 
        ordered=True
    )
    
    # df['ca'] = df['ca'].cat.set_categories(
        # new_categories=[0, 1, 2, 3], 
        # ordered=True
    # )
    
    
    
    df['restecg'] = df['restecg'].cat.reorder_categories(
        new_categories=[0, 1, 2], 
        ordered=True
    )
    
    # df['restecg'] = df['restecg'].cat.set_categories(
        # new_categories=[0, 1, 2], 
        # ordered=True
    # )


    df['target_binary'] = (df['target'] > 0).astype(int)

    df['target_binary'] = df['target_binary'].astype('category')
    df['target'] = df['target'].astype('int8')  


    #Feature and target selection 
    features= df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                    'thal_unknown', 'thal_fixed_defect', 'thal_reversible_defect']]

    target= df['target_binary']


    #Data Splitting 
    X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
    )

    #Scaling 
    scaler = StandardScaler()
    numerical_cols = ['thalach', 'oldpeak']

    
  
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    final_features = [
    'oldpeak',
    'ca',
    'cp',
    'thalach',
    'exang',
    'slope',
    'sex',
    'thal_unknown',
    'thal_reversible_defect'
]

    
    # Extract X and y with final features
    X_train_final = X_train[final_features]
    X_test_final = X_test[final_features]
    
    print("Preprocessing Done!")
    return X_train_final, X_test_final, y_train, y_test, scaler, final_features



def preprocess_single_input(df_raw, scaler, final_features):
    """
    Preprocess a single input row for prediction.
    Assumes scaler and final_features are already fitted/saved from training.
    """
    df = df_raw.copy()

    # df['ca'] = df['ca'].fillna(df['ca'].mode()[0] if not df['ca'].mode().empty else 0)
    # df['thal'] = df['thal'].fillna(df['thal'].mode()[0] if not df['thal'].mode().empty else 3)

    df['ca'] = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)

    # Map thal and create dummies
    df['thal_label'] = df['thal'].map({
        3: 'unknown',
        6: 'fixed_defect',
        7: 'reversible_defect'
    })

    df_dummies = pd.get_dummies(df['thal_label'], prefix='thal')
    df = pd.concat([df, df_dummies], axis=1)

    # Ensure all dummy columns exist
    for col in ['thal_unknown', 'thal_fixed_defect', 'thal_reversible_defect']:
        if col not in df.columns:
            df[col] = 0

    # Set categorical types and categories
    categorical_cols = ['cp', 'slope', 'ca', 'restecg']
    df[categorical_cols] = df[categorical_cols].astype('category')

    df['cp'] = df['cp'].cat.set_categories([1, 2, 3, 4], ordered=True)
    df['slope'] = df['slope'].cat.set_categories([1, 2, 3], ordered=True)
    df['ca'] = df['ca'].cat.set_categories([0, 1, 2, 3], ordered=True)
    df['restecg'] = df['restecg'].cat.set_categories([0, 1, 2], ordered=True)

    # Create target_binary (dummy)
    df['target_binary'] = (df['target'] > 0).astype(int)

    # Select features
    features = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                   'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                   'thal_unknown', 'thal_fixed_defect', 'thal_reversible_defect']].copy()

    # Scale using pre-fitted scaler
    numerical_cols = ['thalach', 'oldpeak']
    features[numerical_cols] = scaler.transform(features[numerical_cols])

    # Select final features
    X_final = features[final_features]

    return X_final


