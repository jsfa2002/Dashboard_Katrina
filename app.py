        # ── IA1: PREDICCIÓN DE VENTAS (Modelos Múltiples) ───────────────
        with ia1:
            st.markdown(f"<h4 style='color:{N(GOLD)};'>Predicción Avanzada de Ventas</h4>", unsafe_allow_html=True)

            # Selector de modelo y horizonte
            modelo_pred = st.selectbox("🤖 Selecciona el modelo de predicción:", ["Regresión Polinomial (actual)", "XGBoost con Optuna", "LSTM (Deep Learning)"], key="modelo_pred")
            horizonte = st.radio("📅 ¿Cuántos meses deseas predecir?", [1, 2, 3], index=2, horizontal=True, key="horizonte")

            # --- Preparación de Datos para Modelos Avanzados (Lags y Medias Móviles) ---
            # Agrupar ventas por mes
            serie = (df_fe.groupby("mes_nombre", observed=True)["total_venta"].sum().reindex(om_ia).reset_index())
            serie.columns = ["mes", "ventas"]
            serie["t"] = np.arange(len(serie))

            # Crear características de series temporales
            df_features = serie.copy()
            # Lags (ventas de meses anteriores)
            for lag in [1, 2, 3]:
                df_features[f'lag_{lag}'] = df_features['ventas'].shift(lag)
            # Medias móviles
            for window in [2, 3]:
                df_features[f'rolling_mean_{window}'] = df_features['ventas'].rolling(window).mean()
                df_features[f'rolling_std_{window}'] = df_features['ventas'].rolling(window).std()
            # Eliminar filas con NaN generados por lags/rolling
            df_features = df_features.dropna().reset_index(drop=True)

            # --- Regresión Polinomial (Modelo Original) ---
            if modelo_pred == "Regresión Polinomial (actual)":
                # ... (código original de regresión polinomial)
                st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>Modelo base, sirve como punto de comparación.</p>", unsafe_allow_html=True)

            # --- XGBoost con Optuna ---
            elif modelo_pred == "XGBoost con Optuna":
                import xgboost as xgb
                import optuna
                from sklearn.metrics import mean_absolute_error, mean_squared_error

                # Preparar X e y
                feature_cols = [c for c in df_features.columns if c not in ['mes', 'ventas', 't']]
                X = df_features[feature_cols]
                y = df_features['ventas']

                # División temporal (no aleatoria)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                # Optimización de hiperparámetros con Optuna
                def objective(trial):
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    }
                    model = xgb.XGBRegressor(**param, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    return mae

                with st.spinner('Optimizando hiperparámetros con Optuna...'):
                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                    study.optimize(objective, n_trials=30, show_progress_bar=False)

                # Entrenar modelo final con los mejores parámetros
                best_params = study.best_params
                best_model = xgb.XGBRegressor(**best_params, random_state=42)
                best_model.fit(X_train, y_train)

                # --- Predicción Futura ---
                # Obtener el último registro real conocido
                last_known = df_features.iloc[-1:].copy()
                predictions = []
                for i in range(horizonte):
                    # Predecir el siguiente mes
                    pred = best_model.predict(last_known[feature_cols])[0]
                    predictions.append(pred)
                    # Actualizar las características para el siguiente ciclo
                    new_row = last_known.iloc[-1:].copy()
                    # Shift lags: lag_1 = ventas actual, lag_2 = lag_1 anterior, etc.
                    for lag in [3, 2, 1]:
                        new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}'].values[0] if lag > 1 else pred
                    # Actualizar rolling means y stds (simplificado para la demo)
                    # En un caso real, necesitarías mantener un historial más completo
                    new_row['rolling_mean_2'] = (new_row['lag_1'] + pred) / 2
                    new_row['rolling_std_2'] = abs(new_row['lag_1'] - pred) / 2
                    new_row['rolling_mean_3'] = (new_row['lag_2'] + new_row['lag_1'] + pred) / 3
                    new_row['rolling_std_3'] = np.std([new_row['lag_2'].values[0], new_row['lag_1'].values[0], pred])
                    last_known = new_row

                # Mostrar resultados
                # ... (código para mostrar tablas y gráficos)
                st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>✅ Modelo XGBoost optimizado con Optuna (MAE: {study.best_value:.2f})</p>", unsafe_allow_html=True)

            # --- LSTM (Deep Learning) ---
            elif modelo_pred == "LSTM (Deep Learning)":
                from sklearn.preprocessing import MinMaxScaler
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout

                # Preparar datos para LSTM (secuencias)
                scaler = MinMaxScaler()
                scaled_ventas = scaler.fit_transform(serie[['ventas']].values)

                def create_sequences(data, seq_length):
                    X, y = [], []
                    for i in range(len(data) - seq_length):
                        X.append(data[i:i+seq_length])
                        y.append(data[i+seq_length])
                    return np.array(X), np.array(y)

                seq_length = 3  # Usar los últimos 3 meses para predecir el siguiente
                X_seq, y_seq = create_sequences(scaled_ventas, seq_length)

                # Dividir en train/test
                split = int(0.8 * len(X_seq))
                X_train, X_test = X_seq[:split], X_seq[split:]
                y_train, y_test = y_seq[:split], y_seq[split:]

                # Construir modelo LSTM
                model = Sequential([
                    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')

                # Entrenar
                with st.spinner('Entrenando modelo LSTM...'):
                    history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test), verbose=0)

                # Predicción recursiva
                last_sequence = scaled_ventas[-seq_length:].reshape(1, seq_length, 1)
                predictions_scaled = []
                for _ in range(horizonte):
                    pred_scaled = model.predict(last_sequence, verbose=0)
                    predictions_scaled.append(pred_scaled[0, 0])
                    # Actualizar secuencia
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1, 0] = pred_scaled[0, 0]

                predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

                # Mostrar resultados
                # ... (código para mostrar tablas y gráficos)
                st.markdown(f"<p style='color:{N(GRAY)};font-size:12px;'>🧠 Modelo LSTM con 2 capas ocultas (Deep Learning)</p>", unsafe_allow_html=True)
