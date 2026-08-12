# PhysioSentinel AI v15.5.4 · Matriz definitiva de Control Autonómico

Esta versión parte de v15.5.3 y mantiene intactas las pestañas, persistencia PostgreSQL/Supabase, modelo XGBoost, capa probabilística calibrada, 70/15/15, Walk-Forward, selección inteligente de características y resto de cálculos.

## Cambio exclusivo: pestaña 10 · Control autonómico

Se sustituye la primera matriz V/S/L/C por una matriz auditable V/S/L/C/R/Q:

- **V · Vagal:** RMSSD, SD1, HF_DOM_PCT, pNN50, consenso HF (HF/HF_LS/HF_AR/HF_WAV_MEAN) e IDX_Vagal con peso reducido.
- **S · Alerta/barorreflejo:** DFA_alpha1, LF_DOM_PCT, MeanHR, LF_HF contextual, consenso LF e IDX_Rigidez con peso reducido. LF/LF_HF no se presentan como simpático puro.
- **L · Regulación lenta:** VLF_DOM_PCT, consenso VLF, DFA_alpha2, Hurst e IDX_Regulacion_Lenta con peso reducido.
- **C · Complejidad/movilidad:** SampEn, DispEn, D2, Lyapunov_LLE, WAV_ENTROPY_GLOBAL, WAV_TRANSITIONS_PER_MIN, HVG small-world/clustering, resúmenes MSE/MDE y RQA (DET, Lmax/N), más tipo de atractor como contexto estructural.
- **R · Reserva/amplitud:** SDNN, SD2, consenso TOTAL e IDX_Amplitud con peso reducido.
- **Q · Calidad/contexto:** N_RRi, Duration_s, Artefactos_pct e IDX_Confianza; Q modula la confianza pero no determina el estado autonómico.

## Control de redundancia

- HF, LF y VLF se combinan entre Welch, Lomb-Scargle, AR y wavelet como un único consenso por banda, evitando cuatro votos independientes.
- TOTAL se agrega del mismo modo.
- MSE1–20 y MDE1–20 se resumen en tres bloques (1–5, 6–10, 11–20), evitando 40 votos correlacionados.
- Los índices compuestos reciben menor peso porque ya contienen varias métricas originales.

## Interpretación no monotónica

DFA_alpha1, DFA_alpha2, Hurst, Lyapunov, entropías, MSE/MDE y métricas de red no se interpretan como “más = mejor”. La clasificación usa desviación longitudinal intra-paciente y reglas fisiológicas específicas. DFA_alpha1 <0.5 se reserva como señal de posible desacoplamiento cuando existe discordancia; valores >1.0–1.15 sólo apoyan el vector de alerta si convergen con otras variables.

## Reglas A–F

Se mantienen las seis formas operativas: desacoplamiento, dos coactivaciones, reciprocidad, retirada vagal y retirada simpática. Las reglas estrictas se enriquecen con el nuevo vector de reserva R y con calidad Q. Si no se cumple una regla completa, se muestra la forma de mayor compatibilidad con confianza reducida.

## Visualización

La pestaña muestra:

- matriz completa de variables/pesos/función;
- vectores V/S/L/C/R en Z intra-paciente;
- Q de calidad/contexto;
- consensos HF/LF/VLF/TOTAL;
- resúmenes MSE/MDE;
- justificación numérica;
- interpretación fisiológica y coste alostático;
- implicación rehabilitadora;
- evolución cronológica y CSV exportable.
